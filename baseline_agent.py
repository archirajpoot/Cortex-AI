"""
Baseline Inference Script — CustomerSupportEnv
==============================================
Runs a deterministic heuristic agent against all three task levels and
reports reproducible scores.

Usage:
    python baseline_agent.py                         # all tasks
    python baseline_agent.py --task easy             # single task
    python baseline_agent.py --episodes 5 --task hard

Requirements:
    Server must be running:  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import requests

SERVER = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")

# ──────────────────────────────────────────────
#  HEURISTIC DECISION ENGINE
# ──────────────────────────────────────────────

CATEGORY_PRIORITY_MAP: Dict[str, Dict[str, str]] = {
    # category → {priority → best_action}
    "billing":   {"critical": "refund", "high": "refund", "medium": "investigate", "low": "investigate"},
    "delivery":  {"critical": "refund", "high": "investigate", "medium": "investigate", "low": "apologize"},
    "quality":   {"critical": "replace", "high": "replace", "medium": "apologize", "low": "apologize"},
    "technical": {"critical": "escalate", "high": "investigate", "medium": "investigate", "low": "apologize"},
    "policy":    {"critical": "escalate", "high": "escalate", "medium": "apologize", "low": "apologize"},
}

SENTIMENT_OVERRIDE: Dict[str, str] = {
    # If sentiment is extreme negative AND certain clues appear, override
    "chargeback": "escalate",
    "businesses": "escalate",
    "no response": "escalate",
    "warranty":    "investigate",
    "acknowledgment": "apologize",
}


def heuristic_agent(complaint: dict, budget_remaining: float, escalation_count: int) -> dict:
    """
    Rule-based heuristic baseline agent.

    Decision logic:
    1. Check for override clues in complaint text
    2. Map category + priority to default action
    3. Adjust for budget (avoid expensive actions if broke)
    4. Assign confidence based on ambiguity level
    """
    text       = complaint.get("text", "").lower()
    category   = complaint.get("category", "delivery")
    priority   = complaint.get("priority", "medium")
    ambiguity  = complaint.get("ambiguity_level", 0.3)
    clues      = [c.lower() for c in complaint.get("context_clues", [])]
    tier       = complaint.get("customer_tier", "regular")

    # Step 1: override check
    decision  = None
    reasoning = ""
    for keyword, action in SENTIMENT_OVERRIDE.items():
        if keyword in text or any(keyword in c for c in clues):
            decision  = action
            reasoning = f"Keyword '{keyword}' detected → override to {action}"
            break

    # Step 2: category + priority map
    if decision is None:
        cat_map  = CATEGORY_PRIORITY_MAP.get(category, {})
        decision = cat_map.get(priority, "apologize")
        reasoning = f"Category '{category}' + priority '{priority}' → {decision}"

    # Step 3: budget guard — avoid refund/replace if budget is critical
    expensive = {"refund", "replace"}
    if decision in expensive and budget_remaining < 50:
        decision  = "escalate"
        reasoning += " [budget override → escalate]"

    # Step 4: escalation cap — don't escalate if already overused
    if decision == "escalate" and escalation_count >= 4:
        decision  = "investigate"
        reasoning += " [escalation cap → investigate]"

    # Step 5: VIP tier — always prefer action over ignore/apologize
    if tier == "vip" and decision in {"ignore", "apologize"}:
        if category in {"billing", "delivery", "quality"}:
            decision  = "investigate"
            reasoning += " [VIP tier upgrade]"

    # Confidence: inversely proportional to ambiguity, boosted if priority=critical
    base_confidence = 1.0 - ambiguity
    if priority == "critical":
        base_confidence = min(1.0, base_confidence + 0.15)
    confidence = round(base_confidence, 2)

    return {
        "complaint_id": complaint["complaint_id"],
        "decision":     decision,
        "confidence":   confidence,
        "reasoning":    reasoning,
        "urgency_flag": priority in {"critical", "high"},
    }


# ──────────────────────────────────────────────
#  GRADERS
# ──────────────────────────────────────────────

def grade_episode(history: list) -> float:
    """
    Score an episode 0.0–1.0.

    Score = proportion of correct/acceptable decisions × avg normalised reward
    """
    if not history:
        score = 0.1
    else:
        correct = sum(1 for h in history if h.get("correct", False))
        ratio   = correct / len(history)
        rewards  = [h.get("reward", 0.0) for h in history]
        avg_r    = sum(rewards) / len(rewards) if rewards else 0.0
        # Normalise avg_r from [-1,1] to [0,1]
        norm_r   = (avg_r + 1.0) / 2.0
        score    = round((ratio * 0.6 + norm_r * 0.4), 4)
    # Clamped securely within strict regex boundaries
    return max(0.001, min(0.999, score))


TASK_GRADERS = {
    "easy": {
        "name": "Easy — Basic Triage",
        "pass_threshold": 0.70,
        "description": "Correct action on 1–2 clear complaints. Threshold: 0.70",
    },
    "medium": {
        "name": "Medium — Operational Support",
        "pass_threshold": 0.55,
        "description": "Handle ambiguous policy/quality complaints. Threshold: 0.55",
    },
    "hard": {
        "name": "Hard — Enterprise Crisis Management",
        "pass_threshold": 0.40,
        "description": "Multi-issue, high-budget-risk scenarios. Threshold: 0.40",
    },
}


# ──────────────────────────────────────────────
#  RUNNER
# ──────────────────────────────────────────────

def run_episode(task_level: str, verbose: bool = True) -> dict:
    """Run one full episode and return structured results."""
    reset_resp = requests.post(f"{SERVER}/reset", timeout=10)
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()
    obs        = reset_data["observation"]

    # task_level is now a direct field on the observation
    actual_level = obs.get("task_level") or task_level
    max_steps    = obs.get("max_steps", 5)

    episode_history   = []
    total_reward      = 0.0
    steps_taken       = 0
    budget_remaining  = obs.get("budget_remaining", 1000.0)
    escalation_count  = obs.get("escalation_count", 0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {obs.get('task_name','?')} [{actual_level.upper()}]")
        desc = obs.get('task_description','').encode('ascii', 'replace').decode()
        print(f"  {desc}")
        print(f"{'='*60}")

    # done is top-level in step responses; use safety cap too
    done = reset_data.get("done", False)
    while not done and steps_taken < max_steps * 3:  # safety cap
        complaints = obs.get("complaints", [])
        if not complaints:
            break

        # Use heuristic agent on each complaint in the batch;
        # submit them one at a time (one step per complaint)
        complaint = complaints[0]

        action_dict = heuristic_agent(complaint, budget_remaining, escalation_count)

        step_resp = requests.post(
            f"{SERVER}/step",
            json={"action": action_dict},
            timeout=10,
        )
        step_resp.raise_for_status()
        step_data        = step_resp.json()
        obs              = step_data["observation"]
        reward           = step_data.get("reward", 0.0)
        done             = step_data.get("done", False)  # top-level 'done'

        budget_remaining = obs.get("budget_remaining", budget_remaining)
        escalation_count = obs.get("escalation_count", escalation_count)
        total_reward    += reward
        steps_taken     += 1

        for fb in obs.get("last_step_feedback", []):
            episode_history.append({"correct": fb.get("correct", False), "reward": reward})
            verdict_raw = fb.get('verdict', '')
            verdict_ascii = verdict_raw.encode('ascii', 'replace').decode()
            if verbose:
                print(f"\n  Step {steps_taken}: [{fb.get('priority','?').upper()}] "
                      f"{fb.get('complaint_text','')[:60]}")
                print(f"    Decision : {fb.get('decision','')} "
                      f"(conf={action_dict['confidence']})")
                print(f"    Verdict  : {verdict_ascii}")
                print(f"    Reward   : {reward:+.4f}")
                rb = fb.get("reward_breakdown", {})
                if rb:
                    print(f"    Breakdown: base={rb.get('weighted_score',0):+.3f}  "
                          f"calib={rb.get('calibration_bonus',0):+.3f}  "
                          f"budget={rb.get('budget_penalty',0):+.3f}")

    score = grade_episode(episode_history)
    grader = TASK_GRADERS.get(actual_level, {})
    passed = score >= grader.get("pass_threshold", 0.5)

    if verbose:
        print(f"\n  -- Episode Summary --")
        print(f"  Steps:         {steps_taken}")
        print(f"  Total Reward:  {total_reward:+.4f}")
        print(f"  Episode Score: {score:.4f} / 1.0")
        print(f"  Threshold:     {grader.get('pass_threshold',0)}")
        result_str = 'PASS' if passed else 'FAIL'
        print(f"  Result:        {result_str}")

    return {
        "task_level":    actual_level,
        "steps":         steps_taken,
        "total_reward":  round(total_reward, 4),
        "score":         score,
        "passed":        passed,
        "threshold":     grader.get("pass_threshold", 0.5),
    }


def run_all(episodes_per_task: int = 3, verbose: bool = True) -> dict:
    """Run multiple episodes across all task levels."""
    results = {}
    for level in ["easy", "medium", "hard"]:
        scores = []
        for ep in range(episodes_per_task):
            if verbose:
                print(f"\n[Episode {ep+1}/{episodes_per_task} for {level}]")
            r = run_episode(level, verbose=verbose)
            scores.append(r["score"])
        avg = round(sum(scores) / len(scores), 4)
        results[level] = {
            "scores":    scores,
            "avg_score": avg,
            "passed":    avg >= TASK_GRADERS.get(level, {}).get("pass_threshold", 0.5),
        }
        if verbose:
            avg = round(sum(scores) / len(scores), 4)
            print(f"\n  * {level.upper()} avg score: {avg:.4f}")

    if verbose:
        print("\n" + "="*60)
        print("  FINAL BASELINE RESULTS")
        print("="*60)
        for level, r in results.items():
            status = 'PASS' if r['passed'] else 'FAIL'
            print(f"  {level.upper():8s} avg={r['avg_score']:.4f}  {status}")
        print("="*60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CustomerSupportEnv Baseline Agent")
    parser.add_argument("--task",     choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--quiet",    action="store_true")
    parser.add_argument("--server",   default="http://127.0.0.1:7860")
    args = parser.parse_args()

    SERVER  = args.server
    verbose = not args.quiet

    # Wait for server
    for attempt in range(10):
        try:
            r = requests.get(f"{SERVER}/health", timeout=3)
            if r.status_code == 200:
                break
        except Exception:
            pass
        print(f"Waiting for server… attempt {attempt+1}/10")
        time.sleep(2)
    else:
        print("ERROR: Could not connect to server.")
        sys.exit(1)

    if args.task == "all":
        results = run_all(episodes_per_task=args.episodes, verbose=verbose)
        print(json.dumps(results, indent=2))
    else:
        result = run_episode(args.task, verbose=verbose)
        print(json.dumps(result, indent=2))
