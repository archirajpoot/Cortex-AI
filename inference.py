"""
Advanced AI Inference Script — CustomerSupportEnv
Hackathon Submission for Meta OpenEnv 2026
Triple-Check Reasoning: Sentiment → Financial → Personalized Resolution
"""

import asyncio
import os
import json
from typing import Dict, Any, List

from openai import OpenAI
from client import CustomerSupportEnv
from models import SupportAction

# ────────────────────────────────────────────────────────
# MANDATORY HACKATHON VARIABLES
# ────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")

# Strict initialization for AST checks
llm_client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"]
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "customer_support_env"


def generate_intelligent_decision(complaint: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Triple-Check Reasoning:
    Step 1 — Sentiment & Policy Analysis
    Step 2 — Financial Impact Analysis
    Step 3 — Personalized Resolution
    """
    name     = complaint.get("customer_name", "Customer")
    tier     = complaint.get("customer_tier", "regular")
    prev     = complaint.get("previous_complaints", 0)
    last     = complaint.get("last_interaction", "None")
    sentiment= complaint.get("sentiment_score", 0.0)
    priority = complaint.get("priority", "medium")
    category = complaint.get("category", "general")
    order_val= complaint.get("estimated_order_value", 0.0)
    text     = complaint.get("text", "")
    ambiguity= complaint.get("ambiguity_level", 0.3)

    budget       = context.get("budget_remaining", 1000.0)
    escalations  = context.get("escalation_count", 0)
    satisfaction = context.get("satisfaction_score", 1.0)

    sl = "Enraged" if sentiment < -0.7 else "Frustrated" if sentiment < -0.3 else "Neutral" if sentiment < 0.1 else "Calm"
    budget_label = "healthy" if budget > 500 else "moderate" if budget > 200 else "tight"
    repeat_note  = f"This is a repeat customer (last issue: {last})." if prev > 0 else "First-time contact."

    prompt = f"""You are an elite Customer Support AI Manager.
You MUST use the Triple-Check Reasoning process below before answering.

=== CUSTOMER PROFILE ===
Name: {name} | Tier: {tier.upper()} | Priority: {priority.upper()} | Category: {category}
Sentiment: {sl} (score: {sentiment:.2f}) | Ambiguity: {ambiguity:.1f}/1.0
Complaint: "{text}"
Order Value: ${order_val:.2f} | {repeat_note}

=== BUSINESS STATE ===
Budget: ${budget:.2f} ({budget_label}) | Escalations Used: {escalations}/4 | Satisfaction: {satisfaction*100:.0f}%

=== AVAILABLE ACTIONS ===
refund | replace | escalate | apologize | ignore | investigate

=== TRIPLE-CHECK REASONING ===

STEP 1 — SENTIMENT & POLICY:
Think: How upset is {name}? (Enraged = churn risk). Does {tier} tier + {priority} priority demand a strong resolution?
Is {repeat_note} boosting their frustration? What does policy say for {category} issues?

STEP 2 — FINANCIAL IMPACT:
Think: What does the action cost vs. budget of ${budget:.0f}? Is churn risk worth the spend?
Are escalations nearly maxed ({escalations}/4)? Will under-resolving hurt satisfaction ({satisfaction*100:.0f}%)?

STEP 3 — PERSONALIZED RESOLUTION:
Think: Given steps 1 and 2, what is the single best action?
Write a warm sentence directly to {name} acknowledging their situation.

Respond ONLY in valid JSON (no markdown fences, no extra text):
{{
    "step1_sentiment": "One clear sentence about {name}'s emotional state, tier impact, and policy direction.",
    "step2_financial": "One clear sentence about cost, budget ({budget_label} at ${budget:.0f}), and churn risk of your decision.",
    "step3_resolution": "One warm sentence addressed directly to {name}, referencing their history and stating the action.",
    "decision": "<refund|replace|escalate|apologize|ignore|investigate>",
    "confidence": 0.85,
    "reasoning": "Brief 1-sentence combined rationale.",
    "urgency_flag": false
}}"""

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        result = json.loads(content)
        # Backfill any missing triple-check keys
        result.setdefault("step1_sentiment",
            f"{name} is {sl}. {tier.upper()} tier + {priority} priority suggests strong resolution needed.")
        result.setdefault("step2_financial",
            f"Budget is {budget_label} (${budget:.0f}). Evaluating cost-benefit of chosen action.")
        result.setdefault("step3_resolution",
            f"{name}, I'm prioritizing your case right now to make this right for you.")
        result.setdefault("decision", "investigate")
        result.setdefault("confidence", 0.6)
        result.setdefault("reasoning", "Decision made via Triple-Check process.")
        result.setdefault("urgency_flag", priority in ("critical", "high"))
        return result
    except Exception:
        # Emergency structured fallback with all triple-check keys
        return {
            "step1_sentiment": f"{name} is {sl}. {tier.upper()} tier with {priority} priority {category} issue requires attention.",
            "step2_financial": f"Budget is {budget_label} at ${budget:.0f}. Investigating is low-cost and avoids budget risk.",
            "step3_resolution": f"{name}, I hear you — I'm investigating this personally to ensure the best outcome for you.",
            "decision": "investigate",
            "confidence": 0.5,
            "reasoning": "Fallback: chose investigate as safe default due to parsing error.",
            "urgency_flag": priority in ("critical", "high"),
        }


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


async def main() -> None:
    # ---------------------------------------------------------
    # Warmup LLM Call for Proxy Check
    # ---------------------------------------------------------
    try:
        llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
    except Exception as e:
        print(f"[DEBUG] Warmup ping failed: {e}", flush=True)

    steps_taken = 0
    correct_count = 0
    score = 0.1
    success = False
    env = None

    try:
        if IMAGE_NAME:
            env = await CustomerSupportEnv.from_docker_image(IMAGE_NAME)
        else:
            SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")
            env = CustomerSupportEnv(base_url=SERVER_URL)

        for task_level in ["easy", "medium", "hard"]:
            TASK_NAME = os.getenv(f"{task_level.upper()}_TASK") or task_level
            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

            rewards: List[float] = []
            steps_taken = 0
            correct_count = 0
            score = 0.1
            success = False

            # Robust Retry Loop for startup race conditions
            result = None
            for attempt in range(15):
                try:
                    result = await env.reset()
                    break
                except Exception as e:
                    if attempt == 0:
                        print(f"[DEBUG] reset() failed: {e}", flush=True)
                    await asyncio.sleep(2)

            if result is None:
                print(f"[DEBUG] Timeout connecting to Env on {task_level} after 30 seconds.", flush=True)
                log_step(step=1, action="investigate", reward=0.0, done=True, error="timeout dummy step")
                log_end(success=False, steps=1, score=0.1, rewards=[0.0])
                continue

            obs = result.observation
            max_steps = getattr(obs, "max_steps", 5)
            done = getattr(result, "done", False)

            for step in range(1, max_steps * 2 + 1):
                if done:
                    break

                complaints = getattr(obs, "complaints", [])
                if not complaints:
                    break

                complaint = complaints[0]

                context = {
                    "budget_remaining":   getattr(obs, "budget_remaining", 1000.0),
                    "escalation_count":   getattr(obs, "escalation_count", 0),
                    "satisfaction_score": getattr(obs, "satisfaction_score", 1.0),
                    "history": getattr(obs, "metadata", {}).get("decision_history", [])
                }

                error_msg = None
                try:
                    ai_output = generate_intelligent_decision(complaint, context)
                except Exception as exc:
                    ai_output = {
                        "step1_sentiment": "Unable to analyse sentiment.",
                        "step2_financial": "Unable to analyse financial impact.",
                        "step3_resolution": "Defaulting to investigate for safety.",
                        "decision": "investigate",
                        "confidence": 0.5,
                        "reasoning": "Exception during AI call.",
                        "urgency_flag": False,
                    }
                    error_msg = str(exc)[:50].replace("\n", " ")

                action_str = ai_output.get("decision", "investigate")

                action = SupportAction(
                    complaint_id=complaint.get("complaint_id", "default"),
                    decision=action_str,
                    confidence=ai_output.get("confidence", 0.7),
                    reasoning=ai_output.get("reasoning", "Triple-Check reasoning applied."),
                    urgency_flag=ai_output.get("urgency_flag", False)
                )

                try:
                    step_result = await env.step(action)
                    obs = step_result.observation
                    reward = step_result.reward or 0.0
                    done = step_result.done
                    if obs.last_step_feedback:
                        if obs.last_step_feedback[0].get("correct"):
                            correct_count += 1
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error_msg = str(exc)[:50].replace("\n", " ")

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            # Official grader formula: 60% accuracy + 40% normalised reward
            if not rewards:
                score = 0.1
            else:
                ratio = correct_count / len(rewards)
                avg_r = sum(rewards) / len(rewards)
                norm_r = (avg_r + 1.0) / 2.0
                score = round(max(0.0, min(1.0, ratio * 0.6 + norm_r * 0.4)), 4)

            thresholds = {"easy": 0.70, "medium": 0.55, "hard": 0.40}
            success = score >= thresholds.get(task_level, 0.5)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as e:
        print(f"[DEBUG] Fatal Error in main loop: {e}", flush=True)
        for t in ["easy", "medium", "hard"]:
            log_start(task=t, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="investigate", reward=0.0, done=True, error="fatal structure crash")
            log_end(success=False, steps=1, score=0.1, rewards=[0.0])

    finally:
        try:
            if env:
                await env.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
