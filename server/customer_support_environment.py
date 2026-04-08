"""
CustomerSupportEnvironment — OpenEnv Environment
================================================
A real-world customer-support decision-making environment where an AI agent
must triage, prioritise, and resolve customer complaints over multi-step episodes.

Design principles
-----------------
* Multi-step episodes  — each episode spans up to N steps; unresolved complaints
  carry over and accumulate backlog penalties.
* Rich observation space — complaints carry category, sentiment, customer tier,
  order value, and contextual clues that the agent must reason about.
* Dynamic reward shaping — rewards consider correctness, confidence calibration,
  cost vs. customer-satisfaction tradeoffs, priority sensitivity, and long-term
  strategy penalties (e.g. always escalating everything).
* Partial correctness — sub-optimal but defensible actions receive partial credit;
  truly wrong actions lose points.
* Three graded task levels — easy / medium / hard — each with its own complaint
  generator and scoring rubric.

OpenEnv API
-----------
  reset()  → SupportObservation
  step(action: SupportAction)  → SupportObservation
  state    → State
"""

import math
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportAction, SupportObservation
except ImportError:
    from models import SupportAction, SupportObservation


# ──────────────────────────────────────────────────────────────
#  COMPLAINT TEMPLATES
# ──────────────────────────────────────────────────────────────

COMPLAINT_TEMPLATES = {
    "easy": [
        {
            "text": "My order arrived 3 weeks late and I need a full refund.",
            "category": "delivery",
            "correct_actions": ["refund"],
            "acceptable_actions": ["refund", "apologize"],
            "priority": "high",
            "sentiment": -0.7,
            "clues": ["late delivery", "refund request", "extended delay"],
            "cost_map": {"refund": 50, "replace": 40, "escalate": 10, "apologize": 0, "ignore": 0, "investigate": 5},
            "satisfaction_map": {"refund": 0.9, "replace": 0.5, "escalate": 0.4, "apologize": 0.3, "ignore": -0.4, "investigate": 0.2},
        },
        {
            "text": "The product stopped working after just one day. Please send a replacement.",
            "category": "quality",
            "correct_actions": ["replace"],
            "acceptable_actions": ["replace", "refund"],
            "priority": "high",
            "sentiment": -0.6,
            "clues": ["defective product", "one day use", "replacement request"],
            "cost_map": {"refund": 80, "replace": 35, "escalate": 10, "apologize": 0, "ignore": 0, "investigate": 5},
            "satisfaction_map": {"refund": 0.8, "replace": 0.95, "escalate": 0.3, "apologize": 0.2, "ignore": -0.5, "investigate": 0.1},
        },
        {
            "text": "I was double-charged on my credit card. Need immediate resolution.",
            "category": "billing",
            "correct_actions": ["refund"],
            "acceptable_actions": ["refund", "investigate"],
            "priority": "critical",
            "sentiment": -0.8,
            "clues": ["double charge", "billing error", "immediate"],
            "cost_map": {"refund": 60, "replace": 30, "escalate": 15, "apologize": 0, "ignore": 0, "investigate": 8},
            "satisfaction_map": {"refund": 0.95, "replace": 0.2, "escalate": 0.5, "apologize": 0.1, "ignore": -0.9, "investigate": 0.6},
        },
    ],
    "medium": [
        {
            "text": "My package shows delivered but I never received it. It's been 5 days.",
            "category": "delivery",
            "correct_actions": ["investigate", "replace"],
            "acceptable_actions": ["investigate", "replace", "refund"],
            "priority": "high",
            "sentiment": -0.5,
            "clues": ["marked delivered", "not received", "5 days waiting"],
            "cost_map": {"refund": 70, "replace": 50, "escalate": 12, "apologize": 0, "ignore": 0, "investigate": 6},
            "satisfaction_map": {"refund": 0.75, "replace": 0.8, "escalate": 0.4, "apologize": 0.2, "ignore": -0.7, "investigate": 0.85},
            "ambiguity": 0.6,
        },
        {
            "text": "Product is slightly different from the photo but works fine.",
            "category": "quality",
            "correct_actions": ["apologize", "investigate"],
            "acceptable_actions": ["apologize", "investigate", "replace"],
            "priority": "low",
            "sentiment": -0.2,
            "clues": ["cosmetic difference", "functional", "photo mismatch"],
            "cost_map": {"refund": 80, "replace": 55, "escalate": 10, "apologize": 0, "ignore": 0, "investigate": 5},
            "satisfaction_map": {"refund": 0.7, "replace": 0.6, "escalate": 0.2, "apologize": 0.75, "ignore": -0.1, "investigate": 0.7},
            "ambiguity": 0.7,
        },
        {
            "text": "I want to return this item but the 30-day return window just expired yesterday.",
            "category": "policy",
            "correct_actions": ["escalate", "apologize"],
            "acceptable_actions": ["escalate", "apologize", "investigate"],
            "priority": "medium",
            "sentiment": -0.35,
            "clues": ["return request", "policy edge case", "just expired"],
            "cost_map": {"refund": 90, "replace": 60, "escalate": 15, "apologize": 0, "ignore": 0, "investigate": 5},
            "satisfaction_map": {"refund": 0.9, "replace": 0.5, "escalate": 0.7, "apologize": 0.55, "ignore": -0.4, "investigate": 0.5},
            "ambiguity": 0.75,
        },
        {
            "text": "The app keeps crashing whenever I try to checkout. I've lost my cart 3 times.",
            "category": "technical",
            "correct_actions": ["investigate", "apologize"],
            "acceptable_actions": ["investigate", "apologize", "escalate"],
            "priority": "high",
            "sentiment": -0.65,
            "clues": ["app crash", "checkout bug", "repeated issue"],
            "cost_map": {"refund": 0, "replace": 0, "escalate": 15, "apologize": 0, "ignore": 0, "investigate": 8},
            "satisfaction_map": {"refund": 0.3, "replace": 0.3, "escalate": 0.65, "apologize": 0.6, "ignore": -0.6, "investigate": 0.85},
            "ambiguity": 0.3,
        },
    ],
    "hard": [
        {
            "text": "I ordered a laptop for work and it died after 6 months. Warranty is 12 months but the repair centre says physical damage voids it.",
            "category": "quality",
            "correct_actions": ["investigate", "escalate"],
            "acceptable_actions": ["investigate", "escalate", "replace"],
            "priority": "high",
            "sentiment": -0.55,
            "clues": ["warranty dispute", "repair centre conflict", "work dependency", "6 months use"],
            "cost_map": {"refund": 900, "replace": 700, "escalate": 20, "apologize": 0, "ignore": 0, "investigate": 15},
            "satisfaction_map": {"refund": 0.85, "replace": 0.8, "escalate": 0.75, "apologize": 0.2, "ignore": -0.8, "investigate": 0.7},
            "ambiguity": 0.85,
        },
        {
            "text": "I'm a long-time VIP customer. My last 3 orders were fine but this one has a small scratch. I don't need a replacement — just acknowledgment.",
            "category": "quality",
            "correct_actions": ["apologize"],
            "acceptable_actions": ["apologize", "investigate"],
            "priority": "medium",
            "sentiment": -0.15,
            "clues": ["VIP customer", "small issue", "no replacement needed", "acknowledgment only"],
            "cost_map": {"refund": 150, "replace": 120, "escalate": 20, "apologize": 0, "ignore": 0, "investigate": 8},
            "satisfaction_map": {"refund": 0.5, "replace": 0.4, "escalate": 0.3, "apologize": 0.95, "ignore": -0.3, "investigate": 0.6},
            "ambiguity": 0.8,
        },
        {
            "text": "This is my second complaint this month. First one was resolved but now there's a NEW defect. I'm considering a chargeback.",
            "category": "quality",
            "correct_actions": ["refund", "escalate"],
            "acceptable_actions": ["refund", "escalate", "replace"],
            "priority": "critical",
            "sentiment": -0.85,
            "clues": ["repeat complaint", "chargeback threat", "escalating frustration", "second defect"],
            "cost_map": {"refund": 200, "replace": 150, "escalate": 20, "apologize": 0, "ignore": 0, "investigate": 10},
            "satisfaction_map": {"refund": 0.9, "replace": 0.75, "escalate": 0.8, "apologize": 0.1, "ignore": -0.95, "investigate": 0.3},
            "ambiguity": 0.4,
        },
        {
            "text": "I cancelled my subscription but I was still charged last month. I raised a ticket 2 weeks ago — no response.",
            "category": "billing",
            "correct_actions": ["refund", "escalate"],
            "acceptable_actions": ["refund", "escalate", "investigate"],
            "priority": "critical",
            "sentiment": -0.9,
            "clues": ["cancelled subscription", "billing after cancel", "2 week no response", "unresolved ticket"],
            "cost_map": {"refund": 30, "replace": 0, "escalate": 20, "apologize": 0, "ignore": 0, "investigate": 10},
            "satisfaction_map": {"refund": 0.95, "replace": 0.1, "escalate": 0.8, "apologize": 0.15, "ignore": -0.99, "investigate": 0.5},
            "ambiguity": 0.15,
        },
        {
            "text": "I'm filing a complaint on behalf of 12 small businesses in our network — all facing the same delayed shipment. We are collectively losing revenue.",
            "category": "delivery",
            "correct_actions": ["escalate", "investigate"],
            "acceptable_actions": ["escalate", "investigate", "refund"],
            "priority": "critical",
            "sentiment": -0.75,
            "clues": ["corporate complaint", "12 businesses", "systemic issue", "revenue loss"],
            "cost_map": {"refund": 2000, "replace": 1500, "escalate": 25, "apologize": 0, "ignore": 0, "investigate": 20},
            "satisfaction_map": {"refund": 0.7, "replace": 0.5, "escalate": 0.9, "apologize": 0.2, "ignore": -0.95, "investigate": 0.8},
            "ambiguity": 0.5,
        },
    ],
}

CUSTOMER_TIERS = ["vip", "regular", "new"]
TIER_WEIGHTS   = [0.15, 0.60, 0.25]

PRIORITY_MULTIPLIERS = {"critical": 1.6, "high": 1.3, "medium": 1.0, "low": 0.7}
TIER_MULTIPLIERS     = {"vip": 1.4, "regular": 1.0, "new": 0.8}

VALID_DECISIONS = {"refund", "replace", "escalate", "apologize", "ignore", "investigate"}

TASK_CONFIGS = {
    "easy": {
        "name": "Basic Triage",
        "description": (
            "Handle 1–2 clear-cut customer complaints per step. "
            "The correct action is strongly signalled. Max 3 steps."
        ),
        "max_steps": 3,
        "n_complaints_per_step": 1,
        "budget": 500.0,
        "pool": "easy",
    },
    "medium": {
        "name": "Operational Support",
        "description": (
            "Handle 2–3 ambiguous complaints per step over 4 steps. "
            "Some complaints are policy edge-cases requiring nuanced decisions."
        ),
        "max_steps": 4,
        "n_complaints_per_step": 2,
        "budget": 800.0,
        "pool": "medium",
    },
    "hard": {
        "name": "Enterprise Crisis Management",
        "description": (
            "Handle 3–5 high-stakes, ambiguous complaints per step over 5 steps. "
            "Manage budget, prevent chargeback threats, and balance VIP satisfaction "
            "with cost efficiency while carrying over unresolved backlog."
        ),
        "max_steps": 5,
        "n_complaints_per_step": 3,
        "budget": 1500.0,
        "pool": "hard",
    },
}


# ──────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def _generate_complaint(template: dict, tier: str) -> dict:
    """Instantiate a complaint from a template with a fresh UUID and tier."""
    c = dict(template)
    c["complaint_id"]          = str(uuid.uuid4())
    c["customer_tier"]         = tier
    c["days_since_purchase"]   = random.randint(1, 90)
    c["previous_complaints"]   = random.randint(0, 5)
    c["estimated_order_value"] = round(random.uniform(20, 600), 2)
    c["ambiguity"]             = template.get("ambiguity", random.uniform(0.1, 0.5))
    return c


def _calibration_bonus(confidence: float, was_correct: bool) -> float:
    """
    Reward well-calibrated confidence.

    High confidence + correct  → bonus up to +0.15
    Low confidence  + wrong    → small bonus (~+0.05, shows appropriate uncertainty)
    High confidence + wrong    → penalty up to -0.2  (overconfidence)
    """
    if was_correct:
        return 0.15 * confidence
    else:
        return -0.2 * confidence + 0.05 * (1 - confidence)


def _budget_penalty(cost: float, budget_remaining: float) -> float:
    """Penalise decisions that blow the budget."""
    if budget_remaining <= 0:
        return -0.3
    if cost > budget_remaining:
        excess_ratio = (cost - budget_remaining) / max(budget_remaining, 1)
        return -0.2 * min(excess_ratio, 1.0)
    return 0.0


def _strategy_penalty(history: list) -> float:
    """
    Detect bad long-term strategies and penalise them.
    - Always ignoring: -0.15 per excess ignore
    - Always escalating: -0.10 per excess escalation (lazy strategy)
    """
    if len(history) < 3:
        return 0.0
    recent = history[-6:]  # last 6 decisions
    ignores    = sum(1 for d in recent if d.get("decision") == "ignore")
    escalates  = sum(1 for d in recent if d.get("decision") == "escalate")
    penalty    = 0.0
    if ignores >= 4:
        penalty -= 0.15
    if escalates >= 5:
        penalty -= 0.10
    return penalty


# ──────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ──────────────────────────────────────────────────────────────

class CustomerSupportEnvironment(Environment):
    """
    OpenEnv environment: Customer Support Decision Making.

    The agent receives batches of customer complaints each step and must
    decide the optimal resolution action for each one, considering:
      - Category and contextual clues
      - Priority and customer tier
      - Sentiment and ambiguity
      - Budget constraints
      - Long-term satisfaction trends
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._reset_count: int = 0
        self._task_level: str = "medium"
        self._config: dict = {}
        self._active_complaints: List[dict] = []
        self._decision_history: List[dict] = []
        self._cumulative_reward: float = 0.0
        self._satisfaction: float = 1.0
        self._budget_remaining: float = 1000.0
        self._escalation_count: int = 0
        self._backlog: List[dict] = []
        self._last_feedback: List[dict] = []
        self._complaint_pool: List[dict] = []
        self._pool_index: int = 0

    # ─── public API ───────────────────────────────────────────

    def reset(self) -> SupportObservation:
        # Pick task level in round-robin so each call rotates easy→medium→hard
        levels = ["easy", "medium", "hard"]
        self._task_level = levels[self._reset_count % 3]
        self._reset_count += 1

        self._config           = TASK_CONFIGS[self._task_level]
        self._state            = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._cumulative_reward = 0.0
        self._satisfaction      = 1.0
        self._budget_remaining  = self._config["budget"]
        self._escalation_count  = 0
        self._backlog           = []
        self._last_feedback     = []
        self._decision_history  = []

        # Build full complaint pool for this episode
        pool_key = self._config["pool"]
        templates = COMPLAINT_TEMPLATES[pool_key]
        random.shuffle(templates)
        self._complaint_pool = []
        # Repeat pool enough times to fill all steps
        total_needed = self._config["max_steps"] * self._config["n_complaints_per_step"] + 5
        while len(self._complaint_pool) < total_needed:
            for t in templates:
                tier = random.choices(CUSTOMER_TIERS, TIER_WEIGHTS)[0]
                self._complaint_pool.append(_generate_complaint(t, tier))
        self._pool_index = 0

        # Draw first batch
        self._active_complaints = self._draw_complaints()

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: SupportAction) -> SupportObservation:
        if not self._active_complaints:
            self.reset()

        self._state.step_count += 1

        # ── Process the single action against its targeted complaint ──
        complaint = self._find_complaint(action.complaint_id)
        if complaint is None:
            # Graceful: unknown id — take first unresolved complaint
            complaint = self._active_complaints[0] if self._active_complaints else None

        step_reward = 0.0
        feedback    = []

        if complaint:
            fb, r = self._evaluate_decision(action, complaint)
            feedback.append(fb)
            step_reward += r
            # Remove resolved complaint
            self._active_complaints = [
                c for c in self._active_complaints
                if c["complaint_id"] != complaint["complaint_id"]
            ]
            # Record history
            self._decision_history.append({
                "step":      self._state.step_count,
                "decision":  action.decision,
                "complaint": complaint["category"],
                "reward":    r,
                "correct":   fb["correct"],
            })

        # ── Carry-over backlog penalty ──────────────────────────────
        backlog_penalty = len(self._backlog) * -0.05
        step_reward += backlog_penalty

        # ── Long-term strategy penalty ──────────────────────────────
        strategy_penalty = _strategy_penalty(self._decision_history)
        step_reward += strategy_penalty

        # ── Normalise reward to [0, 1] roughly ─────────────────────
        step_reward = round(max(-1.0, min(1.0, step_reward)), 4)
        self._cumulative_reward = round(self._cumulative_reward + step_reward, 4)

        self._last_feedback = feedback

        # ── Determine done ──────────────────────────────────────────
        max_steps  = self._config.get("max_steps", 5)
        done       = False
        if not self._active_complaints:
            # Move any remaining backlog in and draw new batch
            if self._backlog:
                self._active_complaints = self._backlog[:self._config["n_complaints_per_step"]]
                self._backlog = self._backlog[self._config["n_complaints_per_step"]:]
            elif self._state.step_count >= max_steps:
                done = True
            else:
                self._active_complaints = self._draw_complaints()

        if self._state.step_count >= max_steps:
            # Push remaining to backlog penalty and end
            leftover_penalty = len(self._active_complaints) * -0.08
            step_reward      = round(step_reward + leftover_penalty, 4)
            done = True

        return self._build_observation(reward=step_reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ─── internal helpers ─────────────────────────────────────

    def _draw_complaints(self) -> List[dict]:
        n = self._config.get("n_complaints_per_step", 1)
        batch = []
        for _ in range(n):
            if self._pool_index < len(self._complaint_pool):
                batch.append(self._complaint_pool[self._pool_index])
                self._pool_index += 1
        return batch

    def _find_complaint(self, complaint_id: str) -> Optional[dict]:
        for c in self._active_complaints:
            if c["complaint_id"] == complaint_id:
                return c
        return None

    def _evaluate_decision(
        self, action: SupportAction, complaint: dict
    ) -> Tuple[dict, float]:
        """
        Core reward calculation for a single decision.

        Returns (feedback_dict, reward_float).
        Reward components:
          1. Correctness:       +0.5 (correct) / +0.2 (acceptable) / -0.3 (wrong)
          2. Priority weight:   multiplies correctness reward
          3. Tier weight:       multiplies again for VIP sensitivity
          4. Confidence calib:  reward/penalise overconfident wrong answers
          5. Budget penalty:    penalise blowing the budget
          6. Satisfaction delta: track customer satisfaction index
        """
        decision = action.decision.lower().strip()
        if decision not in VALID_DECISIONS:
            decision = "ignore"   # normalise unknown

        correct_actions    = complaint.get("correct_actions", [])
        acceptable_actions = complaint.get("acceptable_actions", [])
        cost_map           = complaint.get("cost_map", {})
        sat_map            = complaint.get("satisfaction_map", {})

        is_correct    = decision in correct_actions
        is_acceptable = decision in acceptable_actions
        tier          = complaint.get("customer_tier", "regular")
        priority      = complaint.get("priority", "medium")

        priority_mult = PRIORITY_MULTIPLIERS.get(priority, 1.0)
        tier_mult     = TIER_MULTIPLIERS.get(tier, 1.0)

        # Base correctness score
        if is_correct:
            base = 0.50
            verdict = "✅ Optimal"
        elif is_acceptable:
            base = 0.20
            verdict = "⚡ Acceptable"
        else:
            base = -0.30
            verdict = "❌ Wrong"

        # Apply weights
        weighted = base * priority_mult * tier_mult

        # Calibration bonus/penalty
        calib = _calibration_bonus(action.confidence, is_correct or is_acceptable)

        # Budget impact
        cost    = cost_map.get(decision, 0)
        budget_p = _budget_penalty(cost, self._budget_remaining)
        self._budget_remaining = max(0.0, self._budget_remaining - cost)

        # Satisfaction update
        sat_delta = sat_map.get(decision, 0.0)
        self._satisfaction = round(
            max(0.0, min(1.0, self._satisfaction + sat_delta * 0.1)), 4
        )
        if decision == "escalate":
            self._escalation_count += 1

        total = round(weighted + calib + budget_p, 4)

        feedback = {
            "complaint_id":   complaint["complaint_id"],
            "complaint_text": complaint["text"],
            "category":       complaint["category"],
            "priority":       priority,
            "customer_tier":  tier,
            "decision":       decision,
            "verdict":        verdict,
            "correct":        is_correct or is_acceptable,
            "correct_actions": correct_actions,
            "confidence":     action.confidence,
            "reasoning":      action.reasoning,
            "cost_incurred":  cost,
            "satisfaction_delta": round(sat_delta * 0.1, 4),
            "reward_breakdown": {
                "base_correctness":  round(base, 4),
                "priority_weight":   priority_mult,
                "tier_weight":       tier_mult,
                "weighted_score":    round(weighted, 4),
                "calibration_bonus": round(calib, 4),
                "budget_penalty":    round(budget_p, 4),
                "total":             total,
            },
        }
        return feedback, total

    def _build_observation(self, reward: float, done: bool) -> SupportObservation:
        """Serialise environment state into a SupportObservation."""
        # Build serialisable complaint list (omit internal scoring maps)
        safe_complaints = []
        for c in self._active_complaints:
            safe_complaints.append({
                "complaint_id":          c["complaint_id"],
                "text":                  c["text"],
                "category":              c["category"],
                "priority":              c["priority"],
                "sentiment_score":       c.get("sentiment", 0.0),
                "customer_tier":         c.get("customer_tier", "regular"),
                "days_since_purchase":   c.get("days_since_purchase", 0),
                "previous_complaints":   c.get("previous_complaints", 0),
                "estimated_order_value": c.get("estimated_order_value", 0.0),
                "context_clues":         c.get("clues", []),
                "ambiguity_level":       c.get("ambiguity", 0.3),
            })

        return SupportObservation(
            complaints         = safe_complaints,
            episode_step       = self._state.step_count,
            max_steps          = self._config.get("max_steps", 5),
            cumulative_reward  = self._cumulative_reward,
            satisfaction_score = self._satisfaction,
            budget_remaining   = self._budget_remaining,
            escalation_count   = self._escalation_count,
            backlog_size       = len(self._backlog),
            last_step_feedback = self._last_feedback,
            task_name          = self._config.get("name", ""),
            task_level         = self._task_level,
            task_description   = self._config.get("description", ""),
            done               = done,
            reward             = reward,
            metadata           = {
                "episode_id":       self._state.episode_id,
                "task_level":       self._task_level,
                "step_count":       self._state.step_count,
                "decision_history": self._decision_history[-5:],
            },
        )
