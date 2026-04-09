"""
Data models for the CustomerSupportEnv Environment.

A real-world AI customer support decision-making environment where an agent must
triage complaints, assess urgency, assign actions, and manage customer satisfaction
over multi-step episodes using intelligent heuristics and dynamic reward signals.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ─────────────────────────────────────────────
#  ACTION
# ─────────────────────────────────────────────

class SupportAction(Action):
    """
    Action submitted by the agent for a single customer complaint.

    Fields
    ------
    complaint_id : str         – UUID of the complaint being handled
    decision     : str         – One of: 'refund', 'replace', 'escalate',
                                 'apologize', 'ignore', 'investigate'
    confidence   : float       – Agent's self-reported confidence [0.0, 1.0]
    reasoning    : str         – Free-text explanation of the decision
    urgency_flag : bool        – Whether the agent flags this as urgent
    """
    complaint_id: str = Field(..., description="UUID of the complaint")
    decision: str = Field(..., description="Action: refund|replace|escalate|apologize|ignore|investigate")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Agent confidence [0,1]")
    reasoning: str = Field(default="", description="Agent's reasoning for the decision")
    urgency_flag: bool = Field(default=False, description="Whether the agent considers this urgent")


# ─────────────────────────────────────────────
#  OBSERVATION
# ─────────────────────────────────────────────

class ComplaintRecord(Observation):
    """A single customer complaint with full context."""
    complaint_id: str = Field(..., description="Unique complaint identifier")
    text: str = Field(..., description="Raw complaint text from the customer")
    category: str = Field(..., description="Inferred category: billing|delivery|quality|technical|policy")
    priority: str = Field(..., description="Priority level: critical|high|medium|low")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment [-1=rage, 1=calm]")
    customer_tier: str = Field(..., description="Customer tier: vip|regular|new")
    days_since_purchase: int = Field(..., description="Days since original purchase")
    previous_complaints: int = Field(..., description="Number of prior complaints from this customer")
    estimated_order_value: float = Field(..., description="Estimated value (USD) of the related order")
    context_clues: List[str] = Field(default_factory=list, description="Key contextual hints extracted from text")
    ambiguity_level: float = Field(default=0.0, ge=0.0, le=1.0, description="How ambiguous this complaint is")
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SupportObservation(Observation):
    """
    Full observation returned after reset() or step().

    Contains the active complaint batch plus episode-level tracking data
    so the agent can reason about history and cumulative performance.
    """
    # Current batch of complaints presented to the agent
    complaints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of complaint dicts for the current step"
    )

    # Episode-level state visible to the agent
    episode_step: int = Field(default=0, description="Current step within episode")
    max_steps: int = Field(default=5, description="Maximum steps in this episode")
    cumulative_reward: float = Field(default=0.0, description="Running total reward")
    satisfaction_score: float = Field(default=0.99, ge=0.0, le=1.0, description="Current customer satisfaction index")
    budget_remaining: float = Field(default=1000.0, description="Remaining resolution budget (USD)")
    escalation_count: int = Field(default=0, description="Number of escalations used so far")
    backlog_size: int = Field(default=0, description="Unresolved complaints from previous steps")

    # Decision feedback from previous step (empty on reset)
    last_step_feedback: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-complaint feedback from the previous step"
    )
    task_name: str = Field(default="", description="Name of the current task (easy/medium/hard)")
    task_level: str = Field(default="", description="Difficulty level: easy|medium|hard")
    task_description: str = Field(default="", description="Human-readable task objective")

    # Standard OpenEnv fields
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
