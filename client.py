"""CustomerSupportEnv Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SupportAction, SupportObservation
except ImportError:
    from models import SupportAction, SupportObservation


class CustomerSupportEnv(
    EnvClient[SupportAction, SupportObservation, State]
):
    """
    Client for the CustomerSupportEnv environment.

    Example:
        >>> with CustomerSupportEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset()
        ...     for complaint in obs.observation.complaints:
        ...         action = SupportAction(
        ...             complaint_id=complaint["complaint_id"],
        ...             decision="refund",
        ...             confidence=0.9,
        ...             reasoning="Billing error requires immediate refund",
        ...             urgency_flag=True,
        ...         )
        ...         result = env.step(action)
        ...         print(result.observation.last_step_feedback)
    """

    def _step_payload(self, action: SupportAction) -> Dict:
        return {
            "complaint_id": action.complaint_id,
            "decision":     action.decision,
            "confidence":   action.confidence,
            "reasoning":    action.reasoning,
            "urgency_flag": action.urgency_flag,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupportObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportObservation(
            complaints         = obs_data.get("complaints", []),
            episode_step       = obs_data.get("episode_step", 0),
            max_steps          = obs_data.get("max_steps", 5),
            cumulative_reward  = obs_data.get("cumulative_reward", 0.0),
            satisfaction_score = obs_data.get("satisfaction_score", 1.0),
            budget_remaining   = obs_data.get("budget_remaining", 1000.0),
            escalation_count   = obs_data.get("escalation_count", 0),
            backlog_size       = obs_data.get("backlog_size", 0),
            last_step_feedback = obs_data.get("last_step_feedback", []),
            task_name          = obs_data.get("task_name", ""),
            task_description   = obs_data.get("task_description", ""),
            done               = payload.get("done", False),
            reward             = payload.get("reward", 0.0),
            metadata           = obs_data.get("metadata", {}),
        )
        return StepResult(
            observation = observation,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id  = payload.get("episode_id"),
            step_count  = payload.get("step_count", 0),
        )
