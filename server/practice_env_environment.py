from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PracticeAction, PracticeObservation
except ImportError:
    from models import PracticeAction, PracticeObservation


class PracticeEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> PracticeObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # choose task level
        self.task_level = "medium"   # change: easy / medium / hard

        if self.task_level == "easy":
            self.complaints = [
                {"text": "order late", "type": "refund", "priority": "high"}
            ]

        elif self.task_level == "medium":
            self.complaints = [
                {"text": "order late", "type": "refund", "priority": "high"},
                {"text": "product broken", "type": "replace", "priority": "high"},
                {"text": "payment issue", "type": "refund", "priority": "medium"}
            ]

        else:  # hard
            self.complaints = [
                {"text": "refund after 30 days", "type": "ignore", "priority": "medium"},
                {"text": "product slightly damaged", "type": "replace", "priority": "medium"},
                {"text": "order late but already refunded", "type": "ignore", "priority": "high"}
            ]

        return PracticeObservation(
            echoed_message=str(self.complaints),
            message_length=len(self.complaints),
            done=False,
            reward=0.0,
        )
    def step(self, action: PracticeAction) -> PracticeObservation:
     self._state.step_count += 1
     if not hasattr(self,"complaints"):
         self.reset()
     try:
        
         if hasattr(action, "message"):
            message = action.message
         elif hasattr(action, "action"):
            message = action.action.get("message", "")
         else:
            message = ""

         actions = message.split(",")

         total_reward = 0
         results = []

         for i, complaint in enumerate(self.complaints):
            if i < len(actions):
                user_action = actions[i].strip()

                if complaint["type"] == user_action:
                    if complaint["priority"] == "high":
                        reward = 2
                    elif complaint["priority"] == "medium":
                        reward = 1
                    else:
                        reward = 0.5
                    result = "correct"
                else:
                    reward = -1
                    result = "wrong"

                total_reward += reward
                results.append(
                    f"{complaint['text']} → {result}"
                )

         return PracticeObservation(
            echoed_message=" | ".join(results),
            message_length=len(actions),
            done=True,
            reward=float(total_reward),
        )

     except Exception as e:
        return PracticeObservation(
            echoed_message=f"Error: {str(e)}",
            message_length=0,
            done=True,
            reward=-1.0,
        )
    @property
    def state(self) -> State:
        return self._state
    