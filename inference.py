"""
Advanced AI Inference Script — CustomerSupportEnv
=================================================
Hackathon Submission for Meta OpenEnv 2026

Transforms the simple rule-based agent into a sophisticated LLM-driven Decision Engine.
Implements:
- Multi-step reasoning and state tracking
- Dynamic uncertainty calibration 
- Trade-off evaluation (satisfaction vs budget)
- Structured stdout logging strictly adhering to [START], [STEP], [END] formats.

"""

import os
import json
import time
from typing import Dict, Any

from openai import OpenAI
from client import CustomerSupportEnv
from models import SupportAction

# ────────────────────────────────────────────────────────
# MANDATORY HACKATHON VARIABLES
# ────────────────────────────────────────────────────────
import os

# INITIALISE OPENAI CLIENT
llm_client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"]
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")

# Server URL for the env API
SERVER_URL = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:8000")


def generate_intelligent_decision(complaint: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the LLM to process semantic nuances, calculate uncertainty, and reason
    through budget constraints vs customer retention tradeoffs.
    """
    
    prompt = f"""
    You are an advanced Customer Support AI for an enterprise.
    You must make a highly reasoned decision to maximize long-term reward.
    
    CURRENT STATE:
    Budget Remaining: ${context['budget_remaining']}
    Escalations Used: {context['escalation_count']} / 4
    Satisfaction Score: {context['satisfaction_score'] * 100}%
    
    COMPLAINT DETAILS:
    Text: "{complaint.get('text')}"
    Priority: {complaint.get('priority')}
    Tier: {complaint.get('customer_tier')}
    Sentiment Score: {complaint.get('sentiment_score')} (-1 scale = angry)
    Estimated Value: ${complaint.get('estimated_order_value')}
    
    AVAILABLE ACTIONS:
    refund, replace, escalate, apologize, ignore, investigate.
    
    RULES:
    1. Weigh the cost of 'refund' or 'replace' against the Budget Remaining.
    2. VIP customers require high empathy and stronger resolutions.
    3. If ambiguity is high or sentiment is severe but details are lacking, select 'investigate'.
    4. Guard against overuse of 'escalate' to avoid laziness penalties.
    5. Output absolute Confidence [0.0 to 1.0]. Lower confidence if the decision is risky.
    
    Respond STRICTLY in JSON format:
    {{
        "decision": "<action>",
        "confidence": 0.85,
        "reasoning": "<1-2 sentences explaining tradeoffs and logic>",
        "urgency_flag": true/false
    }}
    """
    
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2, # Low temperature for reliable outputs but probabilistic logic in prompt
    )
    content = response.choices[0].message.content
    return json.loads(content)


def run_inference_episode():
    # Initialize the client context manager
    with CustomerSupportEnv(base_url=SERVER_URL).sync() as env:
        obs_result = env.reset()
        obs = obs_result.observation
        
        # REQUIRED STDOUT FORMAT
        print(f"[START] Episode Initialized | Task: {obs.task_name} | Steps: {obs.max_steps}")
        
        done = obs.done    
        while not done:
            if not obs.complaints:
                break
                
            complaint = obs.complaints[0]
            
            # State passing for context reasoning
            context = {
                "budget_remaining": obs.budget_remaining,
                "escalation_count": obs.escalation_count,
                "satisfaction_score": obs.satisfaction_score,
                "history": obs.metadata.get("decision_history", [])
            }
            
            # Generate advanced prediction
            ai_output = generate_intelligent_decision(complaint, context)
            
            # Form action
            action = SupportAction(
                complaint_id=complaint["complaint_id"],
                decision=ai_output.get("decision", "investigate"),
                confidence=ai_output.get("confidence", 0.7),
                reasoning=ai_output.get("reasoning", "Rule-based reasoning"),
                urgency_flag=ai_output.get("urgency_flag", False)
            )
            
            # Step the environment
            step_result = env.step(action)
            obs = step_result.observation
            done = step_result.done
            
            # REQUIRED STDOUT FORMAT
            # Structured print for validator parsing
            print(f"[STEP] Decision: {action.decision.upper()} | Conf: {action.confidence} | Reward: {step_result.reward:+.4f} | Rationale: {action.reasoning}")
        
        # REQUIRED STDOUT FORMAT
        total_reward = obs.cumulative_reward
        print(f"[END] Episode complete. Cumulative Reward: {total_reward:+.4f}")


if __name__ == "__main__":
    # In a hackathon scenario, the evaluator may call this script directly.
    # We will simulate running an episode to print the structured traces.
    run_inference_episode()
