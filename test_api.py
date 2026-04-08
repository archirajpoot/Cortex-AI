import requests, json

def p(s): 
    print(str(s).encode('ascii', 'replace').decode())

p("=== CustomerSupportEnv Full API Validation ===")

# ── 1. Health ────────────────────────────────────────────────
r = requests.get("http://127.0.0.1:8000/health")
p(f"[1] Health: {r.status_code} {'OK' if r.status_code == 200 else 'FAIL'}")

# ── 2. Reset (Easy) ──────────────────────────────────────────
r = requests.post("http://127.0.0.1:8000/reset")
d = r.json()
obs = d["observation"]
p(f"[2] Reset: {r.status_code}")
p(f"    Task:        {obs['task_name']}")
p(f"    Level:       {obs.get('task_level', 'N/A')}")
p(f"    MaxSteps:    {obs['max_steps']}")
p(f"    Budget:      ${obs['budget_remaining']}")
p(f"    Complaints:  {len(obs['complaints'])}")
c1 = obs["complaints"][0]
p(f"    Complaint:   {c1['text'][:60]}...")
p(f"    Category:    {c1['category']} | Priority: {c1['priority']} | Tier: {c1['customer_tier']}")
p(f"    Clues:       {c1['context_clues']}")
p(f"    Sentiment:   {c1['sentiment_score']} | Ambiguity: {c1['ambiguity_level']:.2f}")

# ── 3. Correct Step ──────────────────────────────────────────
action = {
    "complaint_id": c1["complaint_id"],
    "decision":     "refund",
    "confidence":   0.90,
    "reasoning":    "Billing error - customer double charged, refund required",
    "urgency_flag": True,
}
r2 = requests.post("http://127.0.0.1:8000/step", json={"action": action})
d2 = r2.json()
obs2 = d2["observation"]
p(f"\n[3] Step (correct action 'refund'):")
p(f"    Step Reward:      {d2['reward']}")
p(f"    Cumulative:       {obs2['cumulative_reward']}")
p(f"    Budget Left:      ${obs2['budget_remaining']}")
p(f"    Satisfaction:     {obs2['satisfaction_score']}")
for fb in obs2.get("last_step_feedback", []):
    rb = fb.get("reward_breakdown", {})
    p(f"    Verdict:          {fb.get('verdict','?')}")
    p(f"    Base weighted:    {rb.get('weighted_score')}")
    p(f"    Calib bonus:      {rb.get('calibration_bonus')}")
    p(f"    Budget penalty:   {rb.get('budget_penalty')}")
    p(f"    Total reward:     {rb.get('total')}")
    p(f"    Cost incurred:    ${fb.get('cost_incurred')}")

# ── 4. Wrong Action ──────────────────────────────────────────
complaints2 = obs2.get("complaints", [])
if complaints2:
    c2 = complaints2[0]
    action2 = {
        "complaint_id": c2["complaint_id"],
        "decision":     "replace",
        "confidence":   0.95,
        "reasoning":    "Wrong: replace does not apply to billing issues",
        "urgency_flag": False,
    }
    r3 = requests.post("http://127.0.0.1:8000/step", json={"action": action2})
    d3 = r3.json()
    p(f"\n[4] Step (wrong action 'replace' on '{c2['category']}'):")
    p(f"    Reward: {d3['reward']} (should be negative/near 0)")
    for fb in d3["observation"].get("last_step_feedback", []):
        p(f"    Verdict: {fb.get('verdict','?')} | Correct actions: {fb.get('correct_actions')}")
else:
    p("\n[4] No more complaints this episode (episode done)")

# ── 5. Schema ────────────────────────────────────────────────
r4 = requests.get("http://127.0.0.1:8000/schema")
schema = r4.json()
action_fields = list(schema.get("action", {}).get("properties", {}).keys())
obs_fields    = list(schema.get("observation", {}).get("properties", {}).keys())
p(f"\n[5] Schema:")
p(f"    Action fields:      {action_fields}")
p(f"    Observation fields: {obs_fields}")

# ── 6. Full Episode (Medium) ─────────────────────────────────
p("\n[6] Running medium episode (2 complaints/step x 4 steps)...")
r5 = requests.post("http://127.0.0.1:8000/reset")  # now medium
obs5 = r5.json()["observation"]
p(f"    Task:  {obs5['task_name']} ({obs5.get('task_level','?')})")
total_r = 0.0
step = 0
done = obs5.get("done", False)
while not done and step < 8:
    comps = obs5.get("complaints", [])
    if not comps:
        break
    c = comps[0]
    # heuristic: billing->refund, delivery->investigate, quality->replace, technical->investigate, policy->escalate
    decision_map = {"billing": "refund", "delivery": "investigate", "quality": "replace", "technical": "investigate", "policy": "escalate"}
    dec = decision_map.get(c["category"], "investigate")
    act = {"complaint_id": c["complaint_id"], "decision": dec, "confidence": 0.75, "reasoning": f"heuristic: {c['category']}", "urgency_flag": False}
    rs = requests.post("http://127.0.0.1:8000/step", json={"action": act})
    drs = rs.json()
    total_r += drs.get("reward", 0)
    obs5 = drs["observation"]
    done = drs.get("done", False)
    step += 1
p(f"    Steps: {step} | Total reward: {total_r:.4f} | Final satisfaction: {obs5['satisfaction_score']}")

p("\n=== ALL VALIDATIONS COMPLETE ===")
