"""
AI Operations Center — CustomerSupportEnv
Premium Hackathon Dashboard | Meta OpenEnv 2026
Run: python app.py
"""

import json, random, time, os
from typing import Any, Dict, List, Optional
import gradio as gr
import requests

API = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:7860")

# ── SESSION STATE ──────────────────────────────────────────────────────────────
def fresh_session():
    return {
        "complaints": [], "active_idx": 0, "episode_step": 0, "max_steps": 5,
        "cumulative_reward": 0.0, "satisfaction": 1.0, "budget": 1000.0,
        "escalation_count": 0, "backlog_size": 0, "task_name": "", "task_level": "",
        "history": [], "reward_history": [], "episodes_log": [], "done": False,
    }

def demo_session():
    hist = [
        {"step":1,"complaint_text":"Double-charged on credit card.","decision":"refund","verdict":"✅ Optimal","correct":True,"total_reward":0.762,"category":"billing","customer_name":"Shreya"},
        {"step":2,"complaint_text":"Order arrived 3 weeks late.","decision":"apologize","verdict":"❌ Wrong","correct":False,"total_reward":-0.215,"category":"delivery","customer_name":"Arjun"},
        {"step":3,"complaint_text":"Product stopped working after one day.","decision":"replace","verdict":"✅ Optimal","correct":True,"total_reward":0.621,"category":"quality","customer_name":"Priya"},
        {"step":4,"complaint_text":"App crashes every time at checkout.","decision":"investigate","verdict":"✅ Optimal","correct":True,"total_reward":0.480,"category":"technical","customer_name":"Rahul"},
        {"step":5,"complaint_text":"Return window expired yesterday.","decision":"escalate","verdict":"⚡ Acceptable","correct":True,"total_reward":0.310,"category":"policy","customer_name":"Ananya"},
        {"step":6,"complaint_text":"VIP: small scratch, wants acknowledgment.","decision":"apologize","verdict":"✅ Optimal","correct":True,"total_reward":0.540,"category":"quality","customer_name":"Vikram"},
    ]
    return {
        "complaints": [], "active_idx": 0, "episode_step": 6, "max_steps": 5,
        "cumulative_reward": 2.498, "satisfaction": 0.87, "budget": 760.0,
        "escalation_count": 1, "backlog_size": 0, "task_name": "Demo Session",
        "task_level": "medium", "history": hist,
        "reward_history": [0.762, -0.215, 0.621, 0.480, 0.310, 0.540],
        "done": True,
    }

# ── API ────────────────────────────────────────────────────────────────────────
def api_reset():
    try:
        r = requests.post(f"{API}/reset", timeout=8); r.raise_for_status(); return r.json()
    except Exception as e: return {"error": str(e)}

def api_step(action_payload):
    try:
        r = requests.post(f"{API}/step", json={"action": action_payload}, timeout=8)
        r.raise_for_status(); return r.json()
    except Exception as e: return {"error": str(e)}

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
PC = {"critical":"#FF3B30","high":"#FF9500","medium":"#FFCC00","low":"#34C759"}
CI = {"billing":"💳","delivery":"📦","quality":"🔧","technical":"🖥️","policy":"📜"}
TI = {"vip":"👑","regular":"👤","new":"🌱"}
DI = {"refund":"💰","replace":"🔄","escalate":"⬆️","apologize":"🙏","ignore":"🚫","investigate":"🔍"}
VC = {"✅ Optimal":"#34C759","⚡ Acceptable":"#FFCC00","❌ Wrong":"#FF3B30"}
DC = {"refund":"#5E5CE6","replace":"#34C759","escalate":"#FF9500","apologize":"#BF5AF2","ignore":"#FF3B30","investigate":"#64D2FF"}

# ── GRADER ────────────────────────────────────────────────────────────────────
def grade_episode(history: list) -> float:
    if not history: return 0.0
    correct = sum(1 for h in history if h.get("correct", False))
    ratio = correct / len(history)
    avg_r = sum(h.get("total_reward", 0.0) for h in history) / len(history)
    norm_r = (avg_r + 1.0) / 2.0
    return round(max(0.0, min(1.0, ratio * 0.6 + norm_r * 0.4)), 4)

TASK_THRESHOLDS = {"easy": 0.70, "medium": 0.55, "hard": 0.40}

# ── RENDER: TOP LIVE METRICS BAR ──────────────────────────────────────────────
def render_live_metrics(session: dict) -> str:
    sat = session.get("satisfaction", 1.0)
    budget = session.get("budget", 1000.0)
    cr = session.get("cumulative_reward", 0.0)
    hist = session.get("history", [])
    esc = session.get("escalation_count", 0)
    step = session.get("episode_step", 0)
    max_steps = session.get("max_steps", 5)
    acc = int(len([h for h in hist if h.get("correct")]) / max(len(hist), 1) * 100) if hist else 0

    sat_col = "#34C759" if sat > 0.7 else "#FF9500" if sat > 0.4 else "#FF3B30"
    cr_col = "#34C759" if cr >= 0 else "#FF3B30"
    acc_col = "#34C759" if acc > 65 else "#FF9500" if acc > 40 else "#FF3B30"

    level = session.get("task_level", "—").upper()
    task = session.get("task_name", "No Active Session")

    return f"""
<div class="live-bar">
  <div class="lb-brand">
    <span class="lb-logo">🤖</span>
    <div>
      <div class="lb-title">AI Operations Center</div>
      <div class="lb-sub">{task} · <span style="color:var(--accent);">{level}</span></div>
    </div>
  </div>
  <div class="lb-divider"></div>
  <div class="lb-metric">
    <div class="lb-val" style="color:{sat_col};">{int(sat*100)}%</div>
    <div class="lb-lab">Satisfaction</div>
  </div>
  <div class="lb-metric">
    <div class="lb-val">💵 ${budget:.0f}</div>
    <div class="lb-lab">Budget Left</div>
  </div>
  <div class="lb-metric">
    <div class="lb-val" style="color:{cr_col};">{cr:+.2f}</div>
    <div class="lb-lab">Cumul. Reward</div>
  </div>
  <div class="lb-metric">
    <div class="lb-val" style="color:{acc_col};">{acc}%</div>
    <div class="lb-lab">Accuracy</div>
  </div>
  <div class="lb-metric">
    <div class="lb-val">⬆️ {esc}/4</div>
    <div class="lb-lab">Escalations</div>
  </div>
  <div class="lb-metric">
    <div class="lb-val">S{step}/{max_steps}</div>
    <div class="lb-lab">Progress</div>
  </div>
</div>"""

# ── RENDER: COMPLAINT CARD ─────────────────────────────────────────────────────
def render_complaint_card(c: dict, active: bool = False) -> str:
    if not c: return ""
    name = c.get("customer_name", "Valued Customer")
    li = c.get("last_interaction", "None")
    pri = c.get("priority", "medium"); cat = c.get("category", "delivery")
    tier = c.get("customer_tier", "regular")
    col = PC.get(pri, "#888"); sent = c.get("sentiment_score", 0.0)
    amb = c.get("ambiguity_level", 0.3); ab = int(amb * 10)
    sl = "😡 Enraged" if sent < -0.7 else "😠 Frustrated" if sent < -0.3 else "😐 Neutral" if sent < 0.1 else "🙂 Calm"
    clues = "".join(f'<span class="clue-tag">{cl}</span>' for cl in c.get("context_clues", [])[:4])
    active_class = " card-active" if active else ""
    prev = c.get("previous_complaints", 0)
    repeat_badge = f'<span class="repeat-badge">🔁 Repeat Customer</span>' if prev > 0 else ""

    return f"""
<div class="complaint-card{active_class}">
  <div class="card-header">
    <div class="card-identity">
      <div class="customer-avatar" style="background:linear-gradient(135deg,{col}40,{col}20);">{name[0]}</div>
      <div>
        <div class="customer-name-label">{name}</div>
        <div class="customer-meta">{TI.get(tier,'👤')} {tier.upper()} · {CI.get(cat,'📋')} {cat}</div>
      </div>
    </div>
    <div class="card-badges">
      <span class="priority-pill" style="background:{col}22;color:{col};border:1px solid {col}44;">{pri.upper()}</span>
      {repeat_badge}
    </div>
  </div>
  <div class="complaint-body">"{c.get('text','')}"</div>
  {'<div class="last-interaction-row"><span class="li-icon">🕒</span><span class="li-label">Last:</span><span class="li-val">' + li + '</span></div>' if prev > 0 else ''}
  <div class="card-stats">
    <span>📅 {c.get('days_since_purchase',0)}d ago</span>
    <span>💵 ${c.get('estimated_order_value',0):.0f}</span>
    <span>Sentiment: {sl}</span>
    <span>Ambiguity: {'🟨'*ab}{'⬜'*(10-ab)}</span>
  </div>
  {'<div class="clues-row">' + clues + '</div>' if clues else ''}
</div>"""

# ── RENDER: AI REASONING PANEL ─────────────────────────────────────────────────
def render_reasoning_panel(fb: dict, reasoning: str, confidence: float, s1: str = "", s2: str = "", s3: str = "") -> str:
    if not fb:
        return '<div class="reasoning-placeholder"><div class="rp-icon">🧠</div><div>Submit a decision to see Triple-Check reasoning breakdown</div></div>'
    verdict = fb.get("verdict", ""); col = VC.get(verdict, "#888")
    rb = fb.get("reward_breakdown", {}); cp = int(confidence * 100)
    ca = ", ".join(fb.get("correct_actions", []))
    name = fb.get("complaint_text", "")[:50]

    rows = [
        ("Base Correctness", f"{rb.get('weighted_score',0):+.3f}", col),
        ("Priority × Tier", f"{rb.get('priority_weight',1):.1f} × {rb.get('tier_weight',1):.1f}", "#888"),
        ("Confidence Calib.", f"{rb.get('calibration_bonus',0):+.3f}", "#34C759" if rb.get("calibration_bonus", 0) >= 0 else "#FF3B30"),
        ("Budget Penalty", f"{rb.get('budget_penalty',0):+.3f}", "#34C759" if rb.get("budget_penalty", 0) >= 0 else "#FF3B30"),
    ]
    bd_html = "".join(f'<div class="bd-row"><span>{r[0]}</span><span style="color:{r[2]};font-weight:600;">{r[1]}</span></div>' for r in rows)
    total = rb.get("total", 0)

    # Use input steps, fallback to generic if empty
    s1_text = s1 or "Sentiment & policy analysis check complete."
    s2_text = s2 or "Financial impact against remaining budget assessed."
    s3_text = s3 or "Final personalized resolution decided."

    return f"""
<div class="reasoning-panel">
  <div class="rp-header">
    <span class="verdict-chip" style="background:{col}22;color:{col};border:1px solid {col}44;">{verdict}</span>
    <div class="conf-wrap">
      <span style="font-size:11px;color:var(--text-muted);">Confidence</span>
      <div class="conf-track"><div class="conf-fill" style="width:{cp}%;background:{col};"></div></div>
      <span style="font-size:12px;font-weight:700;color:{col};">{cp}%</span>
    </div>
  </div>
  <div class="reasoning-badge">✨ Optimal: <strong>{ca}</strong></div>

  <!-- TRIPLE CHECK UI -->
  <div style="font-size:10px;color:var(--text-muted);margin:16px 0 8px;letter-spacing:1px;font-weight:700;">TRIPLE-CHECK REASONING</div>
  
  <div class="tc-step" style="border-left: 2px solid var(--accent2); padding-left: 12px; margin-bottom: 12px;">
    <div style="color:var(--accent2); font-size:11px; font-weight:700; margin-bottom:4px;">1. SENTIMENT & POLICY</div>
    <div style="color:var(--text); font-size:13px; line-height:1.4;">{s1_text}</div>
  </div>

  <div class="tc-step" style="border-left: 2px solid var(--orange); padding-left: 12px; margin-bottom: 12px;">
    <div style="color:var(--orange); font-size:11px; font-weight:700; margin-bottom:4px;">2. FINANCIAL IMPACT</div>
    <div style="color:var(--text); font-size:13px; line-height:1.4;">{s2_text}</div>
  </div>

  <div class="tc-step" style="border-left: 2px solid var(--green); padding-left: 12px; margin-bottom: 16px;">
    <div style="color:var(--green); font-size:11px; font-weight:700; margin-bottom:4px;">3. PERSONALIZED RESOLUTION</div>
    <div style="color:var(--text); font-size:13px; line-height:1.4; font-style:italic;">"{s3_text}"</div>
  </div>

  <div class="breakdown-card">
    <div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;letter-spacing:1px;">REWARD BREAKDOWN</div>
    {bd_html}
    <div class="bd-total"><span>Total Reward</span><span style="color:{col};font-size:16px;font-weight:800;">{total:+.3f}</span></div>
  </div>
  <div class="rp-footer">
    <span>💰 Cost: <strong>${fb.get('cost_incurred',0)}</strong></span>
    <span>😊 Sat Δ: <strong style="color:{'#34C759' if fb.get('satisfaction_delta',0)>=0 else '#FF3B30'};">{fb.get('satisfaction_delta',0):+.3f}</strong></span>
  </div>
</div>"""

# ── RENDER: HISTORY TIMELINE ───────────────────────────────────────────────────
def render_history_timeline(history: list) -> str:
    if not history:
        return '<div class="timeline-empty">No decisions yet in this episode.</div>'
    rows = ""
    for h in reversed(history[-8:]):
        verdict = h.get("verdict", ""); col = VC.get(verdict, "#888")
        name = h.get("customer_name", "")
        name_tag = f'<span style="color:#9B9AF8;font-weight:600;">{name}</span> · ' if name else ""
        rows += f"""
<div class="tl-item">
  <div class="tl-dot" style="background:{col};box-shadow:0 0 6px {col}66;"></div>
  <div class="tl-content">
    <div class="tl-top">
      <span class="tl-step">S{h.get('step','?')}</span>
      <span class="tl-action">{DI.get(h.get('decision',''),'?')} {h.get('decision','')}</span>
      <span class="tl-verdict" style="color:{col};">{verdict}</span>
      <span class="tl-reward" style="color:{col};">{h.get('total_reward',0):+.3f}</span>
    </div>
    <div class="tl-text">{name_tag}{h.get('complaint_text','')[:55]}…</div>
  </div>
</div>"""
    return f'<div class="timeline">{rows}</div>'

# ── RENDER: REWARD CHART ───────────────────────────────────────────────────────
def render_reward_chart(reward_history: list) -> str:
    if not reward_history:
        return '<div class="chart-empty"><div style="font-size:28px;">📈</div><div>Make decisions to see the reward trend</div></div>'
    W, H, PAD = 440, 120, 24
    rh = [0.0] + list(reward_history) if len(reward_history) == 1 else list(reward_history)
    n = len(rh); min_r = min(min(rh), -0.1); max_r = max(max(rh), 0.1); rng = max_r - min_r
    def sx(i): return PAD + (i / (n - 1)) * (W - 2 * PAD)
    def sy(r): return H - PAD - ((r - min_r) / rng) * (H - 2 * PAD)
    pts = [(sx(i), sy(r)) for i, r in enumerate(rh)]; zero_y = sy(0)
    pd = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}" + "".join(f" L {x:.1f} {y:.1f}" for x, y in pts[1:])
    ad = pd + f" L {pts[-1][0]:.1f} {zero_y:.1f} L {pts[0][0]:.1f} {zero_y:.1f} Z"
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{"#34C759" if rh[i]>=0 else "#FF3B30"}" stroke="#0A0A0F" stroke-width="2"/>' for i, (x, y) in enumerate(pts))
    yl = "".join(f'<text x="{PAD-5}" y="{sy(v)+4:.1f}" text-anchor="end" font-size="9" fill="#444">{v:+.1f}</text><line x1="{PAD}" y1="{sy(v):.1f}" x2="{W-PAD}" y2="{sy(v):.1f}" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>' for v in [max_r, 0, min_r])
    xl = "".join(f'<text x="{sx(i):.1f}" y="{H-4}" text-anchor="middle" font-size="9" fill="#444">S{i}</text>' for i in range(n))
    zl = f'<line x1="{PAD}" y1="{zero_y:.1f}" x2="{W-PAD}" y2="{zero_y:.1f}" stroke="rgba(255,255,255,0.15)" stroke-width="1" stroke-dasharray="4,3"/>'
    avg_r = sum(rh) / len(rh); last_r = rh[-1]
    return f"""
<div class="chart-wrap">
  <div class="chart-header">
    <span class="chart-title-text">📈 Step Reward Trend</span>
    <div style="display:flex;gap:12px;">
      <span style="font-size:11px;font-weight:600;color:{'#34C759' if avg_r>=0 else '#FF3B30'};">Avg {avg_r:+.3f}</span>
      <span style="font-size:11px;font-weight:600;color:{'#34C759' if last_r>=0 else '#FF3B30'};">Last {last_r:+.3f}</span>
    </div>
  </div>
  <svg viewBox="0 0 {W} {H}" style="width:100%;height:{H}px;display:block;">
    <defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5E5CE6" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#5E5CE6" stop-opacity="0.0"/>
    </linearGradient></defs>
    {yl}{zl}<path d="{ad}" fill="url(#rg)"/>
    <path d="{pd}" fill="none" stroke="#5E5CE6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    {circles}{xl}
  </svg>
</div>"""

# ── RENDER: CATEGORY CHART ─────────────────────────────────────────────────────
def render_category_chart(history: list) -> str:
    if not history:
        return '<div class="chart-empty"><div style="font-size:28px;">📊</div><div>Submit decisions to see category breakdown</div></div>'
    cat_data = {}
    for h in history:
        cat = h.get("category", "unknown")
        if cat not in cat_data: cat_data[cat] = {"correct": 0, "total": 0, "reward": 0.0}
        cat_data[cat]["total"] += 1; cat_data[cat]["reward"] += h.get("total_reward", 0.0)
        if h.get("correct"): cat_data[cat]["correct"] += 1
    rows = ""
    for cat, d in sorted(cat_data.items()):
        pct = int(d["correct"] / max(d["total"], 1) * 100)
        avg_r = d["reward"] / max(d["total"], 1)
        col = "#34C759" if pct > 65 else "#FF9500" if pct > 40 else "#FF3B30"
        rc = "#34C759" if avg_r >= 0 else "#FF3B30"
        rows += f"""<div class="cat-row">
  <span class="cat-label">{CI.get(cat,'📋')} {cat}</span>
  <div class="cat-bar-track"><div class="cat-bar-fill" style="width:{pct}%;background:{col};"></div></div>
  <span style="min-width:34px;font-weight:700;font-size:11px;color:{col};">{pct}%</span>
  <span style="min-width:44px;font-weight:600;font-size:11px;color:{rc};text-align:right;">{avg_r:+.2f}</span>
</div>"""
    return f"""
<div class="chart-wrap">
  <div class="chart-header">
    <span class="chart-title-text">📊 Category Accuracy</span>
    <div style="display:flex;gap:8px;font-size:10px;">
      <span style="color:#34C759;">■ &gt;65%</span>
      <span style="color:#FF9500;">■ &gt;40%</span>
      <span style="color:#FF3B30;">■ Low</span>
    </div>
  </div>
  {rows}
</div>"""

# ── RENDER: GRADER SCORECARD ──────────────────────────────────────────────────
def render_episode_scorecard(session: dict) -> str:
    history = session.get("history", [])
    if not history:
        return '<div class="chart-empty"><div style="font-size:28px;">🎯</div><div>Complete an episode to see your grade</div></div>'
    score = grade_episode(history)
    level = session.get("task_level", "medium")
    threshold = TASK_THRESHOLDS.get(level, 0.55)
    passed = score >= threshold
    color = "#34C759" if passed else "#FF3B30"
    verdict = "✅ PASS" if passed else "❌ FAIL"
    pct = int(score * 100)
    correct = sum(1 for h in history if h.get("correct", False))
    fill_deg = int(score * 360)

    return f"""
<div class="scorecard">
  <div class="sc-header">🎯 Episode Grade Report</div>
  <div class="sc-body">
    <div class="score-ring" style="background:conic-gradient({color} {fill_deg}deg,rgba(255,255,255,0.06) 0deg);">
      <div class="score-inner">
        <div class="score-big" style="color:{color};">{score:.3f}</div>
        <div class="score-of">/ 1.000</div>
      </div>
    </div>
    <div class="score-info">
      <div class="score-verdict" style="color:{color};">{verdict}</div>
      <div class="score-row"><span>Task Level</span><strong style="color:#9B9AF8;text-transform:uppercase;">{level}</strong></div>
      <div class="score-row"><span>Threshold</span><strong>{threshold}</strong></div>
      <div class="score-row"><span>Correct</span><strong style="color:{color};">{correct}/{len(history)}</strong></div>
      <div class="score-row"><span>Reward</span><strong style="color:{'#34C759' if session.get('cumulative_reward',0)>=0 else '#FF3B30'};">{session.get('cumulative_reward',0):+.3f}</strong></div>
    </div>
  </div>
  <div class="score-bar-wrap">
    <div class="score-bar-track"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{color},{color}88);border-radius:4px;transition:width 0.8s ease;"></div></div>
    <span style="font-size:12px;font-weight:700;color:{color};min-width:38px;">{pct}%</span>
  </div>
  <div style="font-size:10px;color:var(--text-muted);margin-top:8px;text-align:center;">Formula: 60% decision accuracy + 40% normalised reward</div>
</div>"""

# ── RENDER: EPISODE LOG ───────────────────────────────────────────────────────
def render_episode_log(episodes_log: list) -> str:
    if not episodes_log:
        return '<div class="chart-empty"><div style="font-size:28px;">📝</div><div>Complete episodes to build your log</div></div>'
    rows = "".join(
        f'<div class="ep-row">'
        f'<span class="ep-num">#{i+1}</span>'
        f'<span class="ep-level" style="color:{"#34C759" if e["level"]=="easy" else "#FF9500" if e["level"]=="medium" else "#FF3B30"}">{e["level"].upper()}</span>'
        f'<div class="ep-bar"><div style="width:{int(e["score"]*100)}%;height:100%;background:{"#34C759" if e["passed"] else "#FF3B30"};border-radius:3px;"></div></div>'
        f'<span class="ep-score" style="color:{"#34C759" if e["passed"] else "#FF3B30"}">{e["score"]:.3f}</span>'
        f'<span class="ep-res">{"PASS" if e["passed"] else "FAIL"}</span>'
        f'</div>'
        for i, e in enumerate(episodes_log[-10:])
    )
    avg = sum(e["score"] for e in episodes_log) / len(episodes_log)
    passes = sum(1 for e in episodes_log if e["passed"])
    return f"""
<div class="chart-wrap">
  <div class="chart-header">
    <span class="chart-title-text">📝 Episode Log ({len(episodes_log)} runs)</span>
    <span style="font-size:11px;color:var(--text-muted);">Avg <strong style="color:#9B9AF8;">{avg:.3f}</strong> · Pass <strong style="color:#34C759;">{passes}/{len(episodes_log)}</strong></span>
  </div>
  {rows}
</div>"""

# ── RENDER: BASELINE HINT ─────────────────────────────────────────────────────
CAT_PRIORITY_MAP = {"billing":{"critical":"refund","high":"refund","medium":"investigate","low":"investigate"},"delivery":{"critical":"refund","high":"investigate","medium":"investigate","low":"apologize"},"quality":{"critical":"replace","high":"replace","medium":"apologize","low":"apologize"},"technical":{"critical":"escalate","high":"investigate","medium":"investigate","low":"apologize"},"policy":{"critical":"escalate","high":"escalate","medium":"apologize","low":"apologize"}}
SENT_OVERRIDE = {"chargeback":"escalate","no response":"escalate","businesses":"escalate","warranty":"investigate","acknowledgment":"apologize"}

def get_baseline_hint(complaint: dict) -> str:
    if not complaint: return ""
    text = complaint.get("text", "").lower(); cat = complaint.get("category", "delivery")
    pri = complaint.get("priority", "medium"); clues = [c.lower() for c in complaint.get("context_clues", [])]
    tier = complaint.get("customer_tier", "regular"); decision = None
    for kw, act in SENT_OVERRIDE.items():
        if kw in text or any(kw in c for c in clues): decision = act; break
    if not decision: decision = CAT_PRIORITY_MAP.get(cat, {}).get(pri, "apologize")
    col = DC.get(decision, "#888"); icon = DI.get(decision, "?")
    return f'<div class="hint-bar"><span class="hint-robot">🤖 Baseline suggests:</span><span class="hint-action" style="background:{col}22;color:{col};border:1px solid {col}44;">{icon} {decision.upper()}</span><span class="hint-why">({cat}, {pri}{", VIP" if tier=="vip" else ""})</span></div>'

# ── PREMIUM CSS ────────────────────────────────────────────────────────────────
PREMIUM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── TOKENS ── */
:root {
  --bg: #050510;
  --bg2: #0B0B1E;
  --bg3: #12122A;
  --bg4: #181832;
  --accent: #7C73FF;
  --accent2: #CF6AF2;
  --green: #34C759;
  --red: #FF3B30;
  --yellow: #FFCC00;
  --orange: #FF9500;
  --blue: #0A84FF;
  --text: #F0F0FF;
  --text2: #A0A0C0;
  --text3: #62628A;
  --border: rgba(255,255,255,0.08);
  --border2: rgba(255,255,255,0.15);
  --r: 16px;
  --r2: 12px;
  --r3: 8px;
  --shadow: 0 12px 40px rgba(0,0,0,0.6);
  --shadow2: 0 4px 16px rgba(0,0,0,0.4);
}

/* ── ANIMATIONS ── */
@keyframes glowPulse {
  0% { box-shadow: 0 0 5px rgba(124,115,255,0.2); }
  50% { box-shadow: 0 0 20px rgba(124,115,255,0.4); }
  100% { box-shadow: 0 0 5px rgba(124,115,255,0.2); }
}

@keyframes slideInUp {
  from { opacity: 0; transform: translateY(30px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes headerFlow {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes popIn {
  0% { opacity: 0; transform: scale(0.95); }
  60% { transform: scale(1.02); }
  100% { opacity: 1; transform: scale(1); }
}

@keyframes slideInRight {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes float {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-6px); box-shadow: 0 10px 25px rgba(124,115,255,0.3); }
  100% { transform: translateY(0px); }
}

/* ── GLOBAL ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container { 
    background: var(--bg) !important; 
    background-image: radial-gradient(circle at 50% -20%, #1A1A3E 0%, var(--bg) 60%) !important;
    color: var(--text) !important; 
    font-family: 'Inter', sans-serif !important; 
}
.gradio-container { padding: 0 !important; }
.contain { max-width: 1300px !important; margin: 0 auto !important; }

/* ── LIVE METRICS BAR ── */
.live-bar {
  display: flex; align-items: center; gap: 24px;
  background: rgba(8,8,22,0.85); backdrop-filter: blur(25px);
  border-bottom: 1px solid var(--border2); padding: 14px 28px;
  position: sticky; top: 0; z-index: 999;
  margin: -20px -20px 24px -20px;
  animation: fadeIn 1s ease-out;
}
.lb-brand { display: flex; align-items: center; gap: 12px; margin-right: auto; }
.lb-logo { font-size: 26px; filter: drop-shadow(0 0 8px var(--accent)); }
.lb-title { font-size: 16px; font-weight: 800; color: var(--text); letter-spacing: -0.5px; }
.lb-sub { font-size: 11px; color: var(--text2); margin-top: 1px; font-weight: 500; }
.lb-divider { width: 1px; height: 36px; background: var(--border); }
.lb-metric { display: flex; flex-direction: column; align-items: center; min-width: 75px; transition: transform 0.2s ease; }
.lb-metric:hover { transform: translateY(-3px); }
.lb-val { font-size: 17px; font-weight: 800; color: #fff; line-height: 1.2; }
.lb-lab { font-size: 9px; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; font-weight: 700; }

/* ── MAIN HEADER ── */
.page-header {
  background: linear-gradient(-45deg, #0A0A1E, #14143A, #1C0A36, #0A0A1E);
  background-size: 400% 400%;
  animation: headerFlow 15s ease infinite;
  border: 1px solid rgba(124,115,255,0.3);
  border-radius: 20px; padding: 40px 48px; margin-bottom: 32px;
  position: relative; overflow: hidden;
  box-shadow: 0 20px 50px rgba(0,0,0,0.4);
}
.page-header::after {
  content: ''; position: absolute; bottom: 0; right: 0; width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(124,115,255,0.15) 0%, transparent 70%);
  filter: blur(40px);
}
.ph-title { font-size: 34px; font-weight: 900; background: linear-gradient(135deg, #fff 20%, #BBAAFF 60%, #7C73FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 10px; letter-spacing: -1px; }
.ph-sub { color: var(--text2); font-size: 15px; margin: 0 0 24px; max-width: 600px; line-height: 1.5; }
.ph-chips { display: flex; gap: 10px; flex-wrap: wrap; }
.ph-chip { 
    background: rgba(255,255,255,0.04); 
    border: 1px solid var(--border); 
    border-radius: 20px; padding: 6px 16px; 
    font-size: 12px; color: var(--text2); font-weight: 600;
    transition: all 0.3s ease;
}
.ph-chip:hover { border-color: var(--accent); color: #fff; transform: scale(1.05); background: rgba(124,115,255,0.1); }
.ph-chip.ac { background: rgba(124,115,255,0.15); border-color: rgba(124,115,255,0.5); color: #B0A8FF; box-shadow: 0 0 15px rgba(124,115,255,0.1); }

/* ── FEATURE SPOTLIGHT ── */
.feature-spotlight {
    background: rgba(124,115,255,0.08); 
    border: 1px solid rgba(124,115,255,0.2); 
    border-radius: 12px; padding: 12px 20px; 
    margin-bottom: 24px; display: flex; align-items: center; gap: 14px;
    animation: fadeIn 1s ease-out;
}
.fs-icon { font-size: 18px; color: var(--accent); }
.fs-text { font-size: 13px; font-weight: 600; color: var(--text2); }
.fs-highlight { color: #fff; font-weight: 700; margin-right: 4px; }

/* ── TABS CUSTOMIZATION ── */
.tab-nav { 
    background: transparent !important; 
    border-bottom: 1px solid var(--border) !important;
    margin-bottom: 24px !important;
}
.tab-nav button { 
    font-size: 13px !important; 
    font-weight: 700 !important; 
    color: var(--text2) !important;
    padding: 12px 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    border: none !important;
    background: transparent !important;
}
.tab-nav button:hover { 
    color: #fff !important; 
    transform: translateY(-2px);
    text-shadow: 0 0 8px rgba(255,255,255,0.4);
}
.tab-nav button.selected { 
    color: var(--accent) !important; 
    background: rgba(124,115,255,0.08) !important;
}
.tab-nav button.selected::after {
    content: ''; position: absolute; bottom: 0; left: 15%; right: 15%; height: 3px;
    background: var(--accent); border-radius: 3px 3px 0 0;
    box-shadow: 0 0 10px var(--accent);
}

/* ── PANEL ── */
.panel {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--r); padding: 24px; height: 100%;
  animation: slideInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
  box-shadow: var(--shadow2);
}
.panel-title {
  font-size: 11px; font-weight: 800; letter-spacing: 2px;
  color: var(--text3); text-transform: uppercase; margin-bottom: 20px;
  display: flex; align-items: center; gap: 10px;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── COMPLAINT CARD ── */
.complaint-card {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: var(--r); padding: 22px; margin-bottom: 16px;
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
  cursor: default;
  position: relative; overflow: hidden;
  animation: popIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.complaint-card:hover { 
    background: var(--bg4); border-color: var(--accent); 
    transform: translateY(-5px) scale(1.02); 
    box-shadow: 0 15px 35px rgba(0,0,0,0.5); 
}
.card-active { 
    border-color: var(--accent) !important; 
    box-shadow: 0 0 25px rgba(124,115,255,0.2), var(--shadow); 
    animation: glowPulse 3s infinite;
}
.complaint-body {
  font-size: 15px; color: var(--text); line-height: 1.7;
  padding: 16px 20px; background: rgba(255,255,255,0.03);
  border-radius: var(--r2); border-left: 4px solid var(--accent);
  margin-bottom: 16px; font-style: italic; font-weight: 500;
}

/* ── DECISION CONTROLS ── */
.control-section { 
    background: linear-gradient(135deg, var(--bg3), var(--bg2)); 
    border: 1px solid var(--border); border-radius: var(--r); 
    padding: 24px; margin-top: 20px; box-shadow: var(--shadow); 
}
.gr-button { 
    font-family: 'Inter', sans-serif !important; 
    font-weight: 700 !important; 
    border-radius: var(--r2) !important; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; 
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 13px !important;
}
button[id*="reset_btn"] { 
    background: rgba(124,115,255,0.1) !important; 
    color: var(--accent) !important; 
    border: 1px solid var(--accent) !important;
}
button[id*="reset_btn"]:hover { 
    background: var(--accent) !important; 
    color: white !important; 
    box-shadow: 0 5px 20px rgba(124,115,255,0.4) !important;
}
button[id*="submit_btn"] { 
    background: linear-gradient(135deg, #34C759, #28A745) !important; 
    color: #050510 !important; 
    box-shadow: 0 8px 25px rgba(52,199,89,0.3) !important; 
}
button[id*="submit_btn"]:hover { 
    transform: translateY(-3px) scale(1.02) !important; 
    box-shadow: 0 12px 35px rgba(52,199,89,0.5) !important; 
    filter: brightness(1.2); 
}

/* ── REASONING PANEL ── */
.reasoning-panel { 
    background: var(--bg3); border: 1px solid var(--border); 
    border-radius: var(--r); padding: 24px;
    transition: all 0.5s ease;
    animation: slideInRight 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.tc-step { 
    transition: all 0.3s ease; 
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    padding: 12px !important;
    margin-bottom: 14px !important;
}
.tc-step:hover { background: rgba(255,255,255,0.05); transform: translateX(5px); }

/* ── TOOLTIPS (Custom implementation via CSS title hack or similar) ── */
/* Since Gradio doesn't output custom attributes easily, we use a general rule for better descriptions */
.tooltip-box {
    position: absolute; background: var(--bg4); border: 1px solid var(--accent);
    padding: 10px; border-radius: 8px; z-index: 1000; display: none;
    font-size: 12px; max-width: 200px; box-shadow: var(--shadow);
}

/* ── GRADIO OVERRIDES & INPUT ANIMATIONS ── */
.gr-panel { background: transparent !important; border: none !important; }
footer { display: none !important; }
.gr-box { box-shadow: none !important; }

.score-ring { width: 90px; height: 90px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; animation: float 4s ease-in-out infinite; }

/* Interactive Inputs */
textarea, input[type="text"], input[type="number"], select { 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
textarea:focus, input[type="text"]:focus { 
    transform: scale(1.01);
    box-shadow: 0 0 15px rgba(124,115,255,0.4) !important; 
    border-color: var(--accent) !important;
}
"""

# ── GRADIO APP ─────────────────────────────────────────────────────────────────
import time
DECISIONS = ["refund", "replace", "escalate", "apologize", "ignore", "investigate"]

with gr.Blocks(title="AI Operations Center -- CustomerSupportEnv") as demo:

    _init = demo_session()
    session_state = gr.State(demo_session)
    gr.HTML(f"<style>{PREMIUM_CSS}</style>")

    # ── LIVE METRIC BAR ──────────────────────────────────────────────────────
    live_metrics_html = gr.HTML(render_live_metrics(_init))

    # ── PAGE HEADER ──────────────────────────────────────────────────────────
    gr.HTML("""<div class="page-header">
  <div class="ph-title">🤖 AI Operations Center</div>
  <div class="ph-sub">Real-world AI customer support with personalized memory, multi-step episodes & dynamic rewards</div>
  <div class="ph-chips">
    <span class="ph-chip ac">OpenEnv Compliant</span>
    <span class="ph-chip ac">Meta Hackathon 2026</span>
    <span class="ph-chip">Agentic & Logic-driven</span>
    <span class="ph-chip">Triple-Check Reasoning</span>
  </div>
</div>
<div id="spotlight_container" class="feature-spotlight">
    <span class="fs-icon">✨</span>
    <div id="spotlight_content" class="fs-text">
        <span class="fs-highlight">Feature Spotlight:</span> Hover over any module tab to see detailed operations description.
    </div>
</div>
<script>
    const descriptions = {
        "Agentic Workflow": "Direct AI agent interaction for real-time triage, logic-driven resolution, and triple-check reasoning over customer tickets.",
        "Administrative Hub": "Central command for grading reports, historical episode logging, and overall system success rate metrics.",
        "Risk Assessment": "Advanced auditing of decision outcomes against company policies and category-specific accuracy KPIs.",
        "Network Monitor": "Live low-latency stream of the decision pipeline, tracking every action and reward signal in real-time.",
        "Resource Allocation": "Dynamic backlog management and budget tracking to optimize financial impact and customer throughput.",
        "System Health": "Environment diagnostics and connectivity status between the Gradio interface and the OpenEnv RL server.",
        "Strategy Optimizer": "Reward trend analysis and policy optimization curves to visualize the agent's multi-step learning progress."
    };

    function setupSpotlight() {
        const tabs = document.querySelectorAll('.tab-nav button');
        const content = document.getElementById('spotlight_content');
        
        tabs.forEach(tab => {
            const label = tab.innerText.trim().replace(/^[^a-zA-Z]+/, ''); // Remove icons
            tab.addEventListener('mouseenter', () => {
                if (descriptions[label]) {
                    content.style.opacity = '0';
                    content.style.transform = 'translateX(10px)';
                    setTimeout(() => {
                        content.innerHTML = '<span class="fs-highlight">' + label + ':</span> ' + descriptions[label];
                        content.style.opacity = '1';
                        content.style.transform = 'translateX(0)';
                    }, 150);
                }
            });
            tab.addEventListener('mouseleave', () => {
                // Return to default after delay or keep last
            });
        });
    }

    // Polling because Gradio tabs might load late
    let interval = setInterval(() => {
        if (document.querySelectorAll('.tab-nav button').length > 0) {
            setupSpotlight();
            clearInterval(interval);
        }
    }, 500);
</script>""")

    # ── MAIN TABBED INTERFACE ────────────────────────────────────────────────
    with gr.Tabs(elem_classes=["tab-nav"]):
        
        # 🟢 TAB 1: AGENTIC WORKFLOW (Main Triage)
        with gr.Tab("⚡ Agentic Workflow"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    task_info_html = gr.HTML(f'<div class="task-badge">🎯 Task: <strong>Demo Preview</strong> — Click <strong>🔄 New Episode</strong> to start live mode</div>')
                    
                    gr.HTML('<div class="panel-title">📨 Active Support Ticket</div>')
                    complaint_html = gr.HTML(render_complaint_card(_init["complaints"][0] if _init.get("complaints") else {}, active=True) or '<div class="episode-done">🏁 Demo Complete — Click New Episode to start</div>')
                    
                    hint_html = gr.HTML("")

                    # Controls
                    with gr.Group(elem_classes="control-section"):
                        gr.HTML('<div class="panel-title" style="margin-top:0px;">🎯 Decision Controls</div>')
                        with gr.Row():
                            decision_dd = gr.Dropdown(choices=DECISIONS, label="Action", value="investigate", interactive=True, scale=2)
                            confidence_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.75, label="Confidence [0–1]", interactive=True, scale=3)
                            with gr.Column(scale=1):
                                gr.HTML('<div style="height:24px;"></div>')
                                urgency_cb = gr.Checkbox(label="⚠️ Flag as Urgent", value=False, interactive=True)
                        
                        with gr.Accordion("🔍 Triple-Check Reasoning Input", open=True, elem_id="tc_accordion"):
                            with gr.Column():
                                s1_box = gr.Textbox(label="STEP 1: Sentiment & Policy Analysis", placeholder="e.g. Customer Arjun is enraged. Policy says 30-day window is active.", lines=1, interactive=True)
                                s2_box = gr.Textbox(label="STEP 2: Financial Impact Analysis", placeholder="e.g. Refund will cost $50. Budget is healthy at $800. Risk of churn is high.", lines=1, interactive=True)
                                s3_box = gr.Textbox(label="STEP 3: Personalized Resolution", placeholder="e.g. I'll handle this for you, Arjun.", lines=1, interactive=True)
                        
                        with gr.Row():
                            reset_btn = gr.Button("🔄 New Episode", elem_id="reset_btn", variant="primary", scale=1)
                            submit_btn = gr.Button("🚀 Process Ticket", elem_id="submit_btn", variant="secondary", scale=2)
                            autopilot_btn = gr.Button("🤖 Start Auto-Pilot", variant="primary", scale=1)
                            stop_auto_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
                        status_html = gr.HTML("")

                with gr.Column(scale=5):
                    gr.HTML('<div class="panel-title">🧠 AI Reasoning Breakdown</div>')
                    reasoning_html = gr.HTML(render_reasoning_panel({}, "", 0.0))

        # 🏢 TAB 2: ADMINISTRATIVE HUB (Grades & Logs)
        with gr.Tab("🏢 Administrative Hub"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-title">🎯 Grade Report</div>')
                    scorecard_html = gr.HTML(render_episode_scorecard(_init))
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-title">📝 Episode History</div>')
                    episode_log_html = gr.HTML(render_episode_log([]))

        # ⚖️ TAB 3: RISK ASSESSMENT (Compliance & Accuracy)
        with gr.Tab("⚖️ Risk Assessment"):
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="panel-title">📊 Category Accuracy & Policy Risk</div>')
                    category_chart_html = gr.HTML(render_category_chart(_init["history"]))
                    gr.HTML("""<div style="margin-top:20px; padding:15px; background:rgba(255,149,0,0.05); border-radius:10px; border:1px solid rgba(255,149,0,0.2);">
                        <div style="color:var(--orange); font-weight:700; margin-bottom:8px;">⚠️ POLICY RISK MONITOR</div>
                        <div style="font-size:12px; color:var(--text2); line-height:1.4;">
                            AI agents are monitored for policy compliance. Decisions like <strong>ignore</strong> on <strong>critical</strong> priority 
                            billing issues significantly increase risk scores. Ensure reasoning justifies any policy deviations.
                        </div>
                    </div>""")

        # 🌐 TAB 4: NETWORK MONITOR (Live Timeline)
        with gr.Tab("🌐 Network Monitor"):
            gr.HTML('<div class="panel-title">⏱️ Real-time Decision Stream</div>')
            history_html = gr.HTML(render_history_timeline(_init["history"]))

        # 📦 TAB 5: RESOURCE ALLOCATION (Queue & Budget)
        with gr.Tab("📦 Resource Allocation"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML('<div class="panel-title">📋 Backlog Queue</div>')
                    queue_html = gr.HTML("")
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-title">💰 Budget Utilization</div>')
                    gr.HTML(f"""<div style="background:var(--bg3); border:1px solid var(--border); border-radius:14px; padding:20px; text-align:center;">
                        <div style="font-size:12px; color:var(--text2); margin-bottom:10px;">VIRTUAL ACCOUNT BALANCE</div>
                        <div style="font-size:32px; font-weight:800; color:var(--green);">$ {_init['budget']:.0f}</div>
                        <div style="font-size:11px; color:var(--text3); margin-top:20px;">Usage Guidelines: Prioritize high-value customers for refunds only when necessary.</div>
                    </div>""")

        # 🏥 TAB 6: SYSTEM HEALTH (Environment Status)
        with gr.Tab("🏥 System Health"):
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="panel-title">🚦 Environment Status</div>')
                    gr.HTML(f"""<div style="display:flex; flex-direction:column; gap:16px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(52,199,89,0.1); padding:16px; border-radius:10px; border:1px solid rgba(52,199,89,0.3);">
                            <span style="font-weight:600;">OpenEnv Server</span>
                            <span style="color:var(--green); font-weight:800;">● ONLINE</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(108,99,255,0.1); padding:16px; border-radius:10px; border:1px solid rgba(108,99,255,0.3);">
                            <span style="font-weight:600;">Gradio Operations Platform</span>
                            <span style="color:var(--accent); font-weight:800;">● ACTIVE</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(0,0,0,0.2); padding:16px; border-radius:10px; border:1px solid var(--border);">
                            <span style="font-weight:600;">Data Refresh Rate</span>
                            <span style="color:var(--text2);">Dynamic (Event Driven)</span>
                        </div>
                    </div>""")

        # 📈 TAB 7: STRATEGY OPTIMIZER (Reward Performance)
        with gr.Tab("📈 Strategy Optimizer"):
            gr.HTML('<div class="panel-title">🚀 Learning Progress & Reward Curves</div>')
            reward_chart_html = gr.HTML(render_reward_chart(_init["reward_history"]))

    # ── CALLBACKS ─────────────────────────────────────────────────────────────
    def do_reset(_session):
        data = api_reset()
        if "error" in data:
            err_msg = data["error"]
            return (
                _session,
                render_live_metrics(_session),
                f'<div class="task-badge" style="color:#FF3B30;">❌ Server error: {err_msg} — Is the server running?</div>',
                '<div class="reasoning-placeholder"><div class="rp-icon">⚠️</div><div>Server not reachable. Run: uvicorn server.app:app --port 8000</div></div>',
                "", "",
                render_reasoning_panel({}, "", 0.0),
                render_history_timeline([]),
                render_reward_chart([]),
                render_category_chart([]),
                render_episode_scorecard(fresh_session()),
                "",
            )
        obs = data.get("observation", {})
        session = fresh_session()
        session["complaints"] = obs.get("complaints", [])
        session["episode_step"] = obs.get("episode_step", 0)
        session["max_steps"] = obs.get("max_steps", 5)
        session["budget"] = obs.get("budget_remaining", 1000.0)
        session["satisfaction"] = obs.get("satisfaction_score", 1.0)
        session["cumulative_reward"] = obs.get("cumulative_reward", 0.0)
        session["escalation_count"] = obs.get("escalation_count", 0)
        session["backlog_size"] = obs.get("backlog_size", 0)
        session["task_name"] = obs.get("task_name", "")
        session["task_level"] = obs.get("metadata", {}).get("task_level", obs.get("task_level", ""))
        session["done"] = obs.get("done", False)
        session["reward_history"] = []

        complaints = session["complaints"]
        active_c = complaints[0] if complaints else {}
        task_html = f"""<div class="task-badge">
  🎯 Task: <strong>{session['task_name']}</strong> |
  Level: <strong style="text-transform:uppercase;">{session['task_level']}</strong> |
  Steps: <strong>{session['max_steps']}</strong>
  <div style="font-size:11px;color:var(--text2);margin-top:4px;">{obs.get('task_description','')}</div>
</div>"""
        c_html = render_complaint_card(active_c, active=True) if active_c else '<div class="reasoning-placeholder"><div class="rp-icon">📭</div><div>No complaints loaded</div></div>'
        q_html = "".join(render_complaint_card(c) for c in complaints[1:])
        hint = get_baseline_hint(active_c) if active_c else ""

        return (
            session,
            render_live_metrics(session),
            task_html, c_html, hint, q_html,
            render_reasoning_panel({}, "", 0.0),
            render_history_timeline([]),
            render_reward_chart([]),
            render_category_chart([]),
            render_episode_scorecard(session),
            '<div class="status-ok">✅ New episode started — make your first decision!</div>',
        )

    def do_submit(decision, confidence, urgency, s1, s2, s3, _session):
        if _session.get("done"):
            return (
                _session, render_live_metrics(_session),
                '<div class="episode-done">🏁 Episode Complete! Click New Episode to play again.</div>',
                "", "",
                render_reasoning_panel({}, "", 0.0),
                render_history_timeline(_session.get("history", [])),
                render_reward_chart(_session.get("reward_history", [])),
                render_category_chart(_session.get("history", [])),
                render_episode_scorecard(_session),
                "Episode already done.",
            )
        complaints = _session.get("complaints", [])
        if not complaints:
            return (
                _session, render_live_metrics(_session),
                '<div class="reasoning-placeholder"><div class="rp-icon">📭</div><div>No active complaint. Click New Episode to start.</div></div>',
                "", "",
                render_reasoning_panel({}, "", 0.0),
                render_history_timeline(_session.get("history", [])),
                render_reward_chart(_session.get("reward_history", [])),
                render_category_chart(_session.get("history", [])),
                render_episode_scorecard(_session),
                "No complaint to evaluate.",
            )
        active_c = complaints[0]
        action_payload = {
            "complaint_id": active_c["complaint_id"],
            "decision": decision, "confidence": confidence,
            "reasoning": f"{s1} | {s2} | {s3}", "urgency_flag": urgency,
        }
        data = api_step(action_payload)
        if "error" in data:
            return (
                _session, render_live_metrics(_session),
                f'<div style="color:var(--red);padding:16px;">❌ API Error: {data["error"]}</div>',
                "", "",
                f'<div style="color:var(--red);padding:16px;">❌ Step failed: {data["error"]}</div>',
                render_history_timeline(_session.get("history", [])),
                render_reward_chart(_session.get("reward_history", [])),
                render_category_chart(_session.get("history", [])),
                render_episode_scorecard(_session),
                "Server error — check backend is running.",
            )

        obs = data.get("observation", {}); step_reward = data.get("reward", 0.0); done = data.get("done", False)
        _session["complaints"] = obs.get("complaints", [])
        _session["episode_step"] = obs.get("episode_step", 0)
        _session["cumulative_reward"] = obs.get("cumulative_reward", 0.0)
        _session["satisfaction"] = obs.get("satisfaction_score", 1.0)
        _session["budget"] = obs.get("budget_remaining", 1000.0)
        _session["escalation_count"] = obs.get("escalation_count", 0)
        _session["backlog_size"] = obs.get("backlog_size", 0)
        _session["done"] = done
        _session["reward_history"] = _session.get("reward_history", []) + [step_reward]

        feedback_list = obs.get("last_step_feedback", []); last_fb = feedback_list[0] if feedback_list else {}
        if last_fb:
            rb = last_fb.get("reward_breakdown", {})
            _session["history"].append({
                "step": _session["episode_step"],
                "complaint_text": last_fb.get("complaint_text", ""),
                "customer_name": active_c.get("customer_name", ""),
                "decision": decision,
                "verdict": last_fb.get("verdict", ""),
                "correct": last_fb.get("correct", False),
                "total_reward": rb.get("total", 0.0),
                "category": last_fb.get("category", "unknown"),
            })

        new_complaints = _session["complaints"]
        if done:
            c_html = '<div class="episode-done">🏁 Episode Complete! Click New Episode to play again.</div>'
            q_html = ""; hint = ""; status = f"✅ Episode done! Final reward: {_session['cumulative_reward']:+.4f}"
        elif new_complaints:
            c_html = render_complaint_card(new_complaints[0], active=True)
            q_html = "".join(render_complaint_card(c) for c in new_complaints[1:])
            hint = get_baseline_hint(new_complaints[0])
            reward_col = "#34C759" if step_reward >= 0 else "#FF3B30"
            status = f'<div class="status-ok">Step {_session["episode_step"]} — Reward: <strong style="color:{reward_col};">{step_reward:+.4f}</strong></div>'
        else:
            c_html = '<div class="reasoning-placeholder"><div class="rp-icon">⏳</div><div>Waiting for next batch...</div></div>'
            q_html = ""; hint = ""; status = "Waiting for next complaint batch…"

        return (
            _session, render_live_metrics(_session),
            c_html, hint, q_html,
            render_reasoning_panel(last_fb, f"{s1} | {s2} | {s3}", confidence, s1, s2, s3),
            render_history_timeline(_session["history"]),
            render_reward_chart(_session["reward_history"]),
            render_category_chart(_session["history"]),
            render_episode_scorecard(_session),
            status,
        )

    # ── WIRE BUTTONS ──────────────────────────────────────────────────────────
    reset_outputs = [
        session_state, live_metrics_html, task_info_html,
        complaint_html, hint_html, queue_html,
        reasoning_html, history_html,
        reward_chart_html, category_chart_html, scorecard_html,
        status_html,
    ]
    submit_outputs = [
        session_state, live_metrics_html,
        complaint_html, hint_html, queue_html,
        reasoning_html, history_html,
        reward_chart_html, category_chart_html, scorecard_html,
        status_html,
    ]

    reset_btn.click(fn=do_reset, inputs=[session_state], outputs=reset_outputs)
    submit_btn.click(fn=do_submit, inputs=[decision_dd, confidence_slider, urgency_cb, s1_box, s2_box, s3_box, session_state], outputs=submit_outputs)

    def run_autopilot(_session):
        while not _session.get("done") and _session.get("complaints"):
            active_c = _session["complaints"][0]
            cat = active_c.get("category", "")
            pri = active_c.get("priority", "low")
            text = active_c.get("complaint_text", "").lower()
            clues = active_c.get("context_clues", [])
            decision = CAT_PRIORITY_MAP.get(cat, {}).get(pri, "apologize")
            for kw, act in SENT_OVERRIDE.items():
                if kw in text or any(kw in c for c in clues): decision = act; break
            
            yield do_submit(decision, 0.95, False, "Auto-Pilot: Sentiment Analyzed", "Auto-Pilot: Financials OK", f"Auto-Pilot: {decision.upper()} action deployed.", _session)
            time.sleep(1.8)

    auto_event = autopilot_btn.click(fn=run_autopilot, inputs=[session_state], outputs=submit_outputs)
    stop_auto_btn.click(fn=lambda: None, cancels=[auto_event])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7861, share=False,
        theme=gr.themes.Base(
            primary_hue="violet", secondary_hue="purple", neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter"),
        ),
    )