"""
AI Dashboard — CustomerSupportEnv
Enhanced with live charts, category analytics, and complaint showcase.
Run: python app.py
"""

import json, random, time
from typing import Any, Dict, List, Optional
import gradio as gr
import requests

API = "http://127.0.0.1:8000"

# ── STATE ──────────────────────────────────────────────────────────────────────
def fresh_session():
    return {
        "complaints": [], "active_idx": 0, "episode_step": 0, "max_steps": 5,
        "cumulative_reward": 0.0, "satisfaction": 1.0, "budget": 1000.0,
        "escalation_count": 0, "backlog_size": 0, "task_name": "", "task_level": "",
        "history": [], "reward_history": [], "episodes_log": [], "done": False,
    }

def demo_session():
    """Pre-populated with realistic demo data shown on first load."""
    hist = [
        {"step":1,"complaint_text":"Double-charged on credit card. Needs immediate resolution.","decision":"refund","verdict":"✅ Optimal","correct":True,"total_reward":0.762,"category":"billing"},
        {"step":2,"complaint_text":"Order arrived 3 weeks late and I need a refund.","decision":"apologize","verdict":"❌ Wrong","correct":False,"total_reward":-0.215,"category":"delivery"},
        {"step":3,"complaint_text":"Product stopped working after just one day.","decision":"replace","verdict":"✅ Optimal","correct":True,"total_reward":0.621,"category":"quality"},
        {"step":4,"complaint_text":"App crashes every time I try to checkout.","decision":"investigate","verdict":"✅ Optimal","correct":True,"total_reward":0.480,"category":"technical"},
        {"step":5,"complaint_text":"Return window expired by exactly one day.","decision":"escalate","verdict":"⚡ Acceptable","correct":True,"total_reward":0.310,"category":"policy"},
        {"step":6,"complaint_text":"VIP: small scratch on product, wants acknowledgment.","decision":"apologize","verdict":"✅ Optimal","correct":True,"total_reward":0.540,"category":"quality"},
        {"step":7,"complaint_text":"Package shows delivered but never received.","decision":"investigate","verdict":"✅ Optimal","correct":True,"total_reward":0.590,"category":"delivery"},
        {"step":8,"complaint_text":"Cancelled subscription but still charged last month.","decision":"refund","verdict":"✅ Optimal","correct":True,"total_reward":0.820,"category":"billing"},
    ]
    return {
        "complaints": [], "active_idx": 0, "episode_step": 8, "max_steps": 5,
        "cumulative_reward": 3.908, "satisfaction": 0.87, "budget": 760.0,
        "escalation_count": 1, "backlog_size": 0, "task_name": "Demo Preview",
        "task_level": "medium", "history": hist,
        "reward_history": [0.762, -0.215, 0.621, 0.480, 0.310, 0.540, 0.590, 0.820],
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

# ── RENDER: COMPLAINT CARD ─────────────────────────────────────────────────────
def render_complaint_card(c:dict, active:bool=False)->str:
    if not c: return ""
    pri=c.get("priority","medium"); cat=c.get("category","delivery"); tier=c.get("customer_tier","regular")
    col=PC.get(pri,"#888"); amb=c.get("ambiguity_level",0.3); sent=c.get("sentiment_score",0.0)
    border="3px solid #5E5CE6" if active else f"1px solid {col}40"
    sl="😡 Enraged" if sent<-0.7 else "😠 Frustrated" if sent<-0.3 else "😐 Neutral" if sent<0.1 else "🙂 Calm"
    ab=int(amb*10)
    clues="".join(f'<span class="clue-tag">{cl}</span>' for cl in c.get("context_clues",[])[:4])
    return f"""<div class="complaint-card" style="border:{border};border-left:4px solid {col};">
  <div class="complaint-header">
    <span class="cat-badge" style="background:{col}20;color:{col};">{CI.get(cat,"📋")} {cat.upper()}</span>
    <span class="priority-badge" style="background:{col};color:#000;">{pri.upper()}</span>
    <span class="tier-badge">{TI.get(tier,"👤")} {tier.upper()}</span>
  </div>
  <p class="complaint-text">"{c.get('text','')}"</p>
  <div class="meta-row">
    <span>📅 {c.get('days_since_purchase',0)}d ago</span>
    <span>🔁 {c.get('previous_complaints',0)} prior</span>
    <span>💵 ${c.get('estimated_order_value',0):.2f}</span>
  </div>
  <div class="meta-row" style="margin-top:6px;">
    <span>Sentiment: {sl}</span>
    <span>Ambiguity: {'🟨'*ab}{'⬜'*(10-ab)}</span>
  </div>
  <div class="clues-row">{clues}</div>
  <div style="color:#888;font-size:11px;margin-top:4px;">ID: {c.get('complaint_id','')[:16]}…</div>
</div>"""

# ── RENDER: REASONING PANEL ───────────────────────────────────────────────────
def render_reasoning_panel(fb:dict, reasoning:str, confidence:float)->str:
    if not fb: return '<div class="reasoning-empty">Make a decision to see AI reasoning…</div>'
    verdict=fb.get("verdict",""); col=VC.get(verdict,"#888"); rb=fb.get("reward_breakdown",{}); cp=int(confidence*100)
    ca=", ".join(fb.get("correct_actions",[]))
    bd=f"""<div class="breakdown-grid">
  <div class="bd-row"><span>Base Correctness</span><span style="color:{col};">{rb.get('weighted_score',0):+.3f}</span></div>
  <div class="bd-row"><span>Priority × Tier</span><span>{rb.get('priority_weight',1):.1f} × {rb.get('tier_weight',1):.1f}</span></div>
  <div class="bd-row"><span>Confidence Calib.</span><span style="color:{'#34C759' if rb.get('calibration_bonus',0)>=0 else '#FF3B30'};">{rb.get('calibration_bonus',0):+.3f}</span></div>
  <div class="bd-row"><span>Budget Penalty</span><span style="color:{'#FF3B30' if rb.get('budget_penalty',0)<0 else '#34C759'};">{rb.get('budget_penalty',0):+.3f}</span></div>
  <div class="bd-row total"><span>Total Reward</span><span style="color:{col};">{rb.get('total',0):+.3f}</span></div>
</div>"""
    return f"""<div class="reasoning-panel">
  <div class="reasoning-header">
    <span class="verdict-badge" style="background:{col}20;color:{col};border:1px solid {col};">{verdict}</span>
    <span class="conf-meter">Confidence: <strong>{cp}%</strong>
      <span class="conf-bar"><span style="width:{cp}%;background:{col};"></span></span></span>
  </div>
  <div class="reasoning-text"><strong>🧠 Reasoning:</strong><br><em>{reasoning or 'No reasoning provided'}</em></div>
  <div class="optimal-note">✨ Optimal: <strong>{ca}</strong></div>
  {bd}
  <div class="feedback-extras"><span>💰 Cost: ${fb.get('cost_incurred',0)}</span><span>😊 Sat Δ: {fb.get('satisfaction_delta',0):+.3f}</span></div>
</div>"""

# ── RENDER: ANALYTICS PANEL ───────────────────────────────────────────────────
def render_analytics_panel(session:dict)->str:
    hist=session.get("history",[]); budget=session.get("budget",1000.0); sat=session.get("satisfaction",1.0)
    cr=session.get("cumulative_reward",0.0); step=session.get("episode_step",0); msteps=session.get("max_steps",5)
    esc=session.get("escalation_count",0); bl=session.get("backlog_size",0)
    cp=int(len([h for h in hist if h.get("correct")])/max(len(hist),1)*100) if hist else 0
    sc={"#34C759" if sat>0.7 else "#FF9500" if sat>0.4 else "#FF3B30"}; crc="#34C759" if cr>=0 else "#FF3B30"
    dcc="#34C759" if cp>65 else "#FF9500" if cp>40 else "#FF3B30"; pp=int(step/max(msteps,1)*100)
    sc_color="#34C759" if sat>0.7 else "#FF9500" if sat>0.4 else "#FF3B30"
    dcounts={}
    for h in hist: d=h.get("decision","?"); dcounts[d]=dcounts.get(d,0)+1
    dist="".join(
        f'<div class="dist-row"><span>{DI.get(d,"?")} {d}</span>'
        f'<span class="dist-bar"><span style="width:{int(c/max(len(hist),1)*100)}%;background:{DC.get(d,"#5E5CE6")};"></span></span>'
        f'<span>{c}</span></div>'
        for d,c in sorted(dcounts.items(),key=lambda x:-x[1])
    )
    return f"""<div class="analytics-panel">
  <div class="stat-grid">
    <div class="stat-card"><div class="stat-val" style="color:{crc};">{cr:+.2f}</div><div class="stat-label">Cumulative Reward</div></div>
    <div class="stat-card"><div class="stat-val" style="color:{sc_color};">{int(sat*100)}%</div><div class="stat-label">Satisfaction</div></div>
    <div class="stat-card"><div class="stat-val">💵 ${budget:.0f}</div><div class="stat-label">Budget Left</div></div>
    <div class="stat-card"><div class="stat-val" style="color:{dcc};">{cp}%</div><div class="stat-label">Accuracy</div></div>
  </div>
  <div class="progress-section">
    <div class="prog-label">Episode Progress: Step {step} / {msteps}</div>
    <div class="prog-bar"><div class="prog-fill" style="width:{pp}%;"></div></div>
  </div>
  <div class="mini-stats"><span>⬆️ Escalations: {esc}</span><span>📋 Backlog: {bl}</span><span>📊 Decisions: {len(hist)}</span></div>
  <div class="dist-section">
    <div class="dist-title">Decision Distribution</div>
    {dist or '<div style="color:#888;font-size:12px;">No decisions yet</div>'}
  </div>
</div>"""

# ── RENDER: HISTORY TIMELINE ───────────────────────────────────────────────────
def render_history_timeline(history:list)->str:
    if not history: return '<div class="history-empty">No decisions made yet in this episode.</div>'
    rows=""
    for h in reversed(history[-8:]):
        verdict=h.get("verdict",""); col=VC.get(verdict,"#888")
        rows+=f"""<div class="timeline-item">
  <div class="tl-step">S{h.get('step','?')}</div>
  <div class="tl-body">
    <span class="tl-text">{h.get('complaint_text','')[:45]}…</span>
    <span class="tl-action">{DI.get(h.get('decision',''),'?')} {h.get('decision','')}</span>
    <span class="tl-verdict" style="color:{col};">{verdict}</span>
    <span class="tl-reward" style="color:{col};">{h.get('total_reward',0):+.3f}</span>
  </div>
</div>"""
    return f'<div class="timeline">{rows}</div>'

# ── RENDER: REWARD CHART ───────────────────────────────────────────────────────
def render_reward_chart(reward_history:list)->str:
    if not reward_history:
        return '<div class="chart-container"><div class="chart-title">📈 Step Reward Trend</div><div class="chart-placeholder">Submit decisions to see the reward trend chart…</div></div>'
    W,H,PAD=380,110,20
    rh=[0.0]+list(reward_history) if len(reward_history)==1 else list(reward_history)
    n=len(rh); min_r=min(min(rh),-0.1); max_r=max(max(rh),0.1); rng=max_r-min_r
    def sx(i): return PAD+(i/(n-1))*(W-2*PAD)
    def sy(r): return H-PAD-((r-min_r)/rng)*(H-2*PAD)
    pts=[(sx(i),sy(r)) for i,r in enumerate(rh)]; zero_y=sy(0)
    pd=f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"+"".join(f" L {x:.1f} {y:.1f}" for x,y in pts[1:])
    ad=pd+f" L {pts[-1][0]:.1f} {zero_y:.1f} L {pts[0][0]:.1f} {zero_y:.1f} Z"
    circles="".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{"#34C759" if rh[i]>=0 else "#FF3B30"}" stroke="#0A0A0F" stroke-width="1.5"/>' for i,(x,y) in enumerate(pts))
    yl="".join(f'<text x="{PAD-4}" y="{sy(v)+4:.1f}" text-anchor="end" font-size="9" fill="{"#34C759" if v>0 else "#FF3B30" if v<0 else "#666"}">{v:+.1f}</text><line x1="{PAD}" y1="{sy(v):.1f}" x2="{W-PAD}" y2="{sy(v):.1f}" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>' for v in [max_r,0,min_r])
    xl="".join(f'<text x="{sx(i):.1f}" y="{H-2}" text-anchor="middle" font-size="9" fill="#555">S{i}</text>' for i in range(n))
    zl=f'<line x1="{PAD}" y1="{zero_y:.1f}" x2="{W-PAD}" y2="{zero_y:.1f}" stroke="rgba(255,255,255,0.2)" stroke-width="1" stroke-dasharray="4,3"/>'
    avg_r=sum(rh)/len(rh); last_r=rh[-1]
    return f"""<div class="chart-container">
  <div class="chart-header-row">
    <div class="chart-title">📈 Step Reward Trend</div>
    <div style="display:flex;gap:10px;">
      <span style="font-size:11px;font-weight:600;color:{'#34C759' if avg_r>=0 else '#FF3B30'};">Avg {avg_r:+.3f}</span>
      <span style="font-size:11px;font-weight:600;color:{'#34C759' if last_r>=0 else '#FF3B30'};">Last {last_r:+.3f}</span>
    </div>
  </div>
  <svg viewBox="0 0 {W} {H}" style="width:100%;height:{H}px;display:block;">
    <defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5E5CE6" stop-opacity="0.4"/>
      <stop offset="100%" stop-color="#5E5CE6" stop-opacity="0.0"/>
    </linearGradient></defs>
    {yl}{zl}<path d="{ad}" fill="url(#rg)"/>
    <path d="{pd}" fill="none" stroke="#5E5CE6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    {circles}{xl}
  </svg>
</div>"""

# ── RENDER: CATEGORY CHART ─────────────────────────────────────────────────────
def render_category_chart(history:list)->str:
    if not history:
        return '<div class="chart-container"><div class="chart-title">📊 Category Performance</div><div class="chart-placeholder">Submit decisions to see per-category accuracy…</div></div>'
    cat_data={}
    for h in history:
        cat=h.get("category","unknown")
        if cat not in cat_data: cat_data[cat]={"correct":0,"total":0,"reward":0.0}
        cat_data[cat]["total"]+=1; cat_data[cat]["reward"]+=h.get("total_reward",0.0)
        if h.get("correct"): cat_data[cat]["correct"]+=1
    rows=""
    for cat,d in sorted(cat_data.items()):
        pct=int(d["correct"]/max(d["total"],1)*100); avg_r=d["reward"]/max(d["total"],1)
        col="#34C759" if pct>65 else "#FF9500" if pct>40 else "#FF3B30"
        rc="#34C759" if avg_r>=0 else "#FF3B30"
        rows+=f"""<div class="cat-perf-row">
  <span class="cat-icon-label">{CI.get(cat,"📋")} {cat}</span>
  <div class="cat-bar-wrap">
    <div class="cat-bar-track"><div class="cat-bar-fill" style="width:{pct}%;background:{col};"></div></div>
    <span class="cat-pct-label" style="color:{col};">{pct}%</span>
  </div>
  <span style="font-size:11px;font-weight:600;color:{rc};min-width:42px;text-align:right;">{avg_r:+.2f}</span>
</div>"""
    return f'<div class="chart-container"><div class="chart-title">📊 Category Accuracy</div>{rows}<div style="display:flex;gap:12px;margin-top:8px;font-size:10px;color:var(--text-muted);"><span style="color:#34C759;">■ >65% good</span><span style="color:#FF9500;">■ >40% ok</span><span style="color:#FF3B30;">■ needs work</span></div></div>'

# ── RENDER: ENV OVERVIEW ───────────────────────────────────────────────────────
SAMPLE_SHOWCASE=[
    {"cat":"💳 Billing",  "text":"Double-charged on credit card","level":"Easy","c":"#5E5CE6"},
    {"cat":"📦 Delivery", "text":"Package marked delivered but never arrived","level":"Medium","c":"#FF9500"},
    {"cat":"🔧 Quality",  "text":"Laptop warranty voided by repair centre","level":"Hard","c":"#FF3B30"},
    {"cat":"🖥️ Technical","text":"App crashes every time at checkout","level":"Medium","c":"#34C759"},
    {"cat":"📜 Policy",   "text":"Return window expired by exactly one day","level":"Medium","c":"#BF5AF2"},
    {"cat":"🔧 Quality",  "text":"VIP: small scratch, wants acknowledgment only","level":"Hard","c":"#FF6B6B"},
]

def render_env_overview()->str:
    cards="".join(
        f'<div class="sc-card"><div class="sc-cat" style="color:{s["c"]};">{s["cat"]}</div>'
        f'<div class="sc-text">"{s["text"]}"</div>'
        f'<div class="sc-level" style="background:{s["c"]}18;color:{s["c"]};">{s["level"]}</div></div>'
        for s in SAMPLE_SHOWCASE
    )
    stats="".join(f'<div class="env-stat"><div class="env-stat-val">{v}</div><div class="env-stat-label">{l}</div></div>' for v,l in [("5","Categories"),("3","Task Levels"),("6","Actions"),("7+","Reward Components")])
    return f'<div class="chart-container"><div class="chart-title">🌍 Environment Overview — Sample Complaint Types</div><div class="sc-grid">{cards}</div><div class="env-stats-row">{stats}</div></div>'

# ── CSS ────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg-primary:#0A0A0F;--bg-secondary:#111118;--bg-card:#16161F;--bg-card-hover:#1C1C28;--accent:#5E5CE6;--accent-dim:rgba(94,92,230,0.15);--text-primary:#E8E8F0;--text-secondary:#9898B0;--text-muted:#5A5A70;--border:rgba(255,255,255,0.08);--green:#34C759;--red:#FF3B30;--yellow:#FFCC00;--orange:#FF9500;--purple:#BF5AF2;--radius:12px;--radius-sm:8px;--shadow:0 4px 24px rgba(0,0,0,0.4);}
*{box-sizing:border-box;font-family:'Inter',sans-serif!important;}
body,.gradio-container{background:var(--bg-primary)!important;color:var(--text-primary)!important;}
.main-header{background:linear-gradient(135deg,#0A0A1A 0%,#12122A 50%,#1A0A2A 100%);border:1px solid rgba(94,92,230,0.3);border-radius:16px;padding:28px 36px;margin-bottom:20px;position:relative;overflow:hidden;}
.main-header::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle at 30% 50%,rgba(94,92,230,0.08) 0%,transparent 60%),radial-gradient(circle at 70% 50%,rgba(191,90,242,0.06) 0%,transparent 60%);pointer-events:none;}
.main-header h1{font-size:28px;font-weight:700;background:linear-gradient(135deg,#fff 0%,#BF5AF2 50%,#5E5CE6 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 6px 0;}
.main-header p{color:var(--text-secondary);margin:0;font-size:14px;}
.header-chips{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;}
.chip{background:rgba(255,255,255,0.06);border:1px solid var(--border);border-radius:20px;padding:4px 12px;font-size:12px;color:var(--text-secondary);}
.chip.accent{background:var(--accent-dim);border-color:rgba(94,92,230,0.4);color:#9B9AF8;}
.section-label{font-size:11px;font-weight:600;letter-spacing:1.5px;color:var(--text-muted);text-transform:uppercase;margin-bottom:10px;}
.complaint-card{background:var(--bg-card);border-radius:var(--radius);padding:16px;margin-bottom:12px;transition:all 0.2s ease;}
.complaint-card:hover{background:var(--bg-card-hover);transform:translateY(-1px);box-shadow:var(--shadow);}
.complaint-header{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap;}
.cat-badge,.priority-badge,.tier-badge{border-radius:6px;padding:3px 10px;font-size:11px;font-weight:600;letter-spacing:0.5px;}
.tier-badge{background:rgba(255,255,255,0.08);color:var(--text-secondary);}
.complaint-text{font-size:14px;color:var(--text-primary);line-height:1.5;margin:8px 0;padding:10px 12px;background:rgba(255,255,255,0.04);border-radius:8px;border-left:2px solid var(--accent);}
.meta-row{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);flex-wrap:wrap;}
.clues-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;}
.clue-tag{background:rgba(94,92,230,0.1);border:1px solid rgba(94,92,230,0.25);color:#9B9AF8;border-radius:4px;padding:2px 8px;font-size:11px;}
.feedback-row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-radius:8px;background:var(--bg-card);margin-bottom:6px;gap:12px;border:1px solid var(--border);}
.fb-left{display:flex;flex-direction:column;gap:2px;}.fb-verdict{font-size:12px;font-weight:600;}.fb-text{font-size:12px;color:var(--text-secondary);}
.fb-right{display:flex;flex-direction:column;align-items:flex-end;gap:2px;}.fb-decision{font-size:12px;}.fb-reward{font-size:13px;font-weight:700;}
.reasoning-empty{color:var(--text-muted);font-size:13px;padding:20px;text-align:center;}
.reasoning-panel{background:var(--bg-card);border-radius:var(--radius);padding:18px;border:1px solid var(--border);}
.reasoning-header{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap;}
.verdict-badge{border-radius:6px;padding:4px 12px;font-size:12px;font-weight:600;}
.conf-meter{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text-secondary);}
.conf-bar{width:80px;height:6px;background:rgba(255,255,255,0.1);border-radius:3px;overflow:hidden;}
.conf-bar span{display:block;height:100%;border-radius:3px;transition:width 0.4s ease;}
.reasoning-text{background:rgba(255,255,255,0.04);border-radius:8px;padding:12px;font-size:13px;line-height:1.6;margin-bottom:12px;border-left:2px solid var(--accent);}
.optimal-note{font-size:12px;color:var(--green);margin-bottom:14px;}
.breakdown-grid{background:rgba(0,0,0,0.2);border-radius:8px;padding:12px;}
.bd-row{display:flex;justify-content:space-between;padding:5px 0;font-size:12px;border-bottom:1px solid rgba(255,255,255,0.05);color:var(--text-secondary);}
.bd-row.total{font-weight:700;color:var(--text-primary);border-bottom:none;margin-top:4px;}
.feedback-extras{display:flex;gap:16px;margin-top:10px;font-size:12px;color:var(--text-secondary);}
.analytics-panel{display:flex;flex-direction:column;gap:14px;}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.stat-card{background:var(--bg-card);border-radius:var(--radius-sm);padding:14px;border:1px solid var(--border);text-align:center;}
.stat-val{font-size:22px;font-weight:700;margin-bottom:4px;}.stat-label{font-size:11px;color:var(--text-muted);}
.progress-section{background:var(--bg-card);border-radius:var(--radius-sm);padding:12px;border:1px solid var(--border);}
.prog-label{font-size:12px;color:var(--text-secondary);margin-bottom:8px;}
.prog-bar{background:rgba(255,255,255,0.08);border-radius:4px;height:8px;overflow:hidden;}
.prog-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--purple));border-radius:4px;transition:width 0.5s ease;}
.mini-stats{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);}
.dist-section{background:var(--bg-card);border-radius:var(--radius-sm);padding:12px;border:1px solid var(--border);}
.dist-title{font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:8px;}
.dist-row{display:flex;align-items:center;gap:8px;font-size:12px;margin-bottom:4px;color:var(--text-secondary);}
.dist-bar{flex:1;height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden;}
.dist-bar span{display:block;height:100%;border-radius:2px;transition:width 0.6s ease;}
.history-empty{color:var(--text-muted);font-size:12px;padding:16px;text-align:center;}
.timeline{display:flex;flex-direction:column;gap:6px;}
.timeline-item{display:flex;gap:10px;align-items:flex-start;}
.tl-step{width:28px;height:28px;background:var(--accent-dim);border:1px solid rgba(94,92,230,0.4);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#9B9AF8;flex-shrink:0;}
.tl-body{background:var(--bg-card);border-radius:8px;padding:8px 12px;flex:1;border:1px solid var(--border);}
.tl-text{display:block;font-size:11px;color:var(--text-secondary);margin-bottom:4px;}
.tl-action{font-size:11px;margin-right:8px;}.tl-verdict{font-size:11px;font-weight:600;margin-right:8px;}.tl-reward{font-size:11px;font-weight:700;}
.task-badge{background:linear-gradient(135deg,var(--accent-dim),rgba(191,90,242,0.1));border:1px solid rgba(94,92,230,0.3);border-radius:10px;padding:10px 16px;font-size:13px;color:var(--text-primary);margin-bottom:12px;}
.task-badge strong{color:#9B9AF8;}
.episode-done{background:linear-gradient(135deg,rgba(52,199,89,0.1),rgba(94,92,230,0.1));border:1px solid rgba(52,199,89,0.3);border-radius:12px;padding:20px;text-align:center;font-size:16px;font-weight:600;color:var(--green);}
.gr-button{font-family:'Inter',sans-serif!important;}
button[id*="reset_btn"]{background:linear-gradient(135deg,#5E5CE6,#BF5AF2)!important;color:white!important;}
button[id*="submit_btn"]{background:linear-gradient(135deg,#34C759,#30D158)!important;color:#000!important;font-weight:700!important;}
.gr-input,.gr-box,.gr-form{background:var(--bg-card)!important;border-color:var(--border)!important;}
label,.gr-input label{color:var(--text-secondary)!important;font-size:12px!important;}
textarea,input[type="text"],input[type="number"]{background:var(--bg-secondary)!important;color:var(--text-primary)!important;border-color:var(--border)!important;}
/* ── CHARTS (NEW) ── */
.chart-container{background:var(--bg-card);border-radius:var(--radius);padding:16px;border:1px solid var(--border);margin-bottom:12px;}
.chart-header-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}
.chart-title{font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:10px;letter-spacing:0.5px;}
.chart-placeholder{color:var(--text-muted);font-size:12px;padding:24px;text-align:center;border:1px dashed var(--border);border-radius:8px;}
.cat-perf-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12px;}
.cat-icon-label{min-width:90px;color:var(--text-secondary);}
.cat-bar-wrap{display:flex;align-items:center;gap:8px;flex:1;}
.cat-bar-track{flex:1;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;}
.cat-bar-fill{height:100%;border-radius:3px;transition:width 0.6s ease;}
.cat-pct-label{min-width:32px;font-weight:600;font-size:11px;}
/* ── SAMPLE SHOWCASE ── */
.sc-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;}
.sc-card{background:var(--bg-secondary);border-radius:8px;padding:10px;border:1px solid var(--border);}
.sc-cat{font-size:11px;font-weight:700;margin-bottom:4px;}
.sc-text{font-size:11px;color:var(--text-secondary);margin-bottom:6px;line-height:1.4;}
.sc-level{font-size:10px;font-weight:600;padding:2px 8px;border-radius:4px;display:inline-block;}
.env-stats-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:4px;}
.env-stat{background:var(--bg-secondary);border-radius:8px;padding:8px 14px;border:1px solid var(--border);text-align:center;}
.env-stat-val{font-size:20px;font-weight:700;color:#9B9AF8;}
.env-stat-label{font-size:10px;color:var(--text-muted);}
/* ── BOTTOM DIVIDER ── */
.bottom-section-divider{border-top:1px solid var(--border);margin:16px 0;padding-top:16px;}
/* SCORECARD */
.scorecard{background:var(--bg-card);border-radius:var(--radius);padding:16px;border:1px solid var(--border);}
.scorecard-title{font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:12px;}
.scorecard-empty{color:var(--text-muted);font-size:12px;padding:20px;text-align:center;border:1px dashed var(--border);border-radius:8px;}
.scorecard-body{display:flex;gap:16px;align-items:center;margin-bottom:14px;}
.score-ring{width:80px;height:80px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.score-ring-inner{width:60px;height:60px;border-radius:50%;background:var(--bg-primary);display:flex;flex-direction:column;align-items:center;justify-content:center;}
.score-val{font-size:15px;font-weight:800;line-height:1;}
.score-label{font-size:9px;color:var(--text-muted);}
.score-details{flex:1;display:flex;flex-direction:column;gap:5px;}
.score-verdict{font-size:16px;font-weight:700;margin-bottom:4px;}
.score-row{display:flex;justify-content:space-between;font-size:11px;color:var(--text-secondary);}
.score-bar-wrap{display:flex;align-items:center;gap:8px;}
.score-bar-track{flex:1;height:8px;background:rgba(255,255,255,0.08);border-radius:4px;overflow:hidden;}
/* BASELINE HINT */
.baseline-hint{display:flex;align-items:center;gap:10px;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:8px;padding:8px 12px;margin-top:8px;flex-wrap:wrap;}
.hint-label{font-size:11px;color:var(--text-muted);}
.hint-action{font-size:12px;font-weight:700;padding:3px 10px;border-radius:6px;}
.hint-note{font-size:11px;color:var(--text-muted);}
/* EPISODE LOG */
.ep-log-row{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:11px;}
.ep-num{min-width:28px;color:var(--text-muted);}
.ep-level{min-width:56px;font-weight:600;font-size:10px;}
.ep-bar-wrap{flex:1;height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;}
.ep-score{min-width:40px;font-weight:700;text-align:right;}
.ep-verdict{min-width:32px;font-size:10px;font-weight:600;color:var(--text-muted);text-align:right;}
"""

# ── GRADIO APP ─────────────────────────────────────────────────────────────────
DECISIONS = ["refund","replace","escalate","apologize","ignore","investigate"]

with gr.Blocks(title="CustomerSupportEnv — AI Decision Engine", css=CUSTOM_CSS) as demo:

    _init = demo_session()
    session_state = gr.State(demo_session)

    # Inject CSS directly — most reliable when Gradio is mounted on FastAPI
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")

    # ── Header ──
    gr.HTML("""<div class="main-header">
      <h1>🤖 CustomerSupportEnv — AI Decision Engine</h1>
      <p>Real-world reinforcement learning environment for intelligent customer support triage</p>
      <div class="header-chips">
        <span class="chip accent">OpenEnv Compliant</span>
        <span class="chip">3 Task Levels</span>
        <span class="chip">Dynamic Rewards</span>
        <span class="chip">Confidence Scoring</span>
        <span class="chip">Multi-step Episodes</span>
        <span class="chip">Meta × Scaler Hackathon 2026</span>
      </div>
    </div>""")

    # ── Main 3-column row ──
    with gr.Row():

        # LEFT COLUMN
        with gr.Column(scale=5, min_width=300):
            task_info_html = gr.HTML(f'<div class="task-badge">📊 <strong>Demo Preview</strong> — Click <strong>🔄 New Episode</strong> to start a live session</div>')
            gr.HTML('<div class="section-label">📨 Active Complaint</div>')
            complaint_html = gr.HTML('<div class="reasoning-empty">Click New Episode to load a real complaint from the environment.</div>')
            gr.HTML('<div class="section-label" style="margin-top:14px;">📋 Other Complaints in Queue</div>')
            queue_html = gr.HTML("")
            gr.HTML('<div class="section-label" style="margin-top:14px;">🎯 Your Decision</div>')
            decision_dd = gr.Dropdown(choices=DECISIONS, label="Action", value="investigate", interactive=True)
            with gr.Row():
                confidence_slider = gr.Slider(minimum=0.0,maximum=1.0,step=0.05,value=0.75,label="Confidence",interactive=True)
                urgency_cb = gr.Checkbox(label="⚠️ Flag as Urgent", value=False, interactive=True)
            reasoning_box = gr.Textbox(label="Your Reasoning", placeholder="Explain why you chose this action…", lines=2, interactive=True)
            with gr.Row():
                reset_btn  = gr.Button("🔄 New Episode",      elem_id="reset_btn",  variant="primary")
                submit_btn = gr.Button("🚀 Submit Decision",  elem_id="submit_btn", variant="secondary")
            status_html = gr.HTML("")

        # MIDDLE COLUMN
        with gr.Column(scale=5, min_width=280):
            gr.HTML('<div class="section-label">🧠 AI Reasoning Breakdown</div>')
            reasoning_html = gr.HTML('<div class="reasoning-empty">Make a decision to see reasoning…</div>')
            gr.HTML('<div class="section-label" style="margin-top:14px;">⏱️ Decision History</div>')
            history_html = gr.HTML(render_history_timeline(_init["history"]))

        # RIGHT COLUMN
        with gr.Column(scale=4, min_width=240):
            gr.HTML('<div class="section-label">📊 Performance Analytics</div>')
            analytics_html = gr.HTML(render_analytics_panel(_init))

    # ── Bottom charts row ──
    gr.HTML('<div class="bottom-section-divider"></div>')
    with gr.Row():
        with gr.Column(scale=1):
            reward_chart_html = gr.HTML(render_reward_chart(_init["reward_history"]))
        with gr.Column(scale=1):
            category_chart_html = gr.HTML(render_category_chart(_init["history"]))
        with gr.Column(scale=1):
            env_overview_html = gr.HTML(render_env_overview())

    # ── CALLBACKS ──────────────────────────────────────────────────────────────

    def do_reset(_session):
        data = api_reset()
        if "error" in data:
            return (
                _session,
                f'<div class="task-badge" style="color:var(--red);">❌ Server error: {data["error"]}</div>',
                '<div class="reasoning-empty">Server not reachable.</div>',
                "","","",render_analytics_panel(fresh_session()),"","",
                render_reward_chart([]), render_category_chart([])
            )
        obs = data.get("observation", {})
        session = fresh_session()
        session["complaints"]       = obs.get("complaints", [])
        session["episode_step"]     = obs.get("episode_step", 0)
        session["max_steps"]        = obs.get("max_steps", 5)
        session["budget"]           = obs.get("budget_remaining", 1000.0)
        session["satisfaction"]     = obs.get("satisfaction_score", 1.0)
        session["cumulative_reward"]= obs.get("cumulative_reward", 0.0)
        session["escalation_count"] = obs.get("escalation_count", 0)
        session["backlog_size"]     = obs.get("backlog_size", 0)
        session["task_name"]        = obs.get("task_name", "")
        session["task_level"]       = obs.get("metadata", {}).get("task_level", "")
        session["done"]             = obs.get("done", False)
        session["active_idx"]       = 0
        session["reward_history"]   = []

        complaints = session["complaints"]; active_c = complaints[0] if complaints else {}
        task_html = f"""<div class="task-badge">
  🎯 Task: <strong>{session['task_name']}</strong> |
  Level: <strong>{session['task_level'].upper()}</strong> |
  Steps: <strong>{session['max_steps']}</strong>
</div>
<div style="font-size:12px;color:var(--text-secondary);margin-top:4px;">{obs.get('task_description','')}</div>"""
        c_html = render_complaint_card(active_c, active=True) if active_c else ""
        q_html = "".join(render_complaint_card(c) for c in complaints[1:])
        return (
            session, task_html, c_html, q_html,
            '<div class="reasoning-empty">Make your first decision…</div>',
            '<div class="history-empty">No decisions yet.</div>',
            render_analytics_panel(session), "✅ New episode started", "",
            render_reward_chart([]), render_category_chart([])
        )

    def do_submit(decision, confidence, urgency, reasoning, _session):
        if _session.get("done"):
            return (
                _session,
                '<div class="episode-done">🏁 Episode complete! Start a new episode.</div>',
                "",
                render_reasoning_panel({}, "", 0.0),
                render_history_timeline(_session.get("history", [])),
                render_analytics_panel(_session),
                "Episode already done.", "",
                render_reward_chart(_session.get("reward_history", [])),
                render_category_chart(_session.get("history", [])),
            )
        complaints = _session.get("complaints", [])
        if not complaints:
            return (
                _session,
                '<div class="reasoning-empty">No active complaint.</div>',
                "",
                render_reasoning_panel({}, "", 0.0),
                render_history_timeline(_session.get("history", [])),
                render_analytics_panel(_session),
                "No complaint to evaluate.", "",
                render_reward_chart(_session.get("reward_history", [])),
                render_category_chart(_session.get("history", [])),
            )
        active_c = complaints[0]
        action_payload = {
            "complaint_id": active_c["complaint_id"],
            "decision": decision, "confidence": confidence,
            "reasoning": reasoning, "urgency_flag": urgency,
        }
        data = api_step(action_payload)
        if "error" in data:
            return (
                _session,
                f'<div style="color:var(--red);">❌ Error: {data["error"]}</div>',
                "",
                f'<div style="color:var(--red);">❌ API Error: {data["error"]}</div>',
                render_history_timeline(_session.get("history", [])),
                render_analytics_panel(_session),
                f"Server error: {data['error']}", "",
                render_reward_chart(_session.get("reward_history", [])),
                render_category_chart(_session.get("history", [])),
            )
        obs        = data.get("observation", {})
        step_reward= data.get("reward", 0.0)
        done       = data.get("done", False)

        _session["complaints"]        = obs.get("complaints", [])
        _session["episode_step"]      = obs.get("episode_step", 0)
        _session["cumulative_reward"] = obs.get("cumulative_reward", 0.0)
        _session["satisfaction"]      = obs.get("satisfaction_score", 1.0)
        _session["budget"]            = obs.get("budget_remaining", 1000.0)
        _session["escalation_count"]  = obs.get("escalation_count", 0)
        _session["backlog_size"]      = obs.get("backlog_size", 0)
        _session["done"]              = done
        _session["reward_history"]    = _session.get("reward_history", []) + [step_reward]

        feedback_list = obs.get("last_step_feedback", [])
        last_fb = feedback_list[0] if feedback_list else {}

        if last_fb:
            rb = last_fb.get("reward_breakdown", {})
            _session["history"].append({
                "step":           _session["episode_step"],
                "complaint_text": last_fb.get("complaint_text", ""),
                "decision":       decision,
                "verdict":        last_fb.get("verdict", ""),
                "correct":        last_fb.get("correct", False),
                "total_reward":   rb.get("total", 0.0),
                "category":       last_fb.get("category", "unknown"),
            })

        new_complaints = _session["complaints"]
        if done:
            c_html = '<div class="episode-done">🏁 Episode Complete! Click New Episode to play again.</div>'
            q_html = ""; status = f"Episode done. Final reward: {_session['cumulative_reward']:+.4f}"
        elif new_complaints:
            c_html = render_complaint_card(new_complaints[0], active=True)
            q_html = "".join(render_complaint_card(c) for c in new_complaints[1:])
            status = f"Step {_session['episode_step']} — Reward: {step_reward:+.4f}"
        else:
            c_html = '<div class="reasoning-empty">Waiting for next batch…</div>'
            q_html = ""; status = "Waiting for next complaint batch…"

        return (
            _session, c_html, q_html,
            render_reasoning_panel(last_fb, reasoning, confidence),
            render_history_timeline(_session["history"]),
            render_analytics_panel(_session),
            status, "",
            render_reward_chart(_session["reward_history"]),
            render_category_chart(_session["history"]),
        )

    # ── WIRE BUTTONS ───────────────────────────────────────────────────────────
    reset_btn.click(
        fn=do_reset, inputs=[session_state],
        outputs=[
            session_state, task_info_html, complaint_html, queue_html,
            reasoning_html, history_html, analytics_html, status_html, reasoning_box,
            reward_chart_html, category_chart_html,
        ],
    )
    submit_btn.click(
        fn=do_submit,
        inputs=[decision_dd, confidence_slider, urgency_cb, reasoning_box, session_state],
        outputs=[
            session_state, complaint_html, queue_html,
            reasoning_html, history_html, analytics_html, status_html, reasoning_box,
            reward_chart_html, category_chart_html,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7861, share=False,
        theme=gr.themes.Base(primary_hue="violet",secondary_hue="purple",neutral_hue="gray",font=gr.themes.GoogleFont("Inter")),
        css=CUSTOM_CSS,
    )