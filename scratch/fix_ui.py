import sys

app_path = r"c:\Users\archi\practice_env\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update the CSS section completely
new_css = """PREMIUM_CSS = \"\"\"
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

/* ── TOKENS ── */
:root {
  --bg: #09090E;
  --bg2: #12121A;
  --bg3: #1B1B26;
  --bg4: #242433;
  --accent: #8A84FF;
  --accent-glow: rgba(138,132,255,0.4);
  --accent2: #CF6AF2;
  --green: #34C759;
  --red: #FF3B30;
  --yellow: #FFCC00;
  --orange: #FF9500;
  --blue: #0A84FF;
  --text: #F2F2F7;
  --text2: #AEAEB2;
  --text3: #636366;
  --border: rgba(255,255,255,0.08);
  --border2: rgba(255,255,255,0.15);
  --r: 16px;
  --r2: 12px;
  --r3: 8px;
  --shadow: 0 16px 40px rgba(0,0,0,0.5);
}

/* ── ANIMATIONS ── */
@keyframes glowPulse {
  0% { box-shadow: 0 0 5px rgba(138,132,255,0.2); }
  50% { box-shadow: 0 0 25px rgba(138,132,255,0.5); }
  100% { box-shadow: 0 0 5px rgba(138,132,255,0.2); }
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

@keyframes float {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-6px); box-shadow: 0 10px 25px rgba(138,132,255,0.3); }
  100% { transform: translateY(0px); }
}

@keyframes borderDance {
  0% { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}

/* ── GLOBAL ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container { 
    background: var(--bg) !important; 
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(138,132,255,0.05) 0%, transparent 60%),
        radial-gradient(circle at 85% 10%, rgba(207,106,242,0.05) 0%, transparent 60%) !important;
    color: var(--text) !important; 
    font-family: 'Inter', sans-serif !important; 
}
.gradio-container { padding: 0 !important; }
.contain { max-width: 1300px !important; margin: 0 auto !important; }

/* ── LIVE METRICS BAR ── */
.live-bar {
  display: flex; align-items: center; gap: 24px;
  background: rgba(8,8,22,0.85); backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px);
  border-bottom: 1px solid var(--border2); padding: 14px 28px;
  position: sticky; top: 0; z-index: 999;
  margin: -20px -20px 24px -20px;
  animation: fadeIn 1s ease-out;
  box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}
.lb-brand { display: flex; align-items: center; gap: 12px; margin-right: auto; }
.lb-logo { font-size: 26px; filter: drop-shadow(0 0 8px var(--accent)); }
.lb-title { font-family: 'Outfit', sans-serif; font-size: 17px; font-weight: 800; color: var(--text); letter-spacing: -0.5px; }
.lb-sub { font-size: 11px; color: var(--text2); margin-top: 1px; font-weight: 500; }
.lb-divider { width: 1px; height: 36px; background: var(--border); }
.lb-metric { display: flex; flex-direction: column; align-items: center; min-width: 75px; transition: transform 0.2s ease; }
.lb-metric:hover { transform: translateY(-3px); }
.lb-val { font-size: 17px; font-weight: 800; color: #fff; line-height: 1.2; font-family: 'Outfit', sans-serif;}
.lb-lab { font-size: 9px; color: var(--text3); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; font-weight: 700; }

/* ── MAIN HEADER ── */
.page-header {
  background: linear-gradient(-45deg, #0A0A1E, #14143A, #1C0A36, #0A0A1E);
  background-size: 400% 400%;
  animation: headerFlow 15s ease infinite;
  border: 1px solid rgba(138,132,255,0.3);
  border-radius: 20px; padding: 40px 48px; margin-bottom: 24px;
  position: relative; overflow: hidden;
  box-shadow: 0 20px 50px rgba(0,0,0,0.4);
}
.page-header::after {
  content: ''; position: absolute; bottom: 0; right: 0; width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(138,132,255,0.15) 0%, transparent 70%);
  filter: blur(40px);
}
.ph-title { font-family: 'Outfit', sans-serif; font-size: 34px; font-weight: 900; background: linear-gradient(135deg, #fff 20%, #BBAAFF 60%, #8A84FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 10px; letter-spacing: -1px; }
.ph-sub { color: var(--text2); font-size: 15px; margin: 0 0 24px; max-width: 600px; line-height: 1.5; }
.ph-chips { display: flex; gap: 10px; flex-wrap: wrap; }
.ph-chip { 
    background: rgba(255,255,255,0.04); 
    border: 1px solid var(--border); 
    border-radius: 20px; padding: 6px 16px; 
    font-size: 12px; color: var(--text2); font-weight: 600;
    transition: all 0.3s ease;
}
.ph-chip:hover { border-color: var(--accent); color: #fff; transform: scale(1.05); background: rgba(138,132,255,0.1); }
.ph-chip.ac { background: rgba(138,132,255,0.15); border-color: rgba(138,132,255,0.5); color: #C6C1FF; box-shadow: 0 0 15px rgba(138,132,255,0.15); }

/* ── FEATURE SPOTLIGHT (Smart Hover Board) ── */
.feature-spotlight {
    position: relative;
    background: rgba(18,18,26,0.7); 
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border2); 
    border-radius: 16px; padding: 18px 24px; 
    margin-bottom: 32px; display: flex; align-items: center; gap: 16px;
    animation: fadeIn 1s ease-out;
    box-shadow: var(--shadow);
    overflow: hidden;
    transition: all 0.4s ease;
}
.feature-spotlight::before {
    content: ''; position: absolute; top: 0; left: 0; height: 100%; width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
    border-radius: 4px 0 0 4px;
}
.feature-spotlight.fs-highlight-active {
    background: rgba(27,27,38,0.8);
    border-color: rgba(138,132,255,0.4);
    box-shadow: 0 12px 30px rgba(138,132,255,0.15);
    transform: translateY(-2px);
}
.fs-icon { font-size: 22px; filter: drop-shadow(0 0 6px var(--accent-glow)); }
.fs-text { font-size: 14px; font-weight: 500; color: var(--text2); transition: opacity 0.3s ease, transform 0.3s ease; line-height: 1.5; }
.fs-highlight { color: #fff; font-weight: 700; background: linear-gradient(90deg, #fff, var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

/* ── TASK BADGE ── */
.task-badge {
    display: inline-flex; align-items: center; gap: 10px;
    background: rgba(138,132,255,0.08);
    border: 1px solid rgba(138,132,255,0.2);
    padding: 10px 20px; border-radius: 30px;
    font-size: 13px; font-weight: 500; color: var(--text);
    margin-bottom: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.task-badge strong { color: #fff; font-weight: 700; }

/* ── TABS CUSTOMIZATION (Modern Pills) ── */
.tab-nav { 
    background: rgba(255,255,255,0.02) !important; 
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 32px !important;
    display: flex !important; gap: 8px !important; flex-wrap: wrap !important;
}
.tab-nav button { 
    flex: 1; min-width: 150px;
    font-size: 13px !important; 
    font-weight: 600 !important; 
    font-family: 'Outfit', sans-serif !important;
    color: var(--text2) !important;
    padding: 12px 20px !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
.tab-nav button:hover { 
    color: #fff !important; 
    background: rgba(255,255,255,0.05) !important;
    transform: translateY(-2px);
}
.tab-nav button.selected { 
    color: #fff !important; 
    background: rgba(138,132,255,0.15) !important;
    border: 1px solid rgba(138,132,255,0.3) !important;
    box-shadow: 0 4px 15px rgba(138,132,255,0.15) !important;
}

/* ── PANEL ── */
.panel {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--r); padding: 24px; height: 100%;
  animation: slideInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
  box-shadow: var(--shadow);
}
.panel-title {
  font-family: 'Outfit', sans-serif;
  font-size: 13px; font-weight: 800; letter-spacing: 1.5px;
  color: var(--text2); text-transform: uppercase; margin-bottom: 24px;
  display: flex; align-items: center; gap: 12px;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border2), transparent); }

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
    box-shadow: 0 0 25px rgba(138,132,255,0.2), var(--shadow); 
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
    background: linear-gradient(135deg, rgba(27,27,38,0.8), rgba(18,18,26,0.9)); 
    border: 1px solid rgba(138,132,255,0.25); border-radius: var(--r); 
    padding: 24px; margin-top: 24px; box-shadow: 0 12px 30px rgba(0,0,0,0.4); 
    position: relative; overflow: hidden;
}
.control-section::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent));
    background-size: 200% auto; animation: borderDance 4s linear infinite;
}
.gr-button { 
    font-family: 'Outfit', sans-serif !important; 
    font-weight: 700 !important; 
    border-radius: var(--r2) !important; 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; 
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 13px !important;
}
button[id*="reset_btn"] { 
    background: rgba(138,132,255,0.1) !important; 
    color: var(--accent) !important; 
    border: 1px solid var(--accent) !important;
}
button[id*="reset_btn"]:hover { 
    background: var(--accent) !important; 
    color: white !important; 
    box-shadow: 0 5px 20px rgba(138,132,255,0.4) !important;
}
button[id*="submit_btn"] { 
    background: linear-gradient(135deg, #3DDC84, #2EAA64) !important; 
    color: #050510 !important; 
    box-shadow: 0 8px 25px rgba(61,220,132,0.3) !important; 
    border: none !important;
}
button[id*="submit_btn"]:hover { 
    transform: translateY(-3px) scale(1.02) !important; 
    box-shadow: 0 12px 35px rgba(61,220,132,0.5) !important; 
    filter: brightness(1.1); 
}

/* ── REASONING PANEL ── */
.reasoning-panel { 
    background: var(--bg3); border: 1px solid var(--border); 
    border-radius: var(--r); padding: 24px;
    transition: all 0.5s ease;
    animation: fadeIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.tc-step { 
    transition: all 0.3s ease; 
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    padding: 12px !important;
    margin-bottom: 14px !important;
}
.tc-step:hover { background: rgba(255,255,255,0.05); transform: translateX(5px); }

/* ── GRADIO OVERRIDES & INPUT ANIMATIONS ── */
.gr-panel { background: transparent !important; border: none !important; }
footer { display: none !important; }
.gr-box { box-shadow: none !important; }

.score-ring { width: 90px; height: 90px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; animation: float 4s ease-in-out infinite; }

/* Interactive Inputs */
textarea, input[type="text"], input[type="number"], select { 
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-family: 'Inter', sans-serif !important;
}
textarea:focus, input[type="text"]:focus { 
    transform: scale(1.01);
    box-shadow: 0 0 15px rgba(138,132,255,0.2) !important; 
    border-color: var(--accent) !important;
}
\"\"\""""

if "PREMIUM_CSS = \"\"\"" in code:
    start_idx = code.find("PREMIUM_CSS = \"\"\"")
    end_idx = code.find("\"\"\"", start_idx + 20) + 3
    code = code[:start_idx] + new_css + code[end_idx:]
else:
    print("Could not find PREMIUM_CSS")

# 2. Update the Spotlight Script
old_spotlight = """<div id="spotlight_container" class="feature-spotlight">
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
</script>"""

new_spotlight = """<div id="spotlight_container" class="feature-spotlight">
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
        const container = document.getElementById('spotlight_container');
        
        tabs.forEach(tab => {
            const label = tab.innerText.trim().replace(/^[^a-zA-Z]+/, ''); // Remove icons
            tab.addEventListener('mouseenter', () => {
                if (descriptions[label]) {
                    container.classList.add('fs-highlight-active');
                    content.style.opacity = '0';
                    content.style.transform = 'translateY(8px)';
                    setTimeout(() => {
                        content.innerHTML = '<span class="fs-highlight">Module ' + label + ':</span> ' + descriptions[label];
                        content.style.opacity = '1';
                        content.style.transform = 'translateY(0)';
                    }, 200);
                }
            });
            tab.addEventListener('mouseleave', () => {
                container.classList.remove('fs-highlight-active');
                content.style.opacity = '0';
                content.style.transform = 'translateY(-8px)';
                setTimeout(() => {
                    content.innerHTML = '<span class="fs-highlight">Feature Spotlight:</span> Hover over any module tab to see detailed operations description.';
                    content.style.opacity = '1';
                    content.style.transform = 'translateY(0)';
                }, 200);
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
</script>"""

if old_spotlight in code:
    code = code.replace(old_spotlight, new_spotlight)
else:
    print("Could not find spotlight block.")
    
# 3. Add .task-badge html fix
if '<div class="task-badge">🎯 Task: <strong>Demo Preview</strong> — Click <strong>🔄 New Episode</strong> to start live mode</div>' in code:
    pass # Wait, that already has class task-badge ok, my CSS addresses it.

with open(app_path, "w", encoding="utf-8") as f:
    f.write(code)

print("done")
