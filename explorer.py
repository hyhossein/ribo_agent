#!/usr/bin/env python3
"""
RIBO Evidence Explorer — localhost web app.

Reads prediction traces from results/runs/ and serves an interactive
visual interface showing [doc, page, sentence] attribution.

Usage:
    pip install flask
    python explorer.py
    # Open http://localhost:5001
"""
import glob
import json
import os
from pathlib import Path
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent

# ── Load predictions from the most recent multistep run ──────────

def _load_predictions():
    pattern = str(ROOT / "results" / "runs" / "*multistep*" / "predictions.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        # fallback: any run
        pattern = str(ROOT / "results" / "runs" / "*" / "predictions.jsonl")
        files = sorted(glob.glob(pattern))
    if not files:
        return []
    latest = files[-1]
    preds = []
    with open(latest) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


PREDICTIONS = _load_predictions()

# ── Load eval questions for stems/options ────────────────────────

def _load_eval():
    p = ROOT / "data" / "parsed" / "eval.jsonl"
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            q = json.loads(line)
            out[q["qid"]] = q
    return out

EVAL = _load_eval()

# ── API endpoints ────────────────────────────────────────────────

@app.route("/api/predictions")
def api_predictions():
    summary = []
    for p in PREDICTIONS:
        trace = p.get("extras", {}).get("trace", {})
        summary.append({
            "qid": p["qid"],
            "predicted": p["predicted"],
            "correct": p["correct"],
            "is_correct": p["is_correct"],
            "confidence": trace.get("confidence", 0),
            "n_steps": len(trace.get("steps", [])),
            "n_citations": len(trace.get("all_citations", [])),
            "stem": trace.get("question_stem", EVAL.get(p["qid"], {}).get("stem", "")),
            "options": trace.get("options", EVAL.get(p["qid"], {}).get("options", {})),
            "domain": EVAL.get(p["qid"], {}).get("content_domain", ""),
        })
    return jsonify(summary)


@app.route("/api/trace/<qid>")
def api_trace(qid):
    for p in PREDICTIONS:
        if p["qid"] == qid:
            trace = p.get("extras", {}).get("trace", {})
            if not trace:
                # build minimal trace from prediction
                trace = {
                    "qid": p["qid"],
                    "predicted": p["predicted"],
                    "correct": p["correct"],
                    "is_correct": p["is_correct"],
                    "confidence": 0,
                    "steps": [],
                    "all_citations": [],
                }
            # enrich with eval data
            eq = EVAL.get(qid, {})
            trace["question_stem"] = trace.get("question_stem", eq.get("stem", ""))
            trace["options"] = trace.get("options", eq.get("options", {}))
            trace["content_domain"] = eq.get("content_domain", "")
            return jsonify(trace)
    return jsonify({"error": "not found"}), 404


# ── Serve the frontend ───────────────────────────────────────────

@app.route("/")
def index():
    return FRONTEND_HTML


FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RIBO Evidence Explorer</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Lora:ital@0;1&display=swap" rel="stylesheet">
<style>
:root {
  --bg0: #0A0D14; --bg1: #12151C; --bg2: #1E2330;
  --tx0: #C5CDE0; --tx1: #8B96B0; --tx2: #4A5068; --tx3: #3A4158;
  --green: #10B981; --red: #EF4444; --blue: #3B82F6; --purple: #8B5CF6;
  --yellow: #F59E0B; --pink: #EC4899;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg0); color:var(--tx0); font-family:'DM Sans',system-ui,sans-serif; }
.mono { font-family:'JetBrains Mono',monospace; }
.serif { font-family:'Lora',Georgia,serif; }

/* Layout */
#header { padding:12px 16px; border-bottom:1px solid var(--bg2); display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; }
#header h1 { font-size:14px; font-weight:700; }
#header .sub { font-size:9px; color:var(--tx2); }
#main { display:grid; grid-template-columns:380px 1fr; height:calc(100vh - 50px); overflow:hidden; }
#left { border-right:1px solid var(--bg2); overflow-y:auto; padding:14px; }
#right { overflow-y:auto; padding:14px; display:none; }
#right.active { display:block; }

/* Questions list */
.q-card { padding:8px 10px; background:var(--bg1); border:1px solid var(--bg2); border-radius:5px; cursor:pointer; margin-bottom:5px; transition:all .12s; }
.q-card:hover, .q-card.sel { border-color:#3B82F644; background:#3B82F610; }
.q-label { font-size:8px; font-weight:700; color:var(--tx2); background:var(--bg2); padding:1px 4px; border-radius:2px; display:inline-block; margin-right:4px; }
.q-stem { font-size:11px; line-height:1.45; margin-top:3px; }

/* Options */
.opt { display:flex; align-items:center; gap:6px; padding:4px 8px; border-radius:4px; border:1px solid var(--bg2); margin-bottom:4px; }
.opt.correct { border-color:#10B98150; background:#10B98110; }
.opt.wrong { border-color:#EF444450; background:#EF444410; }
.opt-letter { width:18px; height:18px; border-radius:3px; display:flex; align-items:center; justify-content:center; font-size:9px; font-weight:700; background:var(--bg2); color:var(--tx2); }
.opt.correct .opt-letter { background:var(--green); color:#fff; }
.opt.pick .opt-letter { background:var(--blue); color:#fff; }

/* Steps */
.step { background:var(--bg1); border:1px solid var(--bg2); border-radius:6px; overflow:hidden; margin-bottom:5px; }
.step-hdr { padding:8px 12px; cursor:pointer; display:flex; align-items:center; gap:8px; }
.step-type { font-size:8px; font-weight:700; padding:1px 5px; border-radius:2px; text-transform:uppercase; letter-spacing:.06em; }
.step-body { padding:8px 12px; border-top:1px solid var(--bg2); display:none; }
.step.open .step-body { display:block; }
.step-output { font-size:10px; line-height:1.5; color:var(--tx1); white-space:pre-wrap; background:var(--bg0); padding:8px; border-radius:4px; }

/* Refs */
.ref { padding:8px 10px; background:var(--bg1); border:1px solid var(--bg2); border-radius:5px; margin-bottom:4px; }
.ref.cited { border-left:3px solid var(--green); }
.ref-tag { font-size:8px; color:var(--tx3); background:var(--bg0); display:inline-block; padding:1px 4px; border-radius:2px; margin-top:3px; }
.ref-text { font-size:11px; line-height:1.55; color:var(--tx1); }
.ref.cited .ref-text { color:var(--tx0); }

/* Doc tags */
.doc-tag { display:inline-flex; align-items:center; gap:3px; padding:1px 5px; border-radius:3px; font-size:9px; font-weight:600; white-space:nowrap; }
.doc-dot { width:4px; height:4px; border-radius:50%; }

/* Page strip */
.page-strip { position:relative; height:16px; background:var(--bg2); border-radius:3px; overflow:hidden; margin:3px 0; }
.page-mark { position:absolute; top:2px; bottom:2px; border-radius:2px; }

/* Sim bar */
.sim-bar { display:flex; align-items:center; gap:5px; min-width:80px; }
.sim-track { flex:1; height:3px; background:var(--bg2); border-radius:2px; overflow:hidden; }
.sim-fill { height:100%; border-radius:2px; transition:width .5s ease; }

/* Confidence ring */
.conf-ring { position:relative; width:48px; height:48px; }
.conf-ring svg { transform:rotate(-90deg); }
.conf-val { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; }

.section-label { font-size:9px; font-weight:600; color:var(--tx3); text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px; }
.btn { width:100%; padding:8px 14px; border-radius:5px; border:none; background:linear-gradient(135deg,#3B82F6,#8B5CF6); color:#fff; font-size:11px; font-weight:600; cursor:pointer; }
.btn:disabled { background:var(--bg2); cursor:wait; }
.result-box { background:var(--bg1); border:1px solid var(--bg2); border-radius:6px; padding:12px; margin-bottom:12px; }
</style>
</head>
<body>

<div id="header">
  <div>
    <h1>RIBO Evidence Explorer</h1>
    <div class="sub mono">Multi-Step Reasoning · [doc, page, sentence] Attribution · localhost:5001</div>
  </div>
  <div id="doc-tags"></div>
</div>

<div id="main">
  <div id="left">
    <div class="section-label">Questions (<span id="q-count">0</span>)</div>
    <div id="q-list"></div>
    <div id="detail" style="display:none">
      <div class="result-box">
        <div id="detail-stem" style="font-size:12px;font-weight:600;line-height:1.45;margin-bottom:10px"></div>
        <div id="detail-opts"></div>
        <div id="detail-result" style="display:none;margin-top:10px"></div>
      </div>
      <div id="steps-container" style="display:none">
        <div class="section-label">Pipeline Trace</div>
        <div id="steps-list"></div>
      </div>
    </div>
  </div>
  <div id="right">
    <div class="section-label">Document Map — Where Evidence Lives</div>
    <div id="doc-map" class="result-box"></div>
    <div class="section-label">All Evidence — Ranked by Relevance</div>
    <div id="refs-list"></div>
    <div id="source-dist" class="result-box" style="margin-top:12px"></div>
  </div>
</div>

<script>
const DOCS = {
  OAP_2025:               {title:"Ontario Automobile Policy",short:"OAP",color:"#E05A3A",pages:68},
  RIBO_By_Law_1:          {title:"RIBO By-Law No. 1",short:"ByLaw1",color:"#4F7BE8",pages:33},
  "RIBO_By-Law_1":        {title:"RIBO By-Law No. 1",short:"ByLaw1",color:"#4F7BE8",pages:33},
  RIBO_By_Law_2:          {title:"RIBO By-Law No. 2",short:"ByLaw2",color:"#8B5CF6",pages:18},
  "RIBO_By-Law_2":        {title:"RIBO By-Law No. 2",short:"ByLaw2",color:"#8B5CF6",pages:18},
  RIBO_By_Law_3:          {title:"RIBO By-Law No. 3",short:"ByLaw3",color:"#6366F1",pages:12},
  "RIBO_By-Law_3":        {title:"RIBO By-Law No. 3",short:"ByLaw3",color:"#6366F1",pages:12},
  RIB_Act_1990:           {title:"RIB Act 1990",short:"RIBA",color:"#10A37F",pages:42},
  Ontario_Regulation_989: {title:"Ont. Reg. 989",short:"Reg989",color:"#D97706",pages:16},
  Ontario_Regulation_990: {title:"Ont. Reg. 990",short:"Reg990",color:"#EA8C1A",pages:24},
  Ontario_Regulation_991: {title:"Ont. Reg. 991",short:"Reg991",color:"#F59E0B",pages:20},
  WIKI:                   {title:"LLM Wiki",short:"Wiki",color:"#EC4899",pages:1},
};

const STEP_COLORS = {
  DECOMPOSE:"#7C3AED", RETRIEVE:"#2563EB", RETRIEVE_RAW:"#2563EB",
  WIKI_CHECK:"#EC4899", REASON:"#D97706", VERIFY:"#059669"
};
const STEP_ICONS = {
  DECOMPOSE:"\u{1F50D}", RETRIEVE:"\u{1F4DA}", RETRIEVE_RAW:"\u{1F4DA}",
  WIKI_CHECK:"\u{1F9E0}", REASON:"\u26A1", VERIFY:"\u2713"
};

function docTag(id) {
  const d = DOCS[id] || {short:id.slice(0,8),color:"#666"};
  return `<span class="doc-tag mono" style="background:${d.color}18;border:1px solid ${d.color}40;color:${d.color}"><span class="doc-dot" style="background:${d.color}"></span>${d.short}</span>`;
}

function simBar(score, used) {
  const pct = Math.round(score*100);
  const c = !used?"#555":score>.9?"#10B981":score>.8?"#3B82F6":"#F59E0B";
  return `<div class="sim-bar"><div class="sim-track"><div class="sim-fill" style="width:${pct}%;background:${c}"></div></div><span class="mono" style="font-size:9px;color:${c};font-weight:600;min-width:28px;text-align:right">${pct}%</span></div>`;
}

let predictions = [];
let currentTrace = null;

async function init() {
  const resp = await fetch("/api/predictions");
  predictions = await resp.json();
  document.getElementById("q-count").textContent = predictions.length;

  const list = document.getElementById("q-list");
  list.innerHTML = predictions.map((p, i) => `
    <div class="q-card" onclick="selectQ(${i})" id="qc-${i}">
      <span class="q-label mono">${p.qid}</span>
      ${p.domain ? `<span class="q-label mono">${p.domain}</span>` : ""}
      <span class="q-label mono" style="color:${p.is_correct?'#10B981':'#EF4444'}">${p.is_correct?'✓':'✗'} ${p.predicted||'?'}/${p.correct}</span>
      <div class="q-stem">${p.stem || p.qid}</div>
    </div>
  `).join("");
}

async function selectQ(idx) {
  document.querySelectorAll(".q-card").forEach(c => c.classList.remove("sel"));
  document.getElementById("qc-"+idx).classList.add("sel");

  const p = predictions[idx];
  const resp = await fetch("/api/trace/" + p.qid);
  currentTrace = await resp.json();

  const detail = document.getElementById("detail");
  detail.style.display = "block";

  document.getElementById("detail-stem").textContent = currentTrace.question_stem || p.stem || p.qid;

  const opts = currentTrace.options || p.options || {};
  document.getElementById("detail-opts").innerHTML = Object.entries(opts).map(([k,v]) => {
    const isCorrect = k === (currentTrace.correct || p.correct);
    const isPick = k === (currentTrace.predicted || p.predicted);
    const cls = isCorrect ? "opt correct" : (isPick && !isCorrect ? "opt wrong" : "opt");
    const lcls = isCorrect ? "opt-letter" : (isPick ? "opt-letter" : "opt-letter");
    return `<div class="${cls}"><span class="${lcls} mono">${k}</span><span style="font-size:11px;color:var(--tx1)">${v}</span>${isCorrect?'<span style="margin-left:auto;font-size:9px;color:var(--green)">✓</span>':''}</div>`;
  }).join("");

  // Result
  const res = document.getElementById("detail-result");
  const conf = currentTrace.confidence || 0;
  const correct = currentTrace.is_correct;
  const allCitations = currentTrace.all_citations || [];
  const usedCount = allCitations.filter(c => c.used_in_answer).length;
  const confColor = conf>.9?"#10B981":conf>.8?"#3B82F6":"#F59E0B";

  res.style.display = "flex";
  res.style.alignItems = "center";
  res.style.justifyContent = "space-between";
  res.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px">
      <div class="conf-ring">
        <svg width="48" height="48"><circle cx="24" cy="24" r="21" fill="none" stroke="var(--bg2)" stroke-width="3"/>
        <circle cx="24" cy="24" r="21" fill="none" stroke="${confColor}" stroke-width="3"
          stroke-dasharray="${2*Math.PI*21}" stroke-dashoffset="${2*Math.PI*21*(1-conf)}" stroke-linecap="round"/></svg>
        <div class="conf-val mono" style="color:${confColor}">${Math.round(conf*100)}</div>
      </div>
      <div>
        <div class="mono" style="font-size:9px;color:var(--tx2)">Model answer</div>
        <div class="mono" style="font-size:22px;font-weight:800;color:${correct?'var(--green)':'var(--red)'}">
          ${currentTrace.predicted||'?'} <span style="font-size:10px;font-weight:500;opacity:.6">${correct?'CORRECT':'WRONG'}</span>
        </div>
      </div>
    </div>
    <div style="text-align:right">
      <div class="mono" style="font-size:9px;color:var(--tx2)">evidence</div>
      <div class="mono" style="font-size:18px;font-weight:700">${allCitations.length}</div>
      <div class="mono" style="font-size:9px;color:var(--green)">${usedCount} cited</div>
    </div>
  `;

  // Steps
  const steps = currentTrace.steps || [];
  const sc = document.getElementById("steps-container");
  sc.style.display = steps.length ? "block" : "none";

  document.getElementById("steps-list").innerHTML = steps.map((s, i) => {
    const type = s.step_type || "UNKNOWN";
    const col = STEP_COLORS[type] || "#666";
    const icon = STEP_ICONS[type] || "?";
    const refs = s.citations || [];
    const used = refs.filter(r => r.used_in_answer).length;
    const dur = Math.round(s.duration_ms || 0);
    const output = (s.output_text || "").replace(/</g,"&lt;").replace(/>/g,"&gt;");

    return `<div class="step" id="step-${i}" onclick="toggleStep(${i})">
      <div class="step-hdr" style="border-left:3px solid ${col}">
        <span style="font-size:12px">${icon}</span>
        <div style="flex:1">
          <span class="step-type mono" style="background:${col}20;color:${col}">${type.replace("_"," ")}</span>
          <span style="font-size:11px;font-weight:600;margin-left:6px">${s.label||type}</span>
          <div style="font-size:9px;color:var(--tx2);margin-top:1px">${s.description||""}</div>
        </div>
        <span class="mono" style="font-size:8px;color:var(--tx2)">${refs.length?used+"/"+refs.length+" cited":""}</span>
        <span class="mono" style="font-size:8px;color:var(--tx2);margin-left:6px">${dur}ms</span>
        <span style="color:var(--tx2);font-size:10px;margin-left:4px">\u25B8</span>
      </div>
      <div class="step-body">
        <pre class="step-output mono">${output}</pre>
        ${refs.length ? '<div style="margin-top:8px"><div class="section-label">Evidence ['+refs.length+' sentences]</div>' +
          refs.map((r,j) => refHTML(r,j)).join("") + '</div>' : ''}
      </div>
    </div>`;
  }).join("");

  // Right panel
  const right = document.getElementById("right");
  right.classList.add("active");

  // Doc map
  const byDoc = {};
  allCitations.forEach(c => {
    const did = c.doc_id;
    if (!byDoc[did]) byDoc[did] = [];
    byDoc[did].push(c);
  });

  document.getElementById("doc-map").innerHTML = Object.entries(byDoc).map(([did, refs]) => {
    const d = DOCS[did] || {short:did.slice(0,8),color:"#666",pages:50};
    const marks = refs.map(r => {
      const left = ((r.page-1)/d.pages)*100;
      return `<div class="page-mark" style="left:${left}%;width:${r.used_in_answer?8:5}px;background:${r.used_in_answer?d.color:d.color+'66'}"></div>`;
    }).join("");
    return `<div style="margin-bottom:8px">
      <div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">${docTag(did)}<span class="mono" style="font-size:8px;color:var(--tx2)">${d.pages}pp</span></div>
      <div class="page-strip">${marks}</div>
      <div style="display:flex;justify-content:space-between" class="mono" style="font-size:7px;color:var(--tx3)"><span>p.1</span><span>p.${d.pages}</span></div>
    </div>`;
  }).join("");

  // All refs sorted
  const sorted = [...allCitations].sort((a,b) => (b.similarity||0) - (a.similarity||0));
  document.getElementById("refs-list").innerHTML = sorted.map((r,i) => refHTML(r,i)).join("");

  // Source distribution
  const usedByDoc = {};
  allCitations.filter(c=>c.used_in_answer).forEach(c => {
    usedByDoc[c.doc_id] = (usedByDoc[c.doc_id]||0) + 1;
  });
  const distBars = Object.entries(usedByDoc).map(([did,n]) => {
    const d = DOCS[did]||{short:did,color:"#666"};
    return `<div style="flex:${n};background:${d.color};display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;color:#fff" class="mono">${d.short}</div>`;
  }).join("");

  document.getElementById("source-dist").innerHTML = `
    <div class="section-label">Source Distribution</div>
    <div style="display:flex;gap:2px;height:20px;border-radius:3px;overflow:hidden">${distBars}</div>
    <div class="mono" style="font-size:9px;color:var(--tx2);margin-top:4px">${usedCount} of ${allCitations.length} sentences cited · ${Object.keys(usedByDoc).length} source docs</div>
  `;
}

function refHTML(r, idx) {
  const d = DOCS[r.doc_id] || {short:r.doc_id,color:"#666"};
  const used = r.used_in_answer;
  return `<div class="ref ${used?'cited':''}" style="${used?'border-left-color:'+d.color:''}">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;flex-wrap:wrap;margin-bottom:4px">
      <div style="display:flex;align-items:center;gap:4px">
        <span class="mono" style="font-size:8px;font-weight:700;color:var(--tx2);background:var(--bg2);padding:1px 4px;border-radius:2px">#${idx+1}</span>
        ${docTag(r.doc_id)}
        <span class="mono" style="font-size:9px;color:var(--tx2)">p.${r.page}</span>
        <span class="mono" style="font-size:9px;color:var(--tx2)">s.${r.sentence_idx}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        ${used?'<span class="mono" style="font-size:8px;font-weight:700;color:var(--green);background:#10B98118;padding:1px 5px;border-radius:2px;text-transform:uppercase;letter-spacing:.05em">cited</span>':''}
        ${simBar(r.similarity||0, used)}
      </div>
    </div>
    <div class="mono" style="font-size:9px;font-weight:600;color:${d.color};margin-bottom:3px">${r.citation||''} ${r.section?'§'+r.section:''}</div>
    <div class="ref-text serif">${r.sentence_text||''}</div>
    <div class="ref-tag mono">[${r.doc_id}, p.${r.page}, s.${r.sentence_idx}]</div>
  </div>`;
}

function toggleStep(i) {
  document.getElementById("step-"+i).classList.toggle("open");
}

init();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\n  loaded {len(PREDICTIONS)} predictions from results/runs/")
    print(f"  loaded {len(EVAL)} eval questions")
    print(f"\n  Open http://localhost:5001\n")
    app.run(port=5001, debug=True)
