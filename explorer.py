#!/usr/bin/env python3
"""RIBO Evidence Explorer — all runs, switchable."""
import glob, json, os
from pathlib import Path
from flask import Flask, jsonify, request as req

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent

def _load_all_runs():
    runs = {}
    for d in sorted(glob.glob(str(ROOT / "results" / "runs" / "*"))):
        name = os.path.basename(d)
        pfile = os.path.join(d, "predictions.jsonl")
        mfile = os.path.join(d, "metrics.json")
        if not os.path.exists(pfile): continue
        preds = [json.loads(l) for l in open(pfile) if l.strip()]
        metrics = json.load(open(mfile)) if os.path.exists(mfile) else {}
        correct = sum(1 for p in preds if p.get("is_correct"))
        has_trace = any("trace" in p.get("extras",{}) for p in preds)
        runs[name] = {
            "preds": preds,
            "metrics": metrics,
            "n": len(preds),
            "correct": correct,
            "accuracy": round(correct/len(preds),4) if preds else 0,
            "has_trace": has_trace,
        }
    return runs

def _load_eval():
    p = ROOT / "data" / "parsed" / "eval.jsonl"
    if not p.exists(): return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}

RUNS = _load_all_runs()
EVAL = _load_eval()

@app.route("/api/runs")
def api_runs():
    return jsonify([{"name":k,"n":v["n"],"correct":v["correct"],"accuracy":v["accuracy"],"has_trace":v["has_trace"]} for k,v in RUNS.items()])

@app.route("/api/predictions/<run_name>")
def api_predictions(run_name):
    r = RUNS.get(run_name)
    if not r: return jsonify([])
    out = []
    for p in r["preds"]:
        eq = EVAL.get(p["qid"],{})
        trace = p.get("extras",{}).get("trace",{})
        out.append({
            "qid":p["qid"],"predicted":p["predicted"],"correct":p["correct"],
            "is_correct":p["is_correct"],
            "confidence":trace.get("confidence",0),
            "n_citations":len(trace.get("all_citations",[])),
            "stem":trace.get("question_stem",eq.get("stem","")),
            "options":trace.get("options",eq.get("options",{})),
            "domain":eq.get("content_domain",""),
            "raw_response":p.get("raw_response","")[:500],
        })
    return jsonify(out)

@app.route("/api/trace/<run_name>/<qid>")
def api_trace(run_name, qid):
    r = RUNS.get(run_name)
    if not r: return jsonify({"error":"run not found"}),404
    for p in r["preds"]:
        if p["qid"]==qid:
            trace = p.get("extras",{}).get("trace",{})
            eq = EVAL.get(qid,{})
            trace["question_stem"] = trace.get("question_stem",eq.get("stem",""))
            trace["options"] = trace.get("options",eq.get("options",{}))
            trace["predicted"] = trace.get("predicted",p.get("predicted"))
            trace["correct"] = trace.get("correct",p.get("correct"))
            trace["is_correct"] = trace.get("is_correct",p.get("is_correct"))
            trace["raw_response"] = p.get("raw_response","")
            return jsonify(trace)
    return jsonify({"error":"not found"}),404

@app.route("/")
def index(): return HTML

HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RIBO Evidence Explorer</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Lora:ital@0;1&display=swap" rel="stylesheet">
<style>
:root{--b0:#0A0D14;--b1:#12151C;--b2:#1E2330;--t0:#C5CDE0;--t1:#8B96B0;--t2:#4A5068;--g:#10B981;--r:#EF4444;--bl:#3B82F6;--p:#8B5CF6;--y:#F59E0B;--pk:#EC4899}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--b0);color:var(--t0);font-family:'DM Sans',system-ui,sans-serif;font-size:13px}
.m{font-family:'JetBrains Mono',monospace}
#hdr{padding:10px 16px;border-bottom:1px solid var(--b2);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
#hdr h1{font-size:14px;font-weight:700}
#hdr .sub{font-size:9px;color:var(--t2)}
select{background:var(--b1);color:var(--t0);border:1px solid var(--b2);padding:4px 8px;border-radius:4px;font-size:11px;font-family:'JetBrains Mono',monospace}
#main{display:grid;grid-template-columns:360px 1fr;height:calc(100vh - 48px);overflow:hidden}
#left{border-right:1px solid var(--b2);overflow-y:auto;padding:12px}
#right{overflow-y:auto;padding:12px}
.qc{padding:7px 9px;background:var(--b1);border:1px solid var(--b2);border-radius:5px;cursor:pointer;margin-bottom:4px}
.qc:hover,.qc.sel{border-color:#3B82F644;background:#3B82F610}
.tag{font-size:8px;font-weight:700;color:var(--t2);background:var(--b2);padding:1px 4px;border-radius:2px;display:inline-block;margin-right:3px}
.stem{font-size:11px;line-height:1.4;margin-top:2px}
.opt{display:flex;align-items:center;gap:6px;padding:3px 7px;border-radius:4px;border:1px solid var(--b2);margin-bottom:3px;font-size:11px}
.opt.ok{border-color:#10B98150;background:#10B98110}
.opt.bad{border-color:#EF444450;background:#EF444410}
.ol{width:18px;height:18px;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;background:var(--b2);color:var(--t2);flex-shrink:0}
.opt.ok .ol{background:var(--g);color:#fff}
.opt.pick .ol{background:var(--bl);color:#fff}
.box{background:var(--b1);border:1px solid var(--b2);border-radius:6px;padding:10px;margin-bottom:10px}
.sec{font-size:9px;font-weight:600;color:var(--t2);text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px}
.step{background:var(--b1);border:1px solid var(--b2);border-radius:5px;overflow:hidden;margin-bottom:4px}
.step-h{padding:6px 10px;cursor:pointer;display:flex;align-items:center;gap:6px}
.step-b{padding:6px 10px;border-top:1px solid var(--b2);display:none}
.step.open .step-b{display:block}
.stype{font-size:8px;font-weight:700;padding:1px 4px;border-radius:2px;text-transform:uppercase;letter-spacing:.05em}
pre.out{font-size:10px;line-height:1.45;color:var(--t1);white-space:pre-wrap;background:var(--b0);padding:6px;border-radius:4px;margin:0}
.ref{padding:6px 8px;background:var(--b1);border:1px solid var(--b2);border-radius:4px;margin-bottom:3px;font-size:11px}
.ref.cited{border-left:3px solid var(--g)}
.dt{display:inline-flex;align-items:center;gap:2px;padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600;white-space:nowrap}
.dd{width:4px;height:4px;border-radius:50%}
.ps{position:relative;height:14px;background:var(--b2);border-radius:3px;overflow:hidden;margin:2px 0}
.pm{position:absolute;top:2px;bottom:2px;border-radius:2px}
.sb{display:flex;align-items:center;gap:4px;min-width:70px}
.st{flex:1;height:3px;background:var(--b2);border-radius:2px;overflow:hidden}
.sf{height:100%;border-radius:2px}
.rtag{font-size:8px;color:var(--t2);background:var(--b0);display:inline-block;padding:1px 3px;border-radius:2px;margin-top:2px}
.acc-bar{height:6px;border-radius:3px;margin-top:3px}
</style></head><body>
<div id="hdr">
<div><h1>RIBO Evidence Explorer</h1><div class="sub m">[doc, page, sentence] Attribution</div></div>
<div style="display:flex;align-items:center;gap:8px">
<label class="m" style="font-size:9px;color:var(--t2)">RUN:</label>
<select id="run-sel" onchange="loadRun(this.value)"></select>
<span id="run-acc" class="m" style="font-size:11px;font-weight:700"></span>
</div></div>
<div id="main"><div id="left">
<div class="sec">Questions (<span id="qn">0</span>)</div>
<div id="qlist"></div>
<div id="det" style="display:none"></div>
</div><div id="right"><div id="rp"></div></div></div>
<script>
const DC={"OAP_2025":{s:"OAP",c:"#E05A3A",p:68},"RIB_Act_1990":{s:"RIBA",c:"#10A37F",p:42},"RIBO_By-Law_1":{s:"ByLaw1",c:"#4F7BE8",p:33},"RIBO_By-Law_2":{s:"ByLaw2",c:"#8B5CF6",p:18},"RIBO_By-Law_3":{s:"ByLaw3",c:"#6366F1",p:12},"Ontario_Regulation_989":{s:"Reg989",c:"#D97706",p:16},"Ontario_Regulation_990":{s:"Reg990",c:"#EA8C1A",p:24},"Ontario_Regulation_991":{s:"Reg991",c:"#F59E0B",p:20},"WIKI":{s:"Wiki",c:"#EC4899",p:1}};
const SC={"DECOMPOSE":["#7C3AED","\u{1F50D}"],"RETRIEVE":["#2563EB","\u{1F4DA}"],"RETRIEVE_RAW":["#2563EB","\u{1F4DA}"],"WIKI_CHECK":["#EC4899","\u{1F9E0}"],"REASON":["#D97706","\u26A1"],"VERIFY":["#059669","\u2713"]};
function dt(id){let d=DC[id]||{s:id.slice(0,6),c:"#666"};return`<span class="dt m" style="background:${d.c}18;border:1px solid ${d.c}40;color:${d.c}"><span class="dd" style="background:${d.c}"></span>${d.s}</span>`}
function sb(v,u){let p=Math.round(v*100),c=!u?"#555":v>.9?"#10B981":v>.8?"#3B82F6":"#F59E0B";return`<div class="sb"><div class="st"><div class="sf" style="width:${p}%;background:${c}"></div></div><span class="m" style="font-size:8px;color:${c};font-weight:600">${p}%</span></div>`}
let runs=[],curRun="",curPreds=[];
async function init(){
  let r=await(await fetch("/api/runs")).json();
  runs=r;
  let sel=document.getElementById("run-sel");
  sel.innerHTML=r.map(x=>`<option value="${x.name}">${x.name} (${x.accuracy*100|0}% · ${x.n}q${x.has_trace?" · traced":""})</option>`).join("");
  if(r.length)loadRun(r[r.length-1].name);
}
async function loadRun(name){
  curRun=name;
  let r=runs.find(x=>x.name===name);
  document.getElementById("run-acc").textContent=r?`${(r.accuracy*100).toFixed(1)}% (${r.correct}/${r.n})`:"";
  document.getElementById("run-acc").style.color=r&&r.accuracy>.75?"#10B981":"#EF4444";
  let preds=await(await fetch("/api/predictions/"+name)).json();
  curPreds=preds;
  document.getElementById("qn").textContent=preds.length;
  document.getElementById("qlist").innerHTML=preds.map((p,i)=>`<div class="qc" id="qc${i}" onclick="selQ(${i})">
    <span class="tag m">${p.qid}</span>${p.domain?`<span class="tag m">${p.domain}</span>`:""}
    <span class="tag m" style="color:${p.is_correct?"var(--g)":"var(--r)"}">${p.is_correct?"\u2713":"\u2717"} ${p.predicted||"?"}/${p.correct}</span>
    <div class="stem">${p.stem||p.qid}</div></div>`).join("");
  document.getElementById("det").style.display="none";
  document.getElementById("rp").innerHTML="<div style='color:var(--t2);padding:40px;text-align:center'>Select a question to view evidence</div>";
}
async function selQ(i){
  document.querySelectorAll(".qc").forEach(c=>c.classList.remove("sel"));
  document.getElementById("qc"+i).classList.add("sel");
  let p=curPreds[i];
  let t=await(await fetch(`/api/trace/${curRun}/${p.qid}`)).json();
  let opts=t.options||p.options||{};
  let det=document.getElementById("det");
  det.style.display="block";
  let oh=Object.entries(opts).map(([k,v])=>{
    let ok=k===(t.correct||p.correct),pk=k===(t.predicted||p.predicted),bad=pk&&!ok;
    return`<div class="opt ${ok?"ok":""}${bad?" bad":""}${pk&&!bad?" pick":""}"><span class="ol m">${k}</span>${v}${ok?'<span style="margin-left:auto;font-size:9px;color:var(--g)">\u2713</span>':""}</div>`;
  }).join("");
  let conf=t.confidence||0;
  let cc=conf>.9?"var(--g)":conf>.8?"var(--bl)":"var(--y)";
  let steps=t.steps||[];
  let allC=t.all_citations||[];
  let used=allC.filter(c=>c.used_in_answer).length;
  let stepsH=steps.map((s,j)=>{
    let tp=s.step_type||"?",col=(SC[tp]||["#666"])[0],ic=(SC[tp]||["","?"])[1];
    let refs=s.citations||[];let ru=refs.filter(r=>r.used_in_answer).length;
    let out=(s.output_text||"").replace(/</g,"&lt;");
    let refsH=refs.length?refs.slice(0,10).map((r,ri)=>{
      let d=DC[r.doc_id]||{s:r.doc_id,c:"#666"};
      return`<div class="ref ${r.used_in_answer?"cited":""}" style="${r.used_in_answer?"border-left-color:"+d.c:""}">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:3px;margin-bottom:3px">
          <div style="display:flex;align-items:center;gap:3px"><span class="tag m">#${ri+1}</span>${dt(r.doc_id)}<span class="m" style="font-size:8px;color:var(--t2)">p.${r.page} s.${r.sentence_idx}</span></div>
          <div style="display:flex;align-items:center;gap:4px">${r.used_in_answer?'<span class="m" style="font-size:7px;font-weight:700;color:var(--g);background:#10B98118;padding:1px 4px;border-radius:2px">CITED</span>':""}${sb(r.similarity||0,r.used_in_answer)}</div>
        </div>
        <div class="m" style="font-size:8px;color:${d.c};margin-bottom:2px">${r.citation||""}</div>
        <div style="font-family:Lora,Georgia,serif;color:${r.used_in_answer?"var(--t0)":"var(--t1)"}">${r.sentence_text||""}</div>
        <div class="rtag m">[${r.doc_id}, p.${r.page}, s.${r.sentence_idx}]</div></div>`;
    }).join(""):"";
    return`<div class="step" id="st${j}" onclick="document.getElementById('st${j}').classList.toggle('open')">
      <div class="step-h" style="border-left:3px solid ${col}">
        <span>${ic}</span><div style="flex:1"><span class="stype m" style="background:${col}20;color:${col}">${tp.replace("_"," ")}</span>
        <span style="font-size:11px;font-weight:600;margin-left:4px">${s.label||tp}</span>
        <div style="font-size:9px;color:var(--t2)">${s.description||""}</div></div>
        <span class="m" style="font-size:8px;color:var(--t2)">${refs.length?ru+"/"+refs.length:""}</span>
        <span class="m" style="font-size:8px;color:var(--t2);margin-left:4px">${Math.round(s.duration_ms||0)}ms</span>
        <span style="color:var(--t2);font-size:9px;margin-left:3px">\u25B8</span>
      </div><div class="step-b"><pre class="out m">${out}</pre>${refsH?"<div style='margin-top:6px'><div class='sec'>Evidence</div>"+refsH+"</div>":""}</div></div>`;
  }).join("");
  // Raw response for non-trace runs
  let rawH="";
  if(!steps.length && t.raw_response){
    rawH=`<div class="box"><div class="sec">Raw Response</div><pre class="out m">${(t.raw_response||"").replace(/</g,"&lt;")}</pre></div>`;
  }
  det.innerHTML=`<div class="box">
    <div style="font-size:12px;font-weight:600;line-height:1.4;margin-bottom:8px">${t.question_stem||p.stem||""}</div>
    ${oh}
    <div style="display:flex;align-items:center;justify-content:space-between;margin-top:10px">
      <div style="display:flex;align-items:center;gap:8px">
        <div class="m" style="font-size:20px;font-weight:800;color:${t.is_correct?"var(--g)":"var(--r)"}">${t.predicted||"?"}</div>
        <div><div class="m" style="font-size:8px;color:var(--t2)">confidence</div><div class="m" style="font-size:14px;font-weight:700;color:${cc}">${conf?Math.round(conf*100)+"%":"—"}</div></div>
      </div>
      <div style="text-align:right" class="m"><div style="font-size:8px;color:var(--t2)">evidence</div>
        <div style="font-size:16px;font-weight:700">${allC.length||"—"}</div>
        <div style="font-size:8px;color:var(--g)">${used} cited</div></div>
    </div></div>
    ${steps.length?`<div class="sec">Pipeline Trace — ${steps.length} Steps</div>${stepsH}`:""}${rawH}`;
  // Right panel
  let rp=document.getElementById("rp");
  if(allC.length){
    let byDoc={};allC.forEach(c=>{if(!byDoc[c.doc_id])byDoc[c.doc_id]=[];byDoc[c.doc_id].push(c)});
    let mapH=Object.entries(byDoc).map(([did,refs])=>{
      let d=DC[did]||{s:did,c:"#666",p:50};
      let marks=refs.map(r=>`<div class="pm" style="left:${((r.page-1)/d.p)*100}%;width:${r.used_in_answer?7:4}px;background:${r.used_in_answer?d.c:d.c+"66"}"></div>`).join("");
      return`<div style="margin-bottom:6px"><div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">${dt(did)}<span class="m" style="font-size:7px;color:var(--t2)">${d.p}pp</span></div><div class="ps">${marks}</div></div>`;
    }).join("");
    let sorted=[...allC].sort((a,b)=>(b.similarity||0)-(a.similarity||0));
    let refsH=sorted.map((r,i)=>{
      let d=DC[r.doc_id]||{s:r.doc_id,c:"#666"};
      return`<div class="ref ${r.used_in_answer?"cited":""}" style="${r.used_in_answer?"border-left-color:"+d.c:""}">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:3px;margin-bottom:2px">
          <div style="display:flex;align-items:center;gap:3px"><span class="tag m">#${i+1}</span>${dt(r.doc_id)}<span class="m" style="font-size:8px;color:var(--t2)">p.${r.page} s.${r.sentence_idx}</span></div>
          <div style="display:flex;align-items:center;gap:4px">${r.used_in_answer?'<span class="m" style="font-size:7px;font-weight:700;color:var(--g);background:#10B98118;padding:1px 4px;border-radius:2px">CITED</span>':""}${sb(r.similarity||0,r.used_in_answer)}</div>
        </div>
        <div class="m" style="font-size:8px;color:${d.c};margin-bottom:2px">${r.citation||""}</div>
        <div style="font-family:Lora,Georgia,serif;color:${r.used_in_answer?"var(--t0)":"var(--t1)"}">${r.sentence_text||""}</div>
        <div class="rtag m">[${r.doc_id}, p.${r.page}, s.${r.sentence_idx}]</div></div>`;
    }).join("");
    let usedByDoc={};allC.filter(c=>c.used_in_answer).forEach(c=>{usedByDoc[c.doc_id]=(usedByDoc[c.doc_id]||0)+1});
    let dist=Object.entries(usedByDoc).map(([did,n])=>{let d=DC[did]||{s:did,c:"#666"};return`<div style="flex:${n};background:${d.c};display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;color:#fff" class="m">${d.s}</div>`}).join("");
    rp.innerHTML=`<div class="sec">Document Map</div><div class="box">${mapH}</div>
      <div class="sec">All Evidence — ${allC.length} sentences</div>${refsH}
      <div class="box" style="margin-top:8px"><div class="sec">Source Distribution</div>
        <div style="display:flex;gap:2px;height:18px;border-radius:3px;overflow:hidden">${dist||'<div style="color:var(--t2);font-size:9px">no cited sources</div>'}</div>
        <div class="m" style="font-size:8px;color:var(--t2);margin-top:3px">${used} of ${allC.length} cited · ${Object.keys(usedByDoc).length} docs</div></div>`;
  } else {
    rp.innerHTML=`<div style="color:var(--t2);padding:40px;text-align:center">No trace data for this run.<br>Run with multistep agent to get [doc,page,sentence] attribution.</div>`;
  }
}
init();
</script></body></html>"""

if __name__=="__main__":
    print(f"\n  {len(RUNS)} runs loaded:")
    for k,v in RUNS.items():
        print(f"    {k}: {v['correct']}/{v['n']} ({v['accuracy']:.1%}) trace={v['has_trace']}")
    print(f"  {len(EVAL)} eval questions")
    print(f"\n  Open http://localhost:5001\n")
    app.run(port=5001,debug=True)
