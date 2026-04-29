"""RIBO Evidence Chat — Flask + Ollama (local, fast, no API keys).

Launch:
    source .venv/bin/activate
    python chat_explorer.py
    # Opens http://localhost:5001
"""
import json, re, time
from pathlib import Path
import httpx
from flask import Flask, jsonify, request as req

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent

# ── Data loaders (cached) ──
_chunks = None
_eval = None

def get_chunks():
    global _chunks
    if _chunks is None:
        p = ROOT / "data" / "kb" / "chunks.jsonl"
        _chunks = [json.loads(l) for l in open(p)] if p.exists() else []
    return _chunks

def get_eval():
    global _eval
    if _eval is None:
        p = ROOT / "data" / "parsed" / "eval.jsonl"
        _eval = [json.loads(l) for l in open(p)] if p.exists() else []
    return _eval

# ── Ollama direct call (no framework overhead) ──
OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct"

def llm_call(prompt, temperature=0.0, max_tokens=400):
    r = httpx.post(f"{OLLAMA}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 4096},
    }, timeout=120.0)
    r.raise_for_status()
    d = r.json()
    return {
        "text": d.get("response", ""),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "completion_tokens": d.get("eval_count", 0),
    }

# ── BM25-style keyword retrieval ──
def kb_retrieve(query, top_k=5):
    cs = get_chunks()
    qt = set(re.findall(r"\w{3,}", query.lower()))
    scored = []
    for c in cs:
        ct = set(re.findall(r"\w{3,}", c.get("text", "").lower()))
        ci = set(re.findall(r"\w{3,}", c.get("citation", "").lower()))
        score = len(qt & ct) + len(qt & ci) * 3
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    mx = scored[0][0] if scored else 1
    return [{"chunk": c, "score": round(s / mx, 3)} for s, c in scored[:top_k]]

# ── Prompts ──
PROMPT_MCQ = """You are an Ontario RIBO Level 1 insurance exam expert.

STUDY MATERIALS:
{context}

---
Question: {stem}

A. {a}
B. {b}
C. {c}
D. {d}

INSTRUCTIONS:
1. Think step by step.
2. For each key claim, cite the source: [SOURCE: document | section | "exact quote"]
3. Give your final answer: <answer>LETTER</answer>"""

PROMPT_OPEN = """You are an Ontario RIBO Level 1 insurance exam expert.

STUDY MATERIALS:
{context}

---
Question: {stem}

Give a clear answer. For each claim cite: [SOURCE: document | section | "exact quote"]"""

def parse_q(text):
    opts = {}
    om = re.findall(r"([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)", text, re.DOTALL)
    for k, v in om:
        opts[k] = v.strip()
    stem = re.split(r"\n\s*A\.", text)[0].strip() if om else text.strip()
    return stem, opts

# ── Source labels ──
SRC = {
    "OAP_2025": "OAP 1", "RIB_Act_1990": "RIB Act",
    "Ontario_Regulation_989": "Reg 989", "Ontario_Regulation_990": "Reg 990",
    "Ontario_Regulation_991": "Reg 991", "RIBO_By_Law_1": "By-Law 1",
    "RIBO_By_Law_2": "By-Law 2", "RIBO_By_Law_3": "By-Law 3",
}

def run_agent(question, agent_type="ensemble"):
    trace = {"agent": agent_type, "steps": []}
    t0 = time.time()
    stem, opts = parse_q(question)
    is_mcq = len(opts) == 4

    # Retrieve
    kb = kb_retrieve(stem)
    trace["steps"].append({
        "type": "KB_RETRIEVE", "label": "Source Documents",
        "chunks": [{
            "chunk_id": r["chunk"]["chunk_id"],
            "source": SRC.get(r["chunk"]["source"], r["chunk"]["source"]),
            "citation": r["chunk"]["citation"],
            "section": r["chunk"].get("section", ""),
            "text": r["chunk"]["text"][:400],
            "score": r["score"],
        } for r in kb],
        "duration_ms": 0,
        "desc": f"{len(kb)} statutory chunks",
    })

    # Build context
    context = "\n\n---\n\n".join(
        f"[{r['chunk']['citation']}]\n{r['chunk']['text']}" for r in kb
    )[:6000]

    if is_mcq:
        prompt = PROMPT_MCQ.format(
            context=context, stem=stem,
            a=opts.get("A", ""), b=opts.get("B", ""),
            c=opts.get("C", ""), d=opts.get("D", ""),
        )
    else:
        prompt = PROMPT_OPEN.format(context=context, stem=stem)

    # Primary LLM call
    t1 = time.time()
    resp = llm_call(prompt)
    raw = resp["text"]
    if re.search(r"<answer>\s*[A-D]\s*$", raw, re.I):
        raw += "</answer>"
    m = re.search(r"<answer>\s*([A-D])\s*</answer>", raw, re.I)
    predicted = m.group(1).upper() if m else None

    # Extract citations from response
    src_refs = re.findall(r'\[SOURCE:\s*([^|]+)\|\s*([^|]+)\|\s*"([^"]+)"\]', raw)
    citations = [{"doc": d.strip(), "section": s.strip(), "quote": q.strip()} for d, s, q in src_refs]

    trace["steps"].append({
        "type": "REASON", "label": "LLM Reasoning",
        "output": raw, "predicted": predicted, "citations": citations,
        "kb_chunks": [{
            "citation": r["chunk"]["citation"],
            "source": SRC.get(r["chunk"]["source"], r["chunk"]["source"]),
            "section": r["chunk"].get("section", ""),
            "text": r["chunk"]["text"][:200],
            "score": r["score"],
        } for r in kb],
        "duration_ms": round((time.time() - t1) * 1000),
        "desc": "Qwen 2.5 7B with source passages",
    })

    # Ensemble voting (only for MCQ + ensemble mode)
    confidence = 0.75
    if agent_type == "ensemble" and is_mcq and predicted:
        t2 = time.time()
        votes = {predicted: 1}
        details = [{"answer": predicted, "source": "primary"}]
        for _ in range(2):
            try:
                r2 = llm_call(prompt, temperature=0.7)
                txt = r2["text"]
                if re.search(r"<answer>\s*[A-D]\s*$", txt, re.I):
                    txt += "</answer>"
                m2 = re.search(r"<answer>\s*([A-D])\s*</answer>", txt, re.I)
                if m2:
                    v = m2.group(1).upper()
                    votes[v] = votes.get(v, 0) + 1
                    details.append({"answer": v, "source": "vote"})
            except Exception:
                pass
        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        agreement = votes[winner] / total
        confidence = round(0.5 + agreement * 0.45 + (0.05 if citations else 0), 3)
        trace["steps"].append({
            "type": "VOTE", "label": f"Majority Vote ({total}x)",
            "votes": dict(votes), "details": details,
            "original": predicted, "winner": winner,
            "changed": winner != predicted,
            "duration_ms": round((time.time() - t2) * 1000),
            "desc": f"{agreement:.0%} agreement",
        })
        predicted = winner
    elif not is_mcq:
        confidence = round(0.7 + (0.1 if citations else 0), 3)

    trace["predicted"] = predicted
    trace["confidence"] = confidence
    trace["answer_text"] = raw
    trace["total_ms"] = round((time.time() - t0) * 1000)
    return trace

# ── API routes ──
@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = req.json
    try:
        tr = run_agent(data.get("question", ""), data.get("agent", "ensemble"))
        return jsonify(tr)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/eval")
def api_eval():
    return jsonify([{
        "qid": q["qid"], "stem": q["stem"], "options": q["options"],
        "correct": q["correct"], "domain": q.get("content_domain") or "",
    } for q in get_eval()])

@app.route("/")
def index():
    return PAGE

# ── Full HTML/CSS/JS UI ──
PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RIBO Evidence Chat</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Lora:ital@0;1&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--b0:#0A0D14;--b1:#12151C;--b2:#1E2330;--b3:#262D3D;--t0:#C5CDE0;--t1:#8B96B0;--t2:#4A5068;
--g:#10B981;--r:#EF4444;--bl:#3B82F6;--p:#8B5CF6;--y:#F59E0B;--pk:#EC4899}
body{background:var(--b0);color:var(--t0);font-family:'DM Sans',system-ui,sans-serif;height:100vh;overflow:hidden}
.m{font-family:'JetBrains Mono',monospace}.sr{font-family:'Lora',Georgia,serif}
#app{display:grid;grid-template-columns:420px 1fr;height:100vh}

/* Left: Chat */
#left{display:flex;flex-direction:column;border-right:1px solid var(--b2);min-height:0}
#hdr{padding:12px 14px;border-bottom:1px solid var(--b2);flex-shrink:0}
#hdr h1{font-size:15px;font-weight:700}
#chat{flex:1;overflow-y:auto;padding:12px;min-height:0}
.msg{margin-bottom:10px}
.msg.u{display:flex;justify-content:flex-end}
.bub{max-width:92%;padding:10px 12px;border-radius:12px;font-size:12.5px;line-height:1.6}
.msg.u .bub{background:var(--bl);color:#fff;border-bottom-right-radius:3px}
.msg.b .bub{background:var(--b1);border:1px solid var(--b2);border-bottom-left-radius:3px}

.ans-card{background:var(--b0);border:1px solid var(--b2);border-radius:8px;padding:12px;margin-top:8px}
.ans-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.ans-ltr{font-size:32px;font-weight:800;line-height:1}
.ans-ltr.ok{color:var(--g)}.ans-ltr.no{color:var(--r)}.ans-ltr.na{color:var(--bl)}
.conf{padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.reasoning{font-size:11.5px;line-height:1.6;color:var(--t1);margin-bottom:8px;font-family:'Lora',serif}
.meta{font-size:9px;color:var(--t2);margin-top:6px}

.ev{background:var(--b2);border-radius:6px;padding:7px 9px;margin-top:5px;font-size:10px}
.ev:hover{background:var(--b3)}
.ev-path{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.ev-doc{padding:1px 6px;border-radius:3px;font-weight:600;font-size:9px}
.ev-arr{color:var(--t2);font-size:10px}
.ev-q{font-size:10px;color:var(--t1);margin-top:3px;font-family:'Lora',serif;font-style:italic}

.fed{background:var(--b0);border-radius:4px;padding:5px 7px;margin-top:3px;font-size:9px;border-left:2px solid var(--bl);cursor:pointer}
.fed:hover{background:var(--b2)}
.fed-title{font-weight:600;font-size:9px}
.fed-body{color:var(--t1);font-size:9px;margin-top:2px;font-family:'Lora',serif;display:none}
.fed.open .fed-body{display:block}

#inp{padding:10px 14px;border-top:1px solid var(--b2);flex-shrink:0}
#qi{width:100%;background:var(--b1);color:var(--t0);border:1px solid var(--b2);padding:9px;border-radius:8px;font-size:12px;resize:none;height:52px;font-family:inherit}
#qi:focus{outline:none;border-color:var(--bl)}
#ctrls{display:flex;gap:5px;align-items:center;margin-top:7px}
select{background:var(--b1);color:var(--t0);border:1px solid var(--b2);padding:5px 7px;border-radius:5px;font-size:10px}
.btn{padding:6px 14px;border-radius:6px;border:none;font-size:11px;font-weight:600;cursor:pointer}
.bp{background:linear-gradient(135deg,var(--bl),var(--p));color:#fff}.bp:disabled{opacity:.4;cursor:wait}
.bs{padding:4px 8px;font-size:9px;background:var(--b2);color:var(--t1);border:none;border-radius:4px;cursor:pointer}
.bs:hover{background:var(--b3)}
#eql{max-height:200px;overflow-y:auto;border:1px solid var(--b2);border-radius:5px;display:none;margin:5px 0;font-size:9.5px}
.eq{padding:5px 7px;cursor:pointer;border-bottom:1px solid var(--b2)}.eq:hover{background:var(--b2)}
.tg{font-size:7px;font-weight:700;color:var(--t2);background:var(--b2);padding:1px 4px;border-radius:2px;display:inline-block;margin-right:2px}

/* Right: Trace */
#right{overflow-y:auto;padding:14px}
.sec{font-size:9px;font-weight:600;color:var(--t2);text-transform:uppercase;letter-spacing:.07em;margin:12px 0 5px}.sec:first-child{margin-top:0}
.step{background:var(--b1);border:1px solid var(--b2);border-radius:6px;overflow:hidden;margin-bottom:4px}
.sh{padding:7px 10px;cursor:pointer;display:flex;align-items:center;gap:6px;font-size:11px}
.sh:hover{background:var(--b2)}
.sb2{padding:8px 10px;border-top:1px solid var(--b2);display:none}
.step.open .sb2{display:block}
.st2{font-size:8px;font-weight:700;padding:1px 5px;border-radius:3px;text-transform:uppercase;letter-spacing:.04em}
pre.out{font-size:10px;line-height:1.5;color:var(--t1);white-space:pre-wrap;background:var(--b0);padding:8px;border-radius:4px;margin:4px 0;max-height:350px;overflow-y:auto}
.ck{padding:5px 7px;background:var(--b0);border-radius:4px;margin-bottom:3px;border-left:2px solid var(--bl);font-size:10px}
.cr{padding:5px 7px;background:var(--b0);border-radius:4px;margin-bottom:3px;border-left:2px solid var(--g);font-size:10px}
.vb{display:flex;gap:2px;height:26px;border-radius:4px;overflow:hidden;margin:5px 0}
.vs{display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff}
.empty{color:var(--t2);text-align:center;padding:50px 20px;font-size:12px;line-height:2}
.loader{display:inline-block;width:16px;height:16px;border:2px solid var(--b2);border-top-color:var(--bl);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<div id="app">
<div id="left">
<div id="hdr"><h1>RIBO Evidence Chat</h1>
<div class="m" style="font-size:9px;color:var(--t2)">Qwen 2.5 7B · Local Ollama · [doc · section · sentence]</div></div>
<div id="chat">
<div class="msg b"><div class="bub">Ask any RIBO insurance question. Pick from 169 eval questions or type your own.<br><span class="m" style="font-size:9px;color:var(--t2)">Answer + reasoning + evidence trail + ensemble voting</span></div></div>
</div>
<div id="inp">
<div style="display:flex;gap:4px;margin-bottom:5px">
<button class="bs" onclick="toggleEval()">📋 Browse 169 questions</button>
<select id="filter" onchange="renderEval()" style="font-size:9px"><option value="">All</option></select>
</div>
<div id="eql"></div>
<textarea id="qi" placeholder="Ask any Ontario insurance question..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
<div id="ctrls">
<select id="ag"><option value="ensemble">Ensemble+Vote</option><option value="single">Single Pass</option></select>
<button class="btn bp" id="go" onclick="send()">Ask ▶</button>
<span id="stat" class="m" style="font-size:9px;color:var(--t2)"></span>
</div></div></div>

<div id="right"><div class="empty">🔍 Pipeline trace appears here<br>when you ask a question<br><br>Shows: retrieval → reasoning → voting<br>with full evidence trail</div></div>
</div>

<script>
const DC={
  OAP_2025:{s:"OAP 1",c:"#E05A3A"},"RIB Act":{s:"RIB Act",c:"#10A37F"},
  "Reg 989":{s:"Reg 989",c:"#D97706"},"Reg 990":{s:"Reg 990",c:"#EA8C1A"},
  "Reg 991":{s:"Reg 991",c:"#F59E0B"},"By-Law 1":{s:"By-Law 1",c:"#4F7BE8"},
  "By-Law 2":{s:"By-Law 2",c:"#8B5CF6"},"By-Law 3":{s:"By-Law 3",c:"#6366F1"},
};
const SC={KB_RETRIEVE:["#2563EB","📚"],REASON:["#D97706","⚡"],VOTE:["#059669","🗳️"]};
let eqs=[],curAns=null;

function esc(s){return s?String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"):""}
function docColor(name){
  for(let k in DC) if(name&&name.includes(k)) return DC[k];
  return {s:name||"?",c:"#6B7280"};
}

async function init(){
  let r=await(await fetch("/api/eval")).json();eqs=r;
  let doms=[...new Set(r.map(q=>q.domain).filter(Boolean))].sort();
  document.getElementById("filter").innerHTML='<option value="">All ('+r.length+')</option>'+doms.map(d=>'<option>'+d+'</option>').join("");
  renderEval();
}
function renderEval(){
  let f=document.getElementById("filter").value;
  let qs=f?eqs.filter(q=>q.domain===f):eqs;
  document.getElementById("eql").innerHTML=qs.map(q=>
    '<div class="eq" onclick="pick(\''+q.qid+'\')"><span class="tg m">'+q.qid+'</span>'+
    (q.domain?'<span class="tg m">'+q.domain+'</span>':'')+
    '<span class="tg m" style="color:var(--g)">'+q.correct+'</span> '+
    esc(q.stem.slice(0,80))+'...</div>').join("");
}
function toggleEval(){let e=document.getElementById("eql");e.style.display=e.style.display==="none"?"block":"none"}
function pick(qid){
  let q=eqs.find(x=>x.qid===qid);if(!q)return;
  document.getElementById("qi").value=q.stem+"\n\nA. "+q.options.A+"\nB. "+q.options.B+"\nC. "+q.options.C+"\nD. "+q.options.D;
  curAns=q.correct;document.getElementById("eql").style.display="none";
}

function addMsg(cls,html){
  let d=document.createElement("div");d.className="msg "+cls;
  d.innerHTML='<div class="bub">'+html+'</div>';
  let c=document.getElementById("chat");c.appendChild(d);c.scrollTop=c.scrollHeight;
}

async function send(){
  let q=document.getElementById("qi").value.trim();if(!q)return;
  let ag=document.getElementById("ag").value;
  let btn=document.getElementById("go");btn.disabled=true;
  document.getElementById("stat").innerHTML='<span class="loader"></span>';
  addMsg("u",esc(q.split("\n")[0].slice(0,120))+(q.length>120?"...":""));
  document.getElementById("qi").value="";
  let t0=Date.now();
  try{
    let r=await fetch("/api/answer",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question:q,agent:ag})});
    let d=await r.json();
    if(d.error){addMsg("b",'<span style="color:var(--r)">Error: '+esc(d.error)+'</span>');return}
    let el=((Date.now()-t0)/1000).toFixed(1);
    document.getElementById("stat").textContent=el+"s · "+d.agent;
    showAnswer(d,el);
    showTrace(d);
    curAns=null;
  }catch(e){addMsg("b",'<span style="color:var(--r)">'+esc(e.message)+'</span>')}
  btn.disabled=false;
}

function showAnswer(tr,elapsed){
  let ok=curAns&&tr.predicted?tr.predicted===curAns:null;
  let cls=ok===true?"ok":ok===false?"no":"na";
  let conf=tr.confidence||0;
  let confCol=conf>.85?"var(--g)":conf>.7?"var(--bl)":"var(--y)";
  let confBg=conf>.85?"#10B98120":conf>.7?"#3B82F620":"#F59E0B20";
  let raw=tr.answer_text||"";
  let clean=raw.replace(/\[SOURCE:[^\]]+\]/g,"").replace(/<answer>[^<]*<\/answer>/g,"").trim();

  let h='<div class="ans-card"><div class="ans-hdr">';
  if(tr.predicted) h+='<div><span class="ans-ltr '+cls+' m">'+tr.predicted+'</span>'+
    (ok===true?' <span style="color:var(--g);font-size:12px">✓ Correct</span>':'')+
    (ok===false?' <span style="color:var(--r);font-size:12px">✗ Wrong ('+curAns+')</span>':'')+'</div>';
  else h+='<div style="font-size:14px;font-weight:700;color:var(--bl)">Answer</div>';
  h+='<span class="conf m" style="background:'+confBg+';color:'+confCol+'">'+Math.round(conf*100)+'%</span></div>';
  h+='<div class="reasoning">'+esc(clean.slice(0,600))+'</div>';

  // LLM citations
  let step=(tr.steps||[]).find(s=>s.type==="REASON")||{};
  let cites=step.citations||[];
  if(cites.length){
    h+='<div class="m" style="font-size:8px;color:var(--t2);margin:6px 0 3px">EVIDENCE TRAIL:</div>';
    cites.forEach(c=>{
      let d2=docColor(c.doc);
      h+='<div class="ev"><div class="ev-path">'+
        '<span class="ev-doc m" style="background:'+d2.c+'20;color:'+d2.c+';border:1px solid '+d2.c+'40">'+esc(d2.s)+'</span>'+
        '<span class="ev-arr">→</span><span class="m" style="font-size:10px">'+esc(c.section)+'</span></div>'+
        '<div class="ev-q">"'+esc(c.quote)+'"</div></div>';
    });
  }

  // KB chunks fed to model
  let kbChunks=step.kb_chunks||[];
  if(kbChunks.length){
    h+='<div class="m" style="font-size:8px;color:var(--t2);margin:6px 0 3px">SOURCE DOCUMENTS FED:</div>';
    kbChunks.forEach(k=>{
      let d2=docColor(k.source);
      h+='<div class="fed" onclick="this.classList.toggle(\'open\')">'+
        '<span class="fed-title"><span style="color:'+d2.c+'">📄 '+esc(k.source)+'</span>'+
        (k.section?' → §'+esc(k.section):'')+
        ' <span class="m" style="font-size:7px;color:var(--t2)">'+Math.round(k.score*100)+'%</span></span>'+
        '<div class="fed-body">'+esc(k.text)+'</div></div>';
    });
  }

  h+='<div class="meta">'+elapsed+'s · '+tr.agent+(tr.total_ms?' · '+tr.total_ms+'ms total':'')+'</div>';
  h+='</div>';
  addMsg("b",h);
}

function showTrace(tr){
  let mid=document.getElementById("right");
  let h='<div class="sec">Pipeline Trace</div>';
  (tr.steps||[]).forEach((s,i)=>{
    let sc=SC[s.type]||["#6B7280","⚙️"];
    h+='<div class="step'+(i===tr.steps.length-1?' open':'')+'">'+
      '<div class="sh" onclick="this.parentElement.classList.toggle(\'open\')">'+
      '<span class="st2 m" style="background:'+sc[0]+'20;color:'+sc[0]+'">'+sc[1]+' '+s.type+'</span>'+
      '<span style="flex:1;font-size:10px">'+esc(s.label)+'</span>'+
      '<span class="m" style="font-size:8px;color:var(--t2)">'+(s.duration_ms?s.duration_ms+'ms':'')+'</span></div>';
    h+='<div class="sb2">';
    if(s.desc) h+='<div style="font-size:9px;color:var(--t2);margin-bottom:4px">'+esc(s.desc)+'</div>';

    if(s.type==="KB_RETRIEVE"&&s.chunks){
      s.chunks.forEach(c=>{
        let d2=docColor(c.source);
        h+='<div class="ck"><span style="color:'+d2.c+';font-weight:600">'+esc(c.source)+'</span>'+
          (c.section?' → §'+esc(c.section):'')+
          ' <span class="m" style="font-size:7px;color:var(--t2)">'+Math.round(c.score*100)+'%</span>'+
          '<div style="font-size:9px;color:var(--t1);margin-top:2px;font-family:Lora,serif">'+esc(c.text.slice(0,200))+'...</div></div>';
      });
    }
    if(s.type==="REASON"){
      if(s.output) h+='<pre class="out">'+esc(s.output)+'</pre>';
      if(s.citations&&s.citations.length){
        h+='<div style="font-size:8px;color:var(--t2);margin:4px 0 2px">Cited:</div>';
        s.citations.forEach(c=>{
          h+='<div class="cr">'+esc(c.doc)+' → '+esc(c.section)+' → "'+esc(c.quote)+'"</div>';
        });
      }
    }
    if(s.type==="VOTE"&&s.votes){
      let total=Object.values(s.votes).reduce((a,b)=>a+b,0);
      h+='<div class="vb">';
      let colors={A:"#3B82F6",B:"#10B981",C:"#F59E0B",D:"#EC4899"};
      for(let k in s.votes){
        let pct=Math.round(s.votes[k]/total*100);
        h+='<div class="vs m" style="width:'+pct+'%;background:'+(colors[k]||"#6B7280")+'">'+k+' ('+s.votes[k]+')</div>';
      }
      h+='</div>';
      if(s.changed) h+='<div style="font-size:9px;color:var(--y);margin-top:3px">⚠ Vote changed answer from '+s.original+' to '+s.winner+'</div>';
    }
    h+='</div></div>';
  });
  mid.innerHTML=h;
}

init();
</script></body></html>"""

if __name__ == "__main__":
    print("\n  RIBO Evidence Chat → http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
