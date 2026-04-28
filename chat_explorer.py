import json,os,re,time,traceback
from pathlib import Path
from flask import Flask,jsonify,request as req
app=Flask(__name__)
ROOT=Path(__file__).resolve().parent

_llm=None;_wiki=None;_chunks=None;_eval=None
def get_llm():
    global _llm
    if not _llm:
        from ribo_agent.llm import make_client
        _llm=make_client({"backend":"anthropic","api_key":os.environ.get("ANTHROPIC_API_KEY",""),"model":"claude-opus-4-20250514"})
    return _llm
def get_wiki():
    global _wiki
    if _wiki is None:
        p=ROOT/"data"/"kb"/"wiki_compiled.md";_wiki=p.read_text() if p.exists() else ""
    return _wiki
def get_chunks():
    global _chunks
    if _chunks is None:
        p=ROOT/"data"/"kb"/"chunks.jsonl";_chunks=[json.loads(l) for l in open(p)] if p.exists() else []
    return _chunks
def get_eval():
    global _eval
    if _eval is None:
        p=ROOT/"data"/"parsed"/"eval.jsonl";_eval=[json.loads(l) for l in open(p)] if p.exists() else []
    return _eval

def llm_call(prompt,temperature=0.0,max_tokens=600):
    llm=get_llm()
    for a in range(3):
        try:return llm.complete(prompt,temperature=temperature,max_tokens=max_tokens)
        except KeyboardInterrupt:raise
        except:
            if a==2:raise
            time.sleep(5*(a+1))

def wiki_retrieve(query,top_k=3):
    w=get_wiki()
    if not w:return[]
    secs=re.split(r"\n(?=# )",w);qt=set(re.findall(r"\w{3,}",query.lower()));scored=[]
    for sec in secs:
        ls=sec.strip().split("\n");title=ls[0].strip("# ") if ls else "";body="\n".join(ls[1:])
        st=set(re.findall(r"\w{3,}",sec.lower()));tt=set(re.findall(r"\w{3,}",title.lower()))
        o=len(qt&st)+len(qt&tt)*3
        if o>0:scored.append({"title":title,"body":body[:500],"full":sec[:3000],"score":o})
    scored.sort(key=lambda x:x["score"],reverse=True)
    if not scored:return[]
    mx=scored[0]["score"]
    for s in scored:s["score"]=round(s["score"]/mx,3)
    return scored[:top_k]

def kb_retrieve(query,top_k=5):
    cs=get_chunks();qt=set(re.findall(r"\w{3,}",query.lower()));scored=[]
    for c in cs:
        ct=set(re.findall(r"\w{3,}",c.get("text","").lower()))
        ci=set(re.findall(r"\w{3,}",c.get("citation","").lower()))
        score=len(qt&ct)+len(qt&ci)*3
        scored.append((score,c))
    scored.sort(key=lambda x:x[0],reverse=True)
    mx=scored[0][0] if scored and scored[0][0]>0 else 1
    return[{"chunk":c,"score":round(s/mx,3)} for s,c in scored[:top_k] if s>0]

PROMPT_MCQ="""You are an Ontario RIBO Level 1 insurance exam expert.

STUDY MATERIALS:
{context}

---
Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

INSTRUCTIONS:
1. Think step by step
2. For each claim, cite the source: [SOURCE: document | section | "quote"]
3. Final answer: <answer>LETTER</answer>"""

PROMPT_OPEN="""You are an Ontario RIBO Level 1 insurance exam expert.

STUDY MATERIALS:
{context}

---
Question: {stem}

Give a clear answer. For each claim cite: [SOURCE: document | section | "quote"]"""

PROMPT_ZEROSHOT="""You are taking the Ontario RIBO Level 1 insurance broker licensing exam.
Answer this question. Think step by step, then: <answer>LETTER</answer>

{question}"""

def parse_q(text):
    opts={};om=re.findall(r"([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)",text,re.DOTALL)
    if om:
        for k,v in om:opts[k]=v.strip()
    stem=re.split(r"\n\s*A\.",text)[0].strip() if om else text.strip()
    return stem,opts

def run_agent(question,agent_type):
    trace={"agent":agent_type,"steps":[]};t0=time.time()
    stem,opts=parse_q(question);is_mcq=len(opts)==4

    if agent_type=="zeroshot":
        t1=time.time()
        resp=llm_call(PROMPT_ZEROSHOT.format(question=question))
        raw=resp.text
        if re.search(r"<answer>\s*[A-D]\s*$",raw,re.I):raw+="</answer>"
        m=re.search(r"<answer>\s*([A-D])\s*</answer>",raw,re.I)
        trace["steps"].append({"type":"REASON","label":"Zero-shot (Opus)","output":raw,
            "predicted":m.group(1).upper() if m else None,"duration_ms":round((time.time()-t1)*1000),
            "desc":"No retrieval — model answers from parametric knowledge only","citations":[],"wiki_sections":[],"kb_chunks":[]})
        trace["predicted"]=m.group(1).upper() if m else None
        trace["confidence"]=0.7;trace["answer_text"]=raw
        trace["total_ms"]=round((time.time()-t0)*1000)
        return trace

    # Wiki + KB retrieval
    ws=wiki_retrieve(stem);kb=kb_retrieve(stem)
    trace["steps"].append({"type":"WIKI_LOOKUP","label":"Wiki Knowledge",
        "sections":[{"title":s["title"],"body":s["body"],"score":s["score"]} for s in ws],
        "duration_ms":0,"desc":f"{len(ws)} wiki sections"})
    trace["steps"].append({"type":"KB_RETRIEVE","label":"Source Documents",
        "chunks":[{"chunk_id":r["chunk"]["chunk_id"],"source":r["chunk"]["source"],
            "citation":r["chunk"]["citation"],"section":r["chunk"].get("section",""),
            "text":r["chunk"]["text"][:400],"score":r["score"]} for r in kb],
        "duration_ms":0,"desc":f"{len(kb)} statutory chunks"})

    # Build context
    wiki_ctx="\n\n".join([s["full"] for s in ws]) if ws else get_wiki()[:15000]
    chunk_ctx="\n\n---\n\n".join(f"[{r['chunk']['citation']}]\n{r['chunk']['text']}" for r in kb)
    context=wiki_ctx[:12000]+"\n\n---\nSTATUTORY PASSAGES:\n"+chunk_ctx[:5000]

    if is_mcq:
        prompt=PROMPT_MCQ.format(context=context,stem=stem,a=opts.get("A",""),b=opts.get("B",""),c=opts.get("C",""),d=opts.get("D",""))
    else:
        prompt=PROMPT_OPEN.format(context=context,stem=stem)

    t1=time.time()
    resp=llm_call(prompt);raw=resp.text
    if re.search(r"<answer>\s*[A-D]\s*$",raw,re.I):raw+="</answer>"
    m=re.search(r"<answer>\s*([A-D])\s*</answer>",raw,re.I)
    predicted=m.group(1).upper() if m else None
    src_refs=re.findall(r'\[SOURCE:\s*([^|]+)\|\s*([^|]+)\|\s*"([^"]+)"\]',raw)
    citations=[{"doc":d.strip(),"section":s.strip(),"quote":q.strip()} for d,s,q in src_refs]

    trace["steps"].append({"type":"REASON","label":"LLM Reasoning (Opus)","output":raw,
        "predicted":predicted,"citations":citations,
        "wiki_sections":[{"title":s["title"],"score":s["score"]} for s in ws],
        "kb_chunks":[{"citation":r["chunk"]["citation"],"source":r["chunk"]["source"],
            "section":r["chunk"].get("section",""),"text":r["chunk"]["text"][:200],"score":r["score"]} for r in kb],
        "duration_ms":round((time.time()-t1)*1000),"desc":"Opus with wiki + source passages"})

    confidence=0.85
    if agent_type=="ensemble" and is_mcq and predicted:
        t2=time.time();votes={predicted:1};details=[{"answer":predicted,"source":"primary"}]
        for _ in range(2):
            try:
                r2=llm_call(prompt,temperature=0.7);txt=r2.text
                if re.search(r"<answer>\s*[A-D]\s*$",txt,re.I):txt+="</answer>"
                m2=re.search(r"<answer>\s*([A-D])\s*</answer>",txt,re.I)
                if m2:v=m2.group(1).upper();votes[v]=votes.get(v,0)+1;details.append({"answer":v,"source":f"vote"})
            except:pass
        winner=max(votes,key=votes.get);total=sum(votes.values())
        agreement=votes[winner]/total
        confidence=round(0.5+agreement*0.45+(0.05 if citations else 0),3)
        trace["steps"].append({"type":"VOTE","label":f"Majority Vote ({total}x)","votes":dict(votes),
            "details":details,"original":predicted,"winner":winner,"changed":winner!=predicted,
            "duration_ms":round((time.time()-t2)*1000),"desc":f"{agreement:.0%} agreement"})
        predicted=winner
    elif not is_mcq:
        confidence=round(0.7+(0.1 if citations else 0)+(0.1 if len(ws)>1 else 0),3)

    trace["predicted"]=predicted;trace["confidence"]=confidence
    trace["answer_text"]=raw;trace["total_ms"]=round((time.time()-t0)*1000)
    return trace

@app.route("/api/answer",methods=["POST"])
def api_answer():
    data=req.json
    try:
        tr=run_agent(data.get("question",""),data.get("agent","ensemble"))
        return jsonify(tr)
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/compare",methods=["POST"])
def api_compare():
    data=req.json;q=data.get("question","")
    agents=data.get("agents",["zeroshot","rewrite_wiki","ensemble"])
    results={}
    for ag in agents:
        try:results[ag]=run_agent(q,ag)
        except Exception as e:results[ag]={"error":str(e),"agent":ag}
    return jsonify(results)

@app.route("/api/eval")
def api_eval():
    return jsonify([{"qid":q["qid"],"stem":q["stem"],"options":q["options"],
        "correct":q["correct"],"domain":q.get("content_domain","")} for q in get_eval()])

@app.route("/")
def index():return PAGE

PAGE=r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RIBO Evidence Chat</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Lora:ital@0;1&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--b0:#0A0D14;--b1:#12151C;--b2:#1E2330;--b3:#262D3D;--t0:#C5CDE0;--t1:#8B96B0;--t2:#4A5068;
--g:#10B981;--r:#EF4444;--bl:#3B82F6;--p:#8B5CF6;--y:#F59E0B;--pk:#EC4899}
body{background:var(--b0);color:var(--t0);font-family:'DM Sans',system-ui,sans-serif;height:100vh;overflow:hidden}
.m{font-family:'JetBrains Mono',monospace}.sr{font-family:'Lora',Georgia,serif}

/* Three-column layout */
#app{display:grid;grid-template-columns:400px 1fr 1fr;height:100vh}

/* Left: Chat */
#left{display:flex;flex-direction:column;border-right:1px solid var(--b2);min-height:0}
#hdr{padding:10px 12px;border-bottom:1px solid var(--b2);flex-shrink:0}
#hdr h1{font-size:14px;font-weight:700}
#chat{flex:1;overflow-y:auto;padding:10px;min-height:0}
.msg{margin-bottom:8px}
.msg.u{display:flex;justify-content:flex-end}
.bub{max-width:92%;padding:10px;border-radius:10px;font-size:12px;line-height:1.5}
.msg.u .bub{background:var(--bl);color:#fff;border-bottom-right-radius:3px}
.msg.b .bub{background:var(--b1);border:1px solid var(--b2);border-bottom-left-radius:3px}

.ans-card{background:var(--b0);border:1px solid var(--b2);border-radius:8px;padding:10px;margin-top:6px}
.ans-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.ans-ltr{font-size:28px;font-weight:800;line-height:1}
.ans-ltr.ok{color:var(--g)}.ans-ltr.no{color:var(--r)}.ans-ltr.na{color:var(--bl)}
.conf{padding:2px 7px;border-radius:10px;font-size:9px;font-weight:600}
.reasoning{font-size:11px;line-height:1.55;color:var(--t1);margin-bottom:6px;font-family:'Lora',serif}

/* Evidence cards */
.ev{background:var(--b2);border-radius:5px;padding:6px 8px;margin-top:4px;font-size:10px;position:relative;cursor:default}
.ev:hover{background:var(--b3)}
.ev-path{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.ev-doc{padding:1px 5px;border-radius:3px;font-weight:600;font-size:8px}
.ev-arr{color:var(--t2);font-size:9px}
.ev-q{font-size:10px;color:var(--t1);margin-top:3px;font-family:'Lora',serif;font-style:italic}

/* Fed context cards (wiki + chunks that were INPUT to the model) */
.fed{background:var(--b0);border-radius:4px;padding:5px 7px;margin-top:3px;font-size:9px;border-left:2px solid var(--b3)}
.fed.wiki{border-left-color:var(--pk)}.fed.kb{border-left-color:var(--bl)}
.fed-title{font-weight:600;font-size:9px}
.fed-body{color:var(--t1);font-size:9px;margin-top:1px;font-family:'Lora',serif}

#inp{padding:8px 12px;border-top:1px solid var(--b2);flex-shrink:0}
#qi{width:100%;background:var(--b1);color:var(--t0);border:1px solid var(--b2);padding:8px;border-radius:6px;font-size:11px;resize:none;height:50px;font-family:inherit}
#qi:focus{outline:none;border-color:var(--bl)}
#ctrls{display:flex;gap:4px;align-items:center;margin-top:6px}
select{background:var(--b1);color:var(--t0);border:1px solid var(--b2);padding:4px 6px;border-radius:4px;font-size:9px}
.btn{padding:5px 12px;border-radius:5px;border:none;font-size:10px;font-weight:600;cursor:pointer}
.bp{background:linear-gradient(135deg,var(--bl),var(--p));color:#fff}.bp:disabled{background:var(--b2);cursor:wait}
.bs{padding:3px 6px;font-size:8px;background:var(--b2);color:var(--t1);border:none;border-radius:3px;cursor:pointer}
#eql{max-height:180px;overflow-y:auto;border:1px solid var(--b2);border-radius:4px;display:none;margin:4px 0;font-size:9px}
.eq{padding:4px 6px;cursor:pointer;border-bottom:1px solid var(--b2)}.eq:hover{background:var(--b2)}
.tg{font-size:7px;font-weight:700;color:var(--t2);background:var(--b2);padding:1px 3px;border-radius:2px;display:inline-block;margin-right:2px}

/* Middle: Trace */
#mid{overflow-y:auto;padding:12px;border-right:1px solid var(--b2)}
.sec{font-size:8px;font-weight:600;color:var(--t2);text-transform:uppercase;letter-spacing:.07em;margin:10px 0 4px}.sec:first-child{margin-top:0}
.step{background:var(--b1);border:1px solid var(--b2);border-radius:5px;overflow:hidden;margin-bottom:3px}
.sh{padding:6px 8px;cursor:pointer;display:flex;align-items:center;gap:5px;font-size:10px}
.sb2{padding:6px 8px;border-top:1px solid var(--b2);display:none}
.step.open .sb2{display:block}
.st2{font-size:7px;font-weight:700;padding:1px 4px;border-radius:2px;text-transform:uppercase;letter-spacing:.04em}
pre.out{font-size:9px;line-height:1.4;color:var(--t1);white-space:pre-wrap;background:var(--b0);padding:6px;border-radius:3px;margin:3px 0;max-height:300px;overflow-y:auto}
.ws{padding:4px 6px;background:var(--b0);border-radius:3px;margin-bottom:2px;border-left:2px solid var(--pk);font-size:9px}
.ck{padding:4px 6px;background:var(--b0);border-radius:3px;margin-bottom:2px;border-left:2px solid var(--bl);font-size:9px}
.cr{padding:4px 6px;background:var(--b0);border-radius:3px;margin-bottom:2px;border-left:2px solid var(--g);font-size:9px}
.vb{display:flex;gap:2px;height:24px;border-radius:3px;overflow:hidden;margin:4px 0}
.vs{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff}

/* Right: Compare */
#right{overflow-y:auto;padding:12px}
.cmp-card{background:var(--b1);border:1px solid var(--b2);border-radius:6px;padding:10px;margin-bottom:8px}
.cmp-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.cmp-name{font-weight:600;font-size:11px}
.cmp-ltr{font-size:24px;font-weight:800}
.cmp-reasoning{font-size:10px;color:var(--t1);line-height:1.5;font-family:'Lora',serif;margin-top:6px}
.cmp-fed{margin-top:6px;padding-top:6px;border-top:1px solid var(--b2)}
.empty{color:var(--t2);text-align:center;padding:40px 16px;font-size:11px;line-height:1.8}
.loader{display:inline-block;width:14px;height:14px;border:2px solid var(--b2);border-top-color:var(--bl);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<div id="app">
<!-- LEFT: Chat -->
<div id="left">
<div id="hdr"><h1>RIBO Evidence Chat</h1>
<div class="m" style="font-size:8px;color:var(--t2)">Opus · Wiki + KB + Voting · [doc · section · sentence]</div></div>
<div id="chat">
<div class="msg b"><div class="bub">Ask any RIBO question. Pick from 169 eval questions or type your own.<br><span class="m" style="font-size:9px;color:var(--t2)">Answer + reasoning + evidence trail + comparison across architectures.</span></div></div>
</div>
<div id="inp">
<div style="display:flex;gap:3px;margin-bottom:4px">
<button class="bs" onclick="toggleEval()">Browse 169 questions</button>
<select id="filter" onchange="renderEval()" style="font-size:8px"><option value="">All</option></select>
</div>
<div id="eql"></div>
<textarea id="qi" placeholder="Ask any Ontario insurance question..."></textarea>
<div id="ctrls">
<select id="ag"><option value="ensemble">Ensemble+Vote</option><option value="rewrite_wiki">Wiki</option><option value="zeroshot">Zero-shot</option></select>
<button class="btn bp" id="go" onclick="send()">Ask ▶</button>
<button class="btn bs" id="cmp-btn" onclick="compare()" title="Run through all 3 architectures">Compare All</button>
<span id="stat" class="m" style="font-size:8px;color:var(--t2)"></span>
</div></div></div>

<!-- MIDDLE: Trace -->
<div id="mid"><div class="empty">Pipeline trace appears here<br>when you ask a question</div></div>

<!-- RIGHT: Compare -->
<div id="right"><div class="empty">Click "Compare All" to run<br>the same question through<br>Zero-shot · Wiki · Ensemble<br>and see results side by side</div></div>
</div>

<script>
const DC={OAP_2025:{s:"OAP 1",c:"#E05A3A"},"RIBO_By-Law_1":{s:"By-Law 1",c:"#4F7BE8"},
RIBO_By_Law_1:{s:"By-Law 1",c:"#4F7BE8"},"RIBO_By-Law_2":{s:"By-Law 2",c:"#8B5CF6"},
RIBO_By_Law_2:{s:"By-Law 2",c:"#8B5CF6"},"RIBO_By-Law_3":{s:"By-Law 3",c:"#6366F1"},
RIBO_By_Law_3:{s:"By-Law 3",c:"#6366F1"},RIB_Act_1990:{s:"RIB Act",c:"#10A37F"},
Ontario_Regulation_989:{s:"Reg 989",c:"#D97706"},Ontario_Regulation_990:{s:"Reg 990",c:"#EA8C1A"},
Ontario_Regulation_991:{s:"Reg 991",c:"#F59E0B"}};
const SC={WIKI_LOOKUP:["#EC4899","🧠"],KB_RETRIEVE:["#2563EB","📚"],REASON:["#D97706","⚡"],VOTE:["#059669","🗳️"]};
const AGENT_NAMES={zeroshot:"Zero-shot (no retrieval)",rewrite_wiki:"Wiki + KB",ensemble:"Ensemble + Vote"};
let eqs=[],curAns=null,lastQ="";

async function init(){
  let r=await(await fetch("/api/eval")).json();eqs=r;
  let doms=[...new Set(r.map(q=>q.domain).filter(Boolean))];
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
    q.stem.slice(0,80)+'...</div>').join("");
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
  lastQ=q;let ag=document.getElementById("ag").value;
  let btn=document.getElementById("go");btn.disabled=true;
  document.getElementById("stat").innerHTML='<span class="loader"></span>';
  addMsg("u",esc(q.split("\n")[0].slice(0,100))+(q.length>100?"...":""));
  document.getElementById("qi").value="";
  let t0=Date.now();
  try{
    let r=await fetch("/api/answer",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question:q,agent:ag})});
    let d=await r.json();
    if(d.error){addMsg("b",'<span style="color:var(--r)">'+esc(d.error)+'</span>');return}
    let el=((Date.now()-t0)/1000).toFixed(1);
    document.getElementById("stat").textContent=el+"s";
    showAnswer(d,el,ag);
    showTrace(d);
    curAns=null;
  }catch(e){addMsg("b",'<span style="color:var(--r)">'+e.message+'</span>')}
  btn.disabled=false;
}

function showAnswer(tr,elapsed,ag){
  let ok=curAns&&tr.predicted?tr.predicted===curAns:null;
  let cls=ok===true?"ok":ok===false?"no":"na";
  let conf=tr.confidence||0;
  let confCol=conf>.9?"var(--g)":conf>.8?"var(--bl)":"var(--y)";
  let confBg=conf>.9?"#10B98120":conf>.8?"#3B82F620":"#F59E0B20";
  let raw=tr.answer_text||"";
  let clean=raw.replace(/\[SOURCE:[^\]]+\]/g,"").replace(/<answer>[^<]*<\/answer>/g,"").trim();

  let h='<div class="ans-card"><div class="ans-hdr">';
  if(tr.predicted) h+='<div><span class="ans-ltr '+cls+' m">'+tr.predicted+'</span>'+
    (ok===true?' <span style="color:var(--g);font-size:11px">✓ Correct</span>':'')+
    (ok===false?' <span style="color:var(--r);font-size:11px">✗ Wrong ('+curAns+')</span>':'')+'</div>';
  else h+='<div style="font-size:13px;font-weight:700;color:var(--bl)">Answer</div>';
  h+='<span class="conf m" style="background:'+confBg+';color:'+confCol+'">'+Math.round(conf*100)+'%</span></div>';
  h+='<div class="reasoning">'+esc(clean.slice(0,500))+'</div>';

  // Show what was FED to the model
  let reasonStep=(tr.steps||[]).find(s=>s.type==="REASON")||{};

  // LLM citations
  let cites=reasonStep.citations||[];
  if(cites.length){
    h+='<div class="m" style="font-size:7px;color:var(--t2);margin:4px 0 2px">CITED BY LLM:</div>';
    cites.forEach(c=>{
      let d2=docColor(c.doc);
      h+='<div class="ev"><div class="ev-path"><span class="ev-doc m" style="background:'+d2.c+'20;color:'+d2.c+';border:1px solid '+d2.c+'40">'+d2.s+'</span><span class="ev-arr">→</span><span class="m" style="font-size:9px">'+esc(c.section)+'</span></div><div class="ev-q">"'+esc(c.quote)+'"</div></div>';
    });
  }

  // Wiki sections fed
  let wikiSecs=reasonStep.wiki_sections||[];
  if(wikiSecs.length){
    h+='<div class="m" style="font-size:7px;color:var(--t2);margin:4px 0 2px">WIKI CONTEXT FED:</div>';
    wikiSecs.forEach(w=>{h+='<div class="fed wiki"><span class="fed-title" style="color:var(--pk)">🧠 '+esc(w.title)+'</span> <span class="m" style="font-size:7px;color:var(--t2)">'+Math.round(w.score*100)+'%</span></div>'});
  }

  // KB chunks fed
  let kbChunks=reasonStep.kb_chunks||[];
  if(kbChunks.length){
    h+='<div class="m" style="font-size:7px;color:var(--t2);margin:4px 0 2px">SOURCE DOCS FED:</div>';
    kbChunks.forEach(c=>{
      let d2=docColor(c.source);
      h+='<div class="fed kb"><span class="fed-title" style="color:'+d2.c+'">📚 '+esc(c.citation)+'</span> <span class="m" style="font-size:7px;color:var(--t2)">'+Math.round(c.score*100)+'%</span><div class="fed-body">'+esc(c.text.slice(0,100))+'...</div></div>'});
  }

  // Vote info
  let voteStep=(tr.steps||[]).find(s=>s.type==="VOTE");
  if(voteStep){
    let v=voteStep.votes||{};
    h+='<div class="m" style="font-size:7px;color:var(--t2);margin:4px 0 2px">VOTE:</div>';
    h+='<div style="display:flex;gap:2px;height:18px;border-radius:3px;overflow:hidden">';
    let cols={A:"#3B82F6",B:"#8B5CF6",C:"#10B981",D:"#F59E0B"};
    Object.entries(v).forEach(([k,n])=>{h+='<div class="vs m" style="flex:'+n+';background:'+(cols[k]||"#666")+';font-size:10px">'+k+':'+n+'</div>'});
    h+='</div>';
  }

  h+='</div><div class="m" style="font-size:8px;color:var(--t2);margin-top:4px">'+elapsed+'s · '+ag+'</div>';
  addMsg("b",h);
}

function showTrace(tr){
  let rp=document.getElementById("mid");
  let h='<div class="sec">Pipeline · '+(tr.total_ms/1000).toFixed(1)+'s</div>';
  (tr.steps||[]).forEach((s,i)=>{
    let[col,ic]=SC[s.type]||["#666","?"];
    h+='<div class="step" id="ts'+i+'" onclick="document.getElementById(\'ts'+i+'\').classList.toggle(\'open\')"><div class="sh" style="border-left:3px solid '+col+'"><span>'+ic+'</span><span class="st2 m" style="background:'+col+'20;color:'+col+'">'+s.type.replace(/_/g," ")+'</span><span style="font-weight:600;font-size:10px">'+esc(s.label||"")+'</span><span style="flex:1"></span><span class="m" style="font-size:7px;color:var(--t2)">'+(s.duration_ms||0>0?s.duration_ms+"ms":"<1ms")+'</span><span style="color:var(--t2)">▸</span></div><div class="sb2"><div style="font-size:9px;color:var(--t2);margin-bottom:4px">'+esc(s.desc||"")+'</div>';

    if(s.type==="WIKI_LOOKUP"&&s.sections){
      s.sections.forEach(w=>{h+='<div class="ws"><span style="font-weight:600;color:var(--pk);font-size:8px">'+esc(w.title)+'</span><div style="color:var(--t1);font-size:8px;margin-top:1px;font-family:Lora,serif">'+esc((w.body||"").slice(0,150))+'...</div><div class="m" style="font-size:7px;color:var(--t2)">'+Math.round(w.score*100)+'%</div></div>'});
    } else if(s.type==="KB_RETRIEVE"&&s.chunks){
      s.chunks.forEach(c=>{let d2=docColor(c.source);
        h+='<div class="ck"><span style="font-weight:600;color:'+d2.c+';font-size:8px">'+esc(c.citation)+'</span><div style="color:var(--t1);font-size:8px;margin-top:1px;font-family:Lora,serif">'+esc((c.text||"").slice(0,150))+'</div><div class="m" style="font-size:7px;color:var(--t2)">['+c.source+'] '+Math.round(c.score*100)+'%</div></div>'});
    } else if(s.type==="REASON"){
      h+='<pre class="out m">'+esc(s.output||"")+'</pre>';
      if(s.citations&&s.citations.length){h+='<div class="sec">LLM Citations</div>';
        s.citations.forEach(c=>{h+='<div class="cr"><span class="m" style="font-size:8px;font-weight:600;color:var(--g)">'+esc(c.doc)+' · '+esc(c.section)+'</span><div style="font-family:Lora,serif;font-size:9px;color:var(--t0);margin-top:1px">"'+esc(c.quote)+'"</div></div>'});
      }
    } else if(s.type==="VOTE"){
      let v=s.votes||{},cols={A:"#3B82F6",B:"#8B5CF6",C:"#10B981",D:"#F59E0B"};
      h+='<div class="vb">'+Object.entries(v).map(([k,n])=>'<div class="vs m" style="flex:'+n+';background:'+(cols[k]||"#666")+'">'+k+':'+n+'</div>').join("")+'</div>';
      h+='<div class="m" style="font-size:9px">'+(s.changed?'<span style="color:var(--y)">Changed: '+s.original+' → '+s.winner+'</span>':'<span style="color:var(--g)">Confirmed: '+s.winner+'</span>')+'</div>';
    }
    h+='</div></div>';
  });
  rp.innerHTML=h;
}

async function compare(){
  let q=document.getElementById("qi").value.trim()||lastQ;
  if(!q){alert("Enter a question first");return}
  lastQ=q;
  let rp=document.getElementById("right");
  rp.innerHTML='<div class="sec">Comparing 3 architectures...</div><div class="empty"><span class="loader"></span> Running zero-shot, wiki, and ensemble...</div>';
  document.getElementById("cmp-btn").disabled=true;
  try{
    let r=await fetch("/api/compare",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question:q,agents:["zeroshot","rewrite_wiki","ensemble"]})});
    let d=await r.json();
    let h='<div class="sec">Architecture Comparison</div>';
    ["zeroshot","rewrite_wiki","ensemble"].forEach(ag=>{
      let tr=d[ag]||{};
      let ok=curAns&&tr.predicted?tr.predicted===curAns:null;
      let cls=ok===true?"ok":ok===false?"no":"na";
      let conf=tr.confidence||0;
      let raw=tr.answer_text||"";
      let clean=raw.replace(/\[SOURCE:[^\]]+\]/g,"").replace(/<answer>[^<]*<\/answer>/g,"").trim();
      let reasonStep=(tr.steps||[]).find(s=>s.type==="REASON")||{};
      let cites=reasonStep.citations||[];

      h+='<div class="cmp-card"><div class="cmp-hdr"><div class="cmp-name">'+esc(AGENT_NAMES[ag]||ag)+'</div>';
      h+='<div style="display:flex;align-items:center;gap:6px">';
      if(tr.predicted) h+='<span class="cmp-ltr '+cls+' m">'+tr.predicted+'</span>';
      h+='<span class="conf m" style="background:'+(conf>.9?"#10B98120":conf>.8?"#3B82F620":"#F59E0B20")+';color:'+(conf>.9?"var(--g)":conf>.8?"var(--bl)":"var(--y)")+'">'+Math.round(conf*100)+'%</span>';
      h+='<span class="m" style="font-size:8px;color:var(--t2)">'+(tr.total_ms/1000).toFixed(1)+'s</span>';
      h+='</div></div>';
      h+='<div class="cmp-reasoning">'+esc(clean.slice(0,300))+'</div>';

      if(cites.length){
        h+='<div class="cmp-fed"><div class="m" style="font-size:7px;color:var(--t2);margin-bottom:2px">SOURCES:</div>';
        cites.forEach(c=>{let d2=docColor(c.doc);
          h+='<div class="ev" style="margin-top:2px"><div class="ev-path"><span class="ev-doc m" style="background:'+d2.c+'20;color:'+d2.c+'">'+d2.s+'</span><span class="ev-arr">→</span><span class="m" style="font-size:8px">'+esc(c.section)+'</span></div><div class="ev-q">"'+esc(c.quote.slice(0,100))+'"</div></div>'});
        h+='</div>';
      }

      if(ok===true) h+='<div style="margin-top:4px;font-size:9px;color:var(--g)">✓ Correct</div>';
      if(ok===false) h+='<div style="margin-top:4px;font-size:9px;color:var(--r)">✗ Wrong (correct: '+curAns+')</div>';
      h+='</div>';
    });
    rp.innerHTML=h;
  }catch(e){rp.innerHTML='<div class="empty" style="color:var(--r)">'+e.message+'</div>'}
  document.getElementById("cmp-btn").disabled=false;
}

function docColor(name){
  for(let k in DC)if((name||"").toLowerCase().includes(k.toLowerCase().replace(/_/g,""))||k.toLowerCase().includes((name||"").toLowerCase().replace(/[_ ]/g,"")))return DC[k];
  for(let k in DC)if((name||"").toLowerCase().includes(DC[k].s.toLowerCase().replace(/ /g,"")))return DC[k];
  return{s:(name||"?").slice(0,8),c:"#666"};
}
function esc(s){return(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
document.getElementById("qi").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}});
init();
</script></body></html>"""

if __name__=="__main__":
    print(f"  Wiki: {len(get_wiki())} chars | KB: {len(get_chunks())} chunks | Eval: {len(get_eval())} qs")
    print(f"  http://localhost:5001")
    app.run(port=5001,debug=True)
