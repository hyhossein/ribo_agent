"""Full 169-question eval with checkpoint resume."""
import json,time,os,sys
from pathlib import Path
from ribo_agent.parsers.schema import MCQ
from ribo_agent.llm import make_client

ROOT=Path(__file__).resolve().parent
EVAL=ROOT/"data"/"parsed"/"eval.jsonl"
CHECKPOINT=ROOT/"results"/"runs"/"full_multistep_checkpoint.jsonl"
WIKI=ROOT/"data"/"kb"/"wiki_compiled.md"

# Load eval
mcqs=[MCQ(**json.loads(l)) for l in open(EVAL)]
print(f"Eval: {len(mcqs)} questions")

# Load checkpoint
done={}
if CHECKPOINT.exists():
    for l in open(CHECKPOINT):
        p=json.loads(l);done[p["qid"]]=p
    print(f"Checkpoint: {len(done)} already done")

# Setup
llm=make_client({"backend":"anthropic","api_key":os.environ["ANTHROPIC_API_KEY"],"model":"claude-opus-4-20250514"})
wiki=open(WIKI).read() if WIKI.exists() else ""
chunks=[json.loads(l) for l in open(ROOT/"data"/"kb"/"chunks.jsonl")]

import re
def bm25(query,top_k=5):
    qt=set(re.findall(r"\w{3,}",query.lower()));scored=[]
    for c in chunks:
        ct=set(re.findall(r"\w{3,}",c.get("text","").lower()))
        ci=set(re.findall(r"\w{3,}",c.get("citation","").lower()))
        scored.append((len(qt&ct)+len(qt&ci)*3,c))
    scored.sort(key=lambda x:x[0],reverse=True)
    return[c for _,c in scored[:top_k]]

PROMPT="""You are an Ontario RIBO Level 1 insurance exam expert.

STUDY MATERIALS:
{context}

---
Question: {stem}

Options:
A. {a}
B. {b}
C. {c}
D. {d}

Think step by step. Cite sources with [SOURCE: doc | section | "quote"]. Final: <answer>LETTER</answer>"""

correct=0;total=0
for i,mcq in enumerate(mcqs):
    if mcq.qid in done:
        if done[mcq.qid].get("is_correct"):correct+=1
        total+=1;continue
    # Retrieve
    kb=bm25(mcq.stem)
    chunk_ctx="\n\n---\n\n".join(f"[{c['citation']}]\n{c['text']}" for c in kb[:5])
    context=wiki[:15000]+"\n\n---\nPASSAGES:\n"+chunk_ctx[:5000]
    prompt=PROMPT.format(context=context,stem=mcq.stem,
        a=mcq.options["A"],b=mcq.options["B"],c=mcq.options["C"],d=mcq.options["D"])
    # Call with retry
    for attempt in range(5):
        try:
            resp=llm.complete(prompt,temperature=0.0,max_tokens=512)
            break
        except KeyboardInterrupt:raise
        except Exception as e:
            if attempt==4:resp=type('R',(),{"text":""})()
            else:print(f"\n  retry {attempt+1}: {e}");time.sleep(10*(attempt+1))
    raw=resp.text
    if re.search(r"<answer>\s*[A-D]\s*$",raw,re.I):raw+="</answer>"
    m=re.search(r"<answer>\s*([A-D])\s*</answer>",raw,re.I)
    pred=m.group(1).upper() if m else None
    is_ok=pred==mcq.correct
    if is_ok:correct+=1
    total+=1
    # Save checkpoint
    result={"qid":mcq.qid,"predicted":pred,"correct":mcq.correct,"is_correct":is_ok,"raw":raw[:500]}
    with open(CHECKPOINT,"a") as f:f.write(json.dumps(result)+"\n")
    acc=correct/total
    print(f"\r[{total:>3}/{len(mcqs)}] acc={acc:.3f} {'✓' if is_ok else '✗'} {mcq.qid} pred={pred} correct={mcq.correct}  ",end="",flush=True)

print(f"\n\nFINAL: {correct}/{total} = {correct/total:.4f}")
