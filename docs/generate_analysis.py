#!/usr/bin/env python3
"""Generate supplementary analysis artifacts.

Run: python docs/generate_analysis.py
Output:
  docs/confusion_matrix.png
  docs/per_domain_accuracy.png  
  docs/REPRODUCE.md
"""
import json, os, sys
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Find best run (v6 majority vote)
RUNS = ROOT / "results" / "runs"


def _find_run(pattern):
    if not RUNS.exists():
        return None
    for d in sorted(RUNS.iterdir(), reverse=True):
        if pattern in d.name and (d / "predictions.jsonl").exists():
            return d
    return None


def _load_preds(run_dir):
    return [json.loads(l) for l in (run_dir / "predictions.jsonl").open()]


def _load_eval():
    p = ROOT / "data" / "parsed" / "eval.jsonl"
    if not p.exists():
        return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in p.open()}


def confusion_matrix():
    """Generate confusion matrix heatmap for the best run."""
    run = _find_run("v6_3way") or _find_run("v4_confidence") or _find_run("v2_rewrite")
    if not run:
        print("No run found for confusion matrix")
        return
    
    preds = _load_preds(run)
    labels = ["A", "B", "C", "D"]
    matrix = np.zeros((4, 4), dtype=int)
    
    for p in preds:
        gt = p["correct"]
        pr = p.get("predicted")
        if gt in labels and pr in labels:
            matrix[labels.index(gt)][labels.index(pr)] += 1
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap='Blues', aspect='equal')
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix — {os.path.basename(run)[:40]}', fontsize=11, fontweight='bold')
    
    for i in range(4):
        for j in range(4):
            color = 'white' if matrix[i][j] > matrix.max() * 0.6 else 'black'
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out = DOCS / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def per_domain():
    """Generate per-domain accuracy bar chart."""
    run = _find_run("v6_3way") or _find_run("v4_confidence") or _find_run("v2_rewrite")
    if not run:
        print("No run found for per-domain")
        return
    
    preds = {p["qid"]: p for p in _load_preds(run)}
    eval_qs = _load_eval()
    
    if not eval_qs:
        print("No eval.jsonl found")
        return
    
    # Group by content_domain and source
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for qid, pred in preds.items():
        q = eval_qs.get(qid, {})
        domain = q.get("content_domain") or ("Sample" if "sample" in qid else "Practice")
        source = "Sample Exam" if "sample" in qid else "Practice Exam"
        
        by_domain[domain]["total"] += 1
        by_source[source]["total"] += 1
        if pred.get("is_correct"):
            by_domain[domain]["correct"] += 1
            by_source[source]["correct"] += 1
    
    # Plot by domain
    domains = sorted(by_domain.keys())
    accs = [by_domain[d]["correct"] / by_domain[d]["total"] * 100 for d in domains]
    counts = [by_domain[d]["total"] for d in domains]
    
    # Shorten long names
    short = [d[:25] + "..." if len(d) > 28 else d for d in domains]
    
    fig, ax = plt.subplots(figsize=(7, max(3, len(domains) * 0.5)))
    colors = ['#2d6a4f' if a >= 75 else '#e63946' for a in accs]
    bars = ax.barh(range(len(short)), accs, color=colors, height=0.6)
    ax.axvline(x=75, color='#e63946', linestyle='--', linewidth=1, label='Pass mark (75%)')
    
    for i, (bar, acc, n) in enumerate(zip(bars, accs, counts)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}% (n={n})', va='center', fontsize=9)
    
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=10)
    ax.set_title('Accuracy by Content Domain', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    out = DOCS / "per_domain_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    
    # Also print the numbers
    print("\nPer-domain breakdown:")
    for d in domains:
        acc = by_domain[d]["correct"] / by_domain[d]["total"] * 100
        print(f"  {d}: {acc:.1f}% ({by_domain[d]['correct']}/{by_domain[d]['total']})")
    
    print("\nPer-source breakdown:")
    for s in sorted(by_source.keys()):
        acc = by_source[s]["correct"] / by_source[s]["total"] * 100
        print(f"  {s}: {acc:.1f}% ({by_source[s]['correct']}/{by_source[s]['total']})")


def reproduce_guide():
    """Write a one-page reproduction guide."""
    guide = """# How to reproduce the 91.72% result

## Prerequisites

- Python 3.11+
- conda (miniforge recommended)
- Ollama (for open-source models)
- Anthropic API key with ~$10 credit

## Step-by-step (30 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/hyhossein/ribo_agent.git
cd ribo_agent
conda create -n ribo python=3.11 -y
conda activate ribo
pip install -e .[dev]
pip install anthropic

# 2. Parse the exam questions and build the knowledge base
make parse          # produces 169 eval + 386 train MCQs
make kb             # produces 297 section-level chunks
make test           # verifies everything works (87 tests)

# 3. Run zero-shot baselines (local, free)
ollama serve &
ollama pull qwen2.5:7b-instruct
ollama pull phi4-mini
make eval CONFIG=configs/v0_zeroshot_qwen25_7b.yaml      # ~15 min
make eval CONFIG=configs/v0_zeroshot_phi4_mini.yaml       # ~8 min

# 4. Run the wiki agent with Opus (~$8, ~15 min)
export ANTHROPIC_API_KEY="your-key"
make eval CONFIG=configs/v2_rewrite_wiki_opus.yaml

# 5. Run the ensemble agent (~$10, ~20 min)
make eval CONFIG=configs/v3_ensemble_opus.yaml

# 6. Run the elimination agent (~$1, ~10 min)
# (use the inline script in docs/VOTING_ANALYSIS.md or run:)
python docs/run_elimination.py

# 7. Compute the 3-way majority vote (free, instant)
python docs/compute_vote.py

# 8. View the leaderboard
make compare
```

## Expected output

```
#  model                                n    accuracy
1  3-Way Majority Vote (v6)            169  0.9172
2  Confidence Voting (v4)              169  0.8935
3  Rewrite+Wiki + Opus 4 (v2)         169  0.8876
4  Ensemble + Opus 4 (v3)             169  0.8817
5  Elimination + Opus 4 (v5)          169  0.8639
6  Opus 4 zero-shot                   169  0.7870
7  Qwen 2.5 7B zero-shot              169  0.5976
8  Phi-4 Mini zero-shot                169  0.4911
```

## Total cost to reproduce

| Component | Cost |
| :--- | ---: |
| Open-source models (Ollama) | $0 |
| Wiki compilation (one-time, cached) | ~$15 |
| Rewrite+Wiki eval (169 Qs) | ~$8 |
| Ensemble eval (169 Qs) | ~$10 |
| Elimination eval (169 Qs) | ~$1 |
| Majority vote computation | $0 |
| **Total** | **~$34** |

## Key files

| File | Purpose |
| :--- | :--- |
| `data/parsed/eval.jsonl` | 169 ground-truth MCQs |
| `data/kb/chunks.jsonl` | 297 study chunks with citations |
| `data/kb/wiki_compiled.md` | Pre-compiled knowledge wiki (cached) |
| `results/runs/*/predictions.jsonl` | Per-question predictions and traces |
| `results/runs/*/metrics.json` | Accuracy, F1, latency metrics |
| `results/LEADERBOARD.md` | Ranked comparison of all runs |

## Verification

Every number in the report can be independently verified:

```bash
# Check a specific run's accuracy
python -c "
import json
preds = [json.loads(l) for l in open('results/runs/<run_dir>/predictions.jsonl')]
correct = sum(1 for p in preds if p['is_correct'])
print(f'{correct}/{len(preds)} = {correct/len(preds):.4f}')
"

# Check for eval/train leakage (should print 0)
python -c "
from ribo_agent.parsers.dedup import fingerprint, subtract
from ribo_agent.parsers.schema import MCQ
import json
eval_qs = [MCQ(**json.loads(l)) for l in open('data/parsed/eval.jsonl')]
train_qs = [MCQ(**json.loads(l)) for l in open('data/parsed/train.jsonl')]
_, leaked = subtract(train_qs, eval_qs)
print(f'Leakage: {len(leaked)} questions')
"
```
"""
    out = DOCS / "REPRODUCE.md"
    out.write_text(guide)
    print(f"Saved: {out}")


if __name__ == "__main__":
    confusion_matrix()
    per_domain()
    reproduce_guide()
