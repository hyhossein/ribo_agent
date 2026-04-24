# How to reproduce the 91.72% result

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
