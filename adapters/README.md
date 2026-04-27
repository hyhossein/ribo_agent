# Adapters

QLoRA adapter weights for fine-tuned open-source models.

## qlora_v3_filtered/

**Best open-source result: 65.68%** on the 169-question eval set.

- **Base model:** Qwen 2.5 7B Instruct (`qwen2.5:7b-instruct` via Ollama)
- **Method:** Filtered self-distillation (see `docs/QLORA_ANALYSIS.md`)
- **Training data:** 253 chain-of-thought traces where the base model 
  independently arrived at the correct answer (filtered from 386 total)
- **Framework:** MLX QLoRA on Apple M3 Pro (36 GB)
- **Config:** 8 LoRA layers, rank 8, alpha 16, lr=2e-5, 200 iterations
- **Trainable params:** 5.7M (0.076% of 7.6B)
- **Val loss:** 1.49 → 0.61

### Loading the adapter

```python
# With MLX
from mlx_lm import load

model, tokenizer = load(
    "qwen2.5:7b-instruct",
    adapter_path="adapters/qlora_v3_filtered"
)
```

### Prior attempts (not included)

- **qlora_v1_answer_only:** Answer-letter labels only. No improvement.
  Not saved — no value in the weights.
- **qlora_v2_synthetic:** Synthetic CoT from GPT. Degraded to 47.9%.
  Not saved — actively harmful weights.

See `docs/QLORA_ANALYSIS.md` for the full analysis of all three approaches.
