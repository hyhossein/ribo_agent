# RIBO Agent — Technical Deep Dive

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RIBO Agent System                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐ │
│  │ Parsers  │  │ Knowledge │  │ Agent Pipeline       │ │
│  │          │  │ Base      │  │                      │ │
│  │ sample.py│  │           │  │ v0: zero-shot        │ │
│  │practice  │  │ chunker   │  │ v1: wiki compilation │ │
│  │ manual   │  │ ingest    │  │ v2: rewrite + wiki   │ │
│  │          │  │ wiki      │  │ v3: ensemble + RAG   │ │
│  │ 169 eval │  │ BM25      │  │ v4: confidence vote  │ │
│  │ 386 train│  │           │  │ v5: elimination      │ │
│  │          │  │ 297 chunks│  │ v6: 3-way majority   │ │
│  │ dedup.py │  │ wiki cache│  │ v7: few-shot         │ │
│  └──────────┘  └───────────┘  │ v9: QLoRA distill    │ │
│                               └──────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              LLM Backend (Protocol)              │   │
│  │                                                  │   │
│  │  OllamaClient  AnthropicClient  AzureMLClient   │   │
│  │  (local)        (API)            (production)    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Eval Harness                        │   │
│  │                                                  │   │
│  │  metrics.py  runner.py  compare.py  sweep.py     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 2. Data Pipeline

### 2.1 PDF Parsing

Three parsers handle different PDF formats:

- **sample.py**: Uses `pdftotext -layout` for the Sample Questions PDF. Regex-based MCQ extraction. Answer key parsed from metadata table at end of PDF.
- **practice.py**: Uses `pdftotext -layout` for the Practice Exam PDF. Answer key parsed from X-grid where correct answer is marked.
- **manual.py**: Uses PyMuPDF for the RIBO Manual Questions PDFs. Bold-text answer extraction (correct answers are bolded in source).

### 2.2 Data Integrity

- **Deduplication**: SHA-256 fingerprint of normalized (stem + sorted options). `dedup.py::subtract()` ensures zero overlap between train and eval pools.
- **Unit test**: `test_manual_parser.py::test_eval_and_train_pools_have_no_overlap` — asserts eval and train share zero fingerprints.
- **Output**: `eval.jsonl` (169 MCQs), `train.jsonl` (386 MCQs). Each record has: qid, stem, options (A/B/C/D), correct, source, metadata.

### 2.3 Knowledge Base Construction

1. **Ingestion** (`ingest.py`): PDF → text via PyMuPDF, DOCX → text via LibreOffice headless conversion. Cached in `data/interim/study_txt/`.
2. **Chunking** (`chunker.py`): Section-aware splitting. Respects document structure (articles, sections, subsections). Each chunk has: id, source, section_title, text, extras. 297 chunks total across 8 source documents.
3. **Wiki compilation**: All 297 chunks fed to Opus with instructions to organize by topic, resolve cross-references, note contradictions. Output cached as `wiki_compiled.md`. One-time cost ~$20.

## 3. Agent Variants — Technical Details

### v0: Zero-Shot
```python
prompt = f"Question: {stem}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer: "
response = llm.generate(prompt, temperature=0.0)
answer = extract_letter(response)
```

### v2: Rewrite + Wiki
```python
# Stage 1: Rewrite
rewrite_prompt = f"Expand abbreviations, identify regulation: {stem}"
clarified = llm.generate(rewrite_prompt)

# Stage 2: Answer with wiki
answer_prompt = f"WIKI:\n{wiki}\n\nQuestion: {clarified}\n..."
response = llm.generate(answer_prompt, temperature=0.0)
```

### v5: Elimination
```python
# Three-stage elimination
prompt1 = f"Which option is DEFINITELY wrong? {stem} {options}"
eliminated_1 = llm.generate(prompt1)  # Remove worst option

prompt2 = f"Of remaining 3, which is wrong? ..."
eliminated_2 = llm.generate(prompt2)  # Remove next worst

prompt3 = f"Of remaining 2, which is correct? Cite regulation."
answer = llm.generate(prompt3)  # Final answer with citation
```

### v6: 3-Way Majority Vote
```python
answers = {
    "v2_rewrite_wiki": predictions_v2[qid],
    "v3_ensemble": predictions_v3[qid],
    "v5_elimination": predictions_v5[qid],
}
final = Counter(answers.values()).most_common(1)[0][0]
```

### v9: QLoRA Self-Distillation
```python
# Step 1: Generate reasoning traces
for q in train_386:
    response = qwen.generate(q, temperature=0.0, chain_of_thought=True)
    if response.answer == q.correct:
        keep(q, response.reasoning)  # 253 kept, 133 discarded

# Step 2: QLoRA training (MLX on Apple Silicon)
# 8 LoRA layers, lr=2e-5, 200 iters, batch_size=1
# 5.7M trainable params / 7.6B total = 0.076%

# Step 3: Inference with adapters
model = load("Qwen2.5-7B-4bit", adapter_path="adapters/")
answer = model.generate(question, max_tokens=400)
```

## 4. Eval Harness

### Metrics Computed
- **Accuracy**: correct / total
- **Macro-F1**: mean of per-class F1 scores
- **Refusal rate**: questions where no answer extracted
- **Latency**: p50, p90, mean in milliseconds
- **Per-domain accuracy**: breakdown by content domain
- **Confusion matrix**: predicted vs actual (A/B/C/D)

### Compare Tool
- Scans `results/runs/` for prediction files
- Parses directory names: `{date}-{agent}_{model}/`
- Pretty-prints model and agent names via mapping dicts
- Generates markdown leaderboard and splices into README
- `--readme` flag auto-updates the leaderboard section

## 5. Key Findings

| Finding | Evidence |
|---------|---------|
| Knowledge access > model size | Wiki +10pp vs model upgrade +19pp, but wiki is reusable |
| Temperature 0.0 beats voting on regulatory MCQ | Ensemble v3 (T=0.7) broke more than it fixed |
| Different reasoning strategies have orthogonal failures | v2, v3, v5 each get unique questions right |
| Corpus gap is the ceiling | 5/11 irreducible errors = missing homeowners content |
| Filtered self-distillation works at small scale | 253 traces → +5.9pp, but answer-only and synthetic failed |
| Few-shot benefits smaller models more | Phi-4 +3.55pp vs Qwen +1.78pp |

## 6. Azure ML Deployment Guide (SDK v2)

### 6.1 Environment Setup

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-sub-id",
    resource_group_name="rg-ribo-agent",
    workspace_name="ws-ribo-agent",
)
```

### 6.2 Register the Model

```python
# Register the QLoRA-adapted model
model = Model(
    path="./adapters/",
    name="ribo-qwen-qlora",
    version="1",
    description="Qwen 2.5 7B + QLoRA adapters for RIBO Level 1 exam (65.68%)",
    properties={
        "base_model": "Qwen2.5-7B-Instruct-4bit",
        "training_traces": "253 filtered self-distillation",
        "accuracy": "0.6568",
        "framework": "mlx-lm",
    },
)
ml_client.models.create_or_update(model)
```

### 6.3 Scoring Script

```python
# score.py
import json
import logging
from mlx_lm import load, generate

logger = logging.getLogger(__name__)

def init():
    global model, tokenizer
    model_path = os.getenv("AZUREML_MODEL_DIR", ".")
    model, tokenizer = load(
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        adapter_path=os.path.join(model_path, "adapters"),
    )
    logger.info("Model loaded with QLoRA adapters")

def run(raw_data):
    data = json.loads(raw_data)
    question = data["question"]
    options = data["options"]  # {"A": "...", "B": "...", "C": "...", "D": "..."}

    prompt = (
        f"Question: {question}\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n\n"
        f"Think step by step, then state Final Answer: A, B, C, or D."
    )

    response = generate(model, tokenizer, prompt=prompt, max_tokens=400)
    
    # Extract answer
    import re
    m = re.search(r"Final [Aa]nswer:\s*([A-D])", response)
    if not m:
        m = re.search(r"answer is\s*([A-D])", response, re.IGNORECASE)
    if not m:
        m = re.search(r"([A-D])\s*\.?\s*$", response.strip())
    
    answer = m.group(1).upper() if m else None

    return json.dumps({
        "answer": answer,
        "reasoning": response[:500],
        "model": "qwen-2.5-7b-qlora-ribo",
    })
```

### 6.4 Create Endpoint and Deployment

```python
# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="ribo-agent-endpoint",
    description="RIBO Level 1 exam agent — QLoRA Qwen 2.5 7B",
    auth_mode="key",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="qwen-qlora-v1",
    endpoint_name="ribo-agent-endpoint",
    model="ribo-qwen-qlora:1",
    code_configuration=CodeConfiguration(
        code="./src/ribo_agent/",
        scoring_script="score.py",
    ),
    environment=Environment(
        image="mcr.microsoft.com/azureml/mlflow-ubuntu22.04-py311-cpu-inference:latest",
        conda_file="environment.yml",
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# Route 100% traffic
endpoint.traffic = {"qwen-qlora-v1": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

### 6.5 Test the Endpoint

```python
import json

sample_request = {
    "question": "Under the RIB Act, what is the penalty for operating without a licence?",
    "options": {
        "A": "$25,000",
        "B": "$50,000",
        "C": "$100,000",
        "D": "$250,000",
    },
}

response = ml_client.online_endpoints.invoke(
    endpoint_name="ribo-agent-endpoint",
    request_file=json.dumps(sample_request),
)
print(json.loads(response))
```

### 6.6 Blue-Green Deployment for Model Updates

```python
# Deploy new model version alongside existing
new_deployment = ManagedOnlineDeployment(
    name="qwen-qlora-v2",
    endpoint_name="ribo-agent-endpoint",
    model="ribo-qwen-qlora:2",
    # ... same config ...
)
ml_client.online_deployments.begin_create_or_update(new_deployment).result()

# Canary: route 10% to new version
endpoint.traffic = {"qwen-qlora-v1": 90, "qwen-qlora-v2": 10}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# After validation, route 100% to new version
endpoint.traffic = {"qwen-qlora-v2": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Delete old deployment
ml_client.online_deployments.begin_delete(
    name="qwen-qlora-v1", endpoint_name="ribo-agent-endpoint"
).result()
```

## 7. Production Checklist

- [ ] Register model artifacts (adapters + wiki cache) in Azure ML Model Registry
- [ ] Create managed endpoint with auto-scaling (min=1, max=3)
- [ ] Blue-green deployment pattern for model updates
- [ ] Logging: log every prediction with reasoning trace for audit
- [ ] Monitoring: track accuracy drift via periodic eval on held-out set
- [ ] Cost: estimated ~$0.02/question at Standard_DS3_v2 pricing
- [ ] Latency: target <5s p95 for single question
- [ ] Security: API key auth + VNet integration for healthcare compliance
- [ ] FHIR integration: wrap output in FHIR QuestionnaireResponse resource

## 8. Cost Analysis

| Component | Cost |
|-----------|------|
| Wiki compilation (3 builds, cached) | ~$60 |
| Opus eval runs (v0, v2, v3, v5) | ~$90 |
| Sonnet 4 eval | ~$5 |
| Iterative testing and debugging | ~$25 |
| Open-source models (Ollama, MLX) | $0 |
| QLoRA training (Apple Silicon) | $0 |
| Few-shot experiments | $0 |
| **Total** | **~$220** |

Production inference cost (Azure ML): ~$0.02/question

## 9. Negative Results (Documented)

| Experiment | Result | Why it failed |
|-----------|--------|---------------|
| Ensemble v3 (T=0.7 voting) | 88.17% (net -0.59pp) | Temperature noise on deterministic questions |
| Self-consistency Qwen 5x | 59.76% (no change) | Model lacks domain knowledge; voting confirms wrong answers |
| QLoRA answer-only labels | 59.76% (no change) | 6 tokens per example — no reasoning signal |
| QLoRA synthetic reasoning | 47.93% (degraded) | Template-based explanations teach wrong patterns |
| GPT-OSS 20B + wiki context | 49.11% (degraded) | 15K wiki tokens overflow 24GB memory |
| Qwen + wiki context | ~57% (degraded) | 7B model can't process 25K token wiki effectively |

## 10. Next Steps

1. **Opus teacher distillation**: Run 386 training MCQs through Opus with wiki, save reasoning traces. Train QLoRA on Opus traces instead of Qwen self-distillation. Expected: 70-75% local model accuracy.
2. **Embedding-based few-shot**: Replace keyword similarity with sentence-transformer embeddings for better few-shot retrieval.
3. **Category-aware routing**: Train a lightweight classifier that routes questions to the best agent variant based on question type.
4. **Additional corpus**: Add homeowners insurance study material to close the 5-question corpus gap.
5. **FHIR integration**: Wrap the agent in a FHIR-compatible API for healthcare system integration.
