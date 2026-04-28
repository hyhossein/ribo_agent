# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] — 2026-04-23

First real accuracy numbers. Ships a zero-shot agent, a full eval
harness, and per-run markdown reports so every release moves a number.

### Added
- `src/ribo_agent/agents/base.py` — `Agent` Protocol + `Prediction`
  dataclass carrying qid, predicted/correct, latency, token counts.
- `src/ribo_agent/agents/zeroshot.py` — `ZeroShotAgent` with a
  four-tier answer extractor (tag, 'Answer: X', sole letter in tail,
  any letter). Never guesses on ambiguous output; refusal rate is
  first-class in metrics.
- `src/ribo_agent/eval/metrics.py` — accuracy, macro-F1, micro-F1,
  per-class P/R/F1, per-domain + per-cognitive-level accuracy,
  confusion matrix with REFUSED column, latency p50/p90/mean.
  `format_report()` emits markdown suitable for committing.
- `src/ribo_agent/eval/runner.py` — CLI that takes a YAML config,
  runs the agent on `data/parsed/eval.jsonl`, and writes
  `results/runs/<ts>_<config>_<model>/{predictions.jsonl,metrics.json,report.md}`.
  Live progress bar with running accuracy and ETA.
- `src/ribo_agent/eval/compare.py` — leaderboard table across every
  run in `results/runs/` (plain text or markdown).
- Configs: `v0_zeroshot_qwen25_7b.yaml`, `v0_zeroshot_llama31_8b.yaml`,
  `v0_zeroshot_phi35.yaml`.
- Makefile: `eval CONFIG=...`, `eval-all`, `compare`.
- 17 new tests (81 total) using a `MockLLM` so CI runs the full agent
  path in ms without an Ollama server.

### Notes
- LLM backend is Ollama running locally; swap `backend: ollama` →
  `backend: azureml` in the config to route at Azure ML Managed
  Endpoints without code changes.
- Retrieval lands in v0.5.0; zero-shot is the floor RAG lift will be
  measured against.

## [0.3.1] — 2026-04-23

### Fixed
- `src/ribo_agent/kb/ingest.py` now discovers the LibreOffice binary
  on macOS (the Homebrew cask ships it as `soffice`, not
  `libreoffice`). Probes PATH + known absolute paths; clear error if
  neither is present.

## [0.3.0] — 2026-04-23

Knowledge-base build and full training pool. No LLM yet; that lands
in v0.5.0.

### Added
- `src/ribo_agent/parsers/manual.py` — manual-PDF bold-answer
  extractor using PyMuPDF per-span font metadata. Adds 386 MCQs to
  the few-shot pool. MIN_BOLD_FRAC guard skips questions where no
  option is bold rather than guessing.
- `src/ribo_agent/parsers/dedup.py` — SHA fingerprint + `dedup()` +
  `subtract()`. Used as the authoritative leakage check between the
  eval set (169 Q) and the train pool (386 Q). Test asserts zero
  shared fingerprints.
- `src/ribo_agent/kb/ingest.py` — `.doc` → UTF-8 via LibreOffice
  headless, `.pdf` → pdftotext -layout, with a SHA-keyed cache so
  repeated runs skip the slow conversions.
- `src/ribo_agent/kb/chunker.py` — four per-format splitters +
  size normaliser. 298 chunks across 8 study sources. Every chunk
  carries a human-readable citation ("O.Reg. 991 s. 14").
- `src/ribo_agent/kb/build_kb.py` — CLI entry point writing
  `data/kb/chunks.jsonl` plus a summary report.
- `make kb` target + `pytest tests/test_kb_chunker.py` (7 tests).
- CI workflow now also smoke-runs `python -m ribo_agent.kb.build_kb`
  and uploads the parsed JSONL + KB as build artifacts.

### Changed
- `run_parse.py` now produces five JSONL files: three raw parsed
  sets + derived `eval.jsonl` (sample + practice) and `train.jsonl`
  (manual minus eval fingerprints).

### Test count
- v0.2.0: 47 tests
- v0.3.0: 64 tests
- Runtime: ~3.5s

## [0.2.0] — 2026-04-23

Infrastructure release. No new parsing or eval numbers; this version wires
up the bones so every subsequent release ships cleanly.

### Added
- `LLMClient` abstraction (`src/ribo_agent/llm/base.py`) with two
  implementations: `OllamaClient` (local development) and `AzureMLClient`
  (managed-endpoint deployment). Clients share the same `complete()`
  signature; the backend is chosen via config file.
- `configs/v0_baseline.yaml` as the first example configuration showing
  the local→Azure ML switch point.
- Storage abstraction (`src/ribo_agent/io/storage.py`) with local and
  Azure Blob backends, selected via environment variable.
- Continuous Integration on GitHub Actions (`.github/workflows/ci.yml`)
  running `pytest` on every push and pull request against `main`.
- `LICENSE` (MIT), `CODEOWNERS`, `CHANGELOG.md`.
- Status and CI badges on the README.

### Changed
- `README.md` rewritten for a mixed technical + executive audience. The
  day-by-day plan moved to `PLAN.md`.
- `pyproject.toml` dropped `llama-cpp-python` in favour of an HTTP client
  (`httpx`) since both Ollama and Azure ML Managed Endpoints speak HTTP.

## [0.1.0] — 2026-04-23

First usable slice. Everything from EDA through a clean eval set.

### Added
- Day 1 exploratory data analysis (`notebooks/day1_eda.py`,
  `docs/EDA.md`) documenting file inventory, question-file structure
  classification, and two concrete parser traps (form-feed page
  boundaries and drifting X-grid columns).
- Canonical `MCQ` dataclass (`src/ribo_agent/parsers/schema.py`).
- Parser for `Sample-Questions-RIBO-Level-1-Exam.pdf` — 79 MCQs with
  Content Domain / Competency / Cognitive Level metadata.
- Parser for `695993459-Practise-RIBO-Exam.pdf` — 90 MCQs with X-grid
  answer-key column clustering and form-feed-aware question detection.
- `Makefile` targets: `install`, `parse`, `test`, `clean`.
- 35 pytest unit tests covering both parsers, including parametrized
  regression tests for the two EDA-identified traps.

[Unreleased]: https://github.com/hyhossein/ribo_agent/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/hyhossein/ribo_agent/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/hyhossein/ribo_agent/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/hyhossein/ribo_agent/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hyhossein/ribo_agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hyhossein/ribo_agent/releases/tag/v0.1.0
