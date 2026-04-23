# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/hyhossein/ribo_agent/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hyhossein/ribo_agent/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hyhossein/ribo_agent/releases/tag/v0.1.0
