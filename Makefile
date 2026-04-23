.PHONY: help install parse kb test eval eval-all compare clean

CONFIG ?= configs/v0_zeroshot_qwen25_7b.yaml

help:
	@echo "Targets:"
	@echo "  install            pip install -e . (once, inside activated conda env)"
	@echo "  parse              parse question PDFs -> data/parsed/*.jsonl"
	@echo "  kb                 build knowledge base -> data/kb/chunks.jsonl"
	@echo "  test               run pytest"
	@echo "  eval CONFIG=..     run eval for a single config (default: Qwen 7B)"
	@echo "  eval-all           run eval for every configs/v0_zeroshot_*.yaml"
	@echo "  compare            print leaderboard of all runs in results/runs/"
	@echo "  clean              remove derived artifacts (raw inputs stay)"

install:
	pip install -e .

parse:
	python -m ribo_agent.parsers.run_parse all

kb:
	python -m ribo_agent.kb.build_kb

test:
	pytest tests

eval:
	python -m ribo_agent.eval.runner --config $(CONFIG)

eval-all:
	@for cfg in configs/v0_zeroshot_*.yaml; do \
		echo ">>> $$cfg"; \
		python -m ribo_agent.eval.runner --config $$cfg || exit $$?; \
	done

compare:
	python -m ribo_agent.eval.compare

clean:
	rm -rf data/parsed data/interim data/kb data/index
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -exec rm -rf {} +
