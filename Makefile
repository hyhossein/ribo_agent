.PHONY: help install parse kb test clean

help:
	@echo "Targets:"
	@echo "  install   pip install -e . (once, inside activated conda env)"
	@echo "  parse     parse all question PDFs -> data/parsed/*.jsonl"
	@echo "  kb        build knowledge base   -> data/kb/chunks.jsonl"
	@echo "  test      run pytest"
	@echo "  clean     remove derived artifacts (raw inputs stay)"

install:
	pip install -e .

parse:
	python -m ribo_agent.parsers.run_parse all

kb:
	python -m ribo_agent.kb.build_kb

test:
	pytest tests

clean:
	rm -rf data/parsed data/interim data/kb data/index
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -exec rm -rf {} +
