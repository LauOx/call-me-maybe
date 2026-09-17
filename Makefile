# Configuration
#--------------

PYTHON := uv run python

MAIN := src

FLAKE8_EXCLUDE := .venv,venv,env,__pycache__,.git,moulinette,llm_sdk
MYPY_EXCLUDE := (\.venv|venv|env|__pycache__|\.git|moulinette|llm_sdk)

MYPY_FLAGS := --warn-return-any \
              --warn-unused-ignores \
              --ignore-missing-imports \
              --disallow-untyped-defs \
              --check-untyped-defs \
              --exclude '$(MYPY_EXCLUDE)'


# Rules
#------

install:
	uv sync

run:
	$(PYTHON) -m $(MAIN)

debug:
	$(PYTHON) -m pdb -m $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	uv run mypy . --strict --exclude '$(MYPY_EXCLUDE)'

.PHONY: install run debug clean lint lint-strict