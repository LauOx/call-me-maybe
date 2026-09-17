# Configuration
#--------------

PYTHON := uv run python

MAIN := src

MYPY_FLAGS := --warn-return-any \
              --warn-unused-ignores \
              --ignore-missing-imports \
              --disallow-untyped-defs \
              --check-untyped-defs \
			  --exclude=.venv,venv,env,__pycache__,.git


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
	uv run flake8 . --exclude=.venv,venv,env,__pycache__,.git
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 . --exclude=.venv,venv,env,__pycache__,.git
	uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict