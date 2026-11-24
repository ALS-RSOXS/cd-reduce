.PHONY: verify fix lint format format-check type-check install test test-cov validation docs docs-serve release-check

# Verify: check everything without changing files
verify: lint format-check test

fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

format-check:
	uv run ruff format --check src tests

type-check:
	uv run ty check src

install:
	uv sync --all-groups

test:
	uv run pytest -q

test-cov:
	uv run pytest -q --cov=cd_reduce --cov-report=xml --cov-report=term-missing

# HOWTO regression suite (~4 min), deselected from the default test run
validation:
	uv run pytest -q -m validation

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

# Does the version in pyproject.toml / cd_reduce.__version__ still need a release?
release-check:
	python3 scripts/release_check.py
