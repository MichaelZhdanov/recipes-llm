.DEFAULT_GOAL := help
.PHONY: help test lint format

help: ## показать список команд
	@grep -E '^[a-zA-Z_.-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  make %-8s %s\n", $$1, $$2}'

test: ## unit-тесты (pytest)
	uv run pytest

lint: ## ruff check, ruff format --check, mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## автоформатирование и автоисправления ruff
	uv run ruff format .
	uv run ruff check --fix .
