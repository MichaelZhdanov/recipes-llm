.DEFAULT_GOAL := help
.PHONY: help test lint format

# Тот же образ, что в Dockerfile. Скрипты для запуска стека выполняются в нём,
# а не хостовым python3: на разных машинах (и даже в разных shell) это разные версии.
UV_IMAGE := ghcr.io/astral-sh/uv:0.13.5-python3.14-trixie-slim

help: ## показать список команд
	@grep -E '^[a-zA-Z_.-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  make %-8s %s\n", $$1, $$2}'

# Файловая цель: выполняется, только если .env ещё нет. -u — чтобы файл принадлежал
# текущему пользователю, а не root.
.env: ## создать .env со случайными секретами для demo-Paperless
	docker run --rm -v "$(CURDIR):/work" -w /work -u "$$(id -u):$$(id -g)" $(UV_IMAGE) python scripts/init_env.py

test: ## unit-тесты (pytest)
	uv run pytest

lint: ## ruff check, ruff format --check, mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## автоформатирование и автоисправления ruff
	uv run ruff format .
	uv run ruff check --fix .
