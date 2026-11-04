.DEFAULT_GOAL := help
.PHONY: help up down index ask test lint format

# Тот же образ, что в Dockerfile. Скрипты для запуска стека выполняются в нём,
# а не хостовым python3: на разных машинах (и даже в разных shell) это разные версии.
UV_IMAGE := ghcr.io/astral-sh/uv:0.13.5-python3.14-trixie-slim

help: ## показать список команд
	@grep -E '^[a-zA-Z_.-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  make %-8s %s\n", $$1, $$2}'

# Файловая цель: выполняется, только если .env ещё нет. -u — чтобы файл принадлежал
# текущему пользователю, а не root.
.env: ## создать .env со случайными секретами для demo-Paperless
	docker run --rm -v "$(CURDIR):/work" -w /work -u "$$(id -u):$$(id -g)" $(UV_IMAGE) python scripts/init_env.py

# --wait ждёт, пока все healthcheck станут зелёными; первый старт Paperless
# (миграции базы) занимает до пары минут.
up: .env ## поднять Paperless, Valkey, Qdrant и ассистента
	docker compose up -d --build --wait --wait-timeout 300
	@echo "Paperless:  http://localhost:8000 (логин и пароль в .env)"
	@echo "Ассистент:  http://localhost:8080/health"
	@echo "Qdrant:     http://localhost:6333/dashboard"

down: ## остановить стек (данные в томах Docker сохраняются)
	docker compose down

# Утилиты ассистента работают на хосте через uv: им нужен только Qdrant из стека.
# --env-file .env отдаёт им ключи; переменная из shell перекрывает значение из файла.
RUN = uv run --env-file .env python -m assistant.cli

index: .env ## пересобрать индекс: исходники по тегу, эмбеддинги Voyage, Qdrant
	docker compose up -d --wait qdrant
	$(RUN) index

# Вопрос передаётся как "$$Q": shell подставляет значение переменной, но не
# выполняет его, поэтому кавычки и $ в вопросе ничего не сломают.
ask: .env ## найти куски по вопросу: make ask Q="как включить OCR на русском"
	@test -n "$$Q" || { echo 'Укажите вопрос: make ask Q="как включить OCR на русском"'; exit 2; }
	$(RUN) ask "$$Q"

test: ## unit-тесты (pytest)
	uv run pytest

lint: ## ruff check, ruff format --check, mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## автоформатирование и автоисправления ruff
	uv run ruff format .
	uv run ruff check --fix .
