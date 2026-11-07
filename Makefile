.DEFAULT_GOAL := help
.PHONY: help up down data index index-if-needed index-check grocy-seed ask search eval test lint format

# Тот же образ, что в Dockerfile. Скрипты для запуска стека выполняются в нём,
# а не хостовым python3: на разных машинах (и даже в разных shell) это разные версии.
UV_IMAGE := ghcr.io/astral-sh/uv:0.13.5-python3.14-trixie-slim

help: ## показать список команд
	@grep -E '^[a-zA-Z_.-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  make %-8s %s\n", $$1, $$2}'

# -u — чтобы файл принадлежал текущему пользователю, а не root.
INIT_ENV = docker run --rm -v "$(CURDIR):/work" -w /work -u "$$(id -u):$$(id -g)" $(UV_IMAGE) python scripts/init_env.py

# Файловая цель: выполняется, только если .env ещё нет.
.env: ## создать .env со случайными секретами
	$(INIT_ENV)

# init_env.py ещё раз: в старый .env он допишет секреты, появившиеся в новой
# версии, заданные значения не трогает.
# --wait ждёт, пока все healthcheck станут зелёными.
up: .env ## поднять стек: Qdrant и профили из COMPOSE_PROFILES (.env): api, bot, grocy
	$(INIT_ENV)
	docker compose up -d --build --wait --wait-timeout 300
	@if [ -n "$$(docker compose ps -q grocy)" ]; then $(MAKE) --no-print-directory grocy-seed; fi
	@docker compose ps --format '{{.Service}}: {{.Status}}'
	@echo "HTTP API:   http://localhost:$$(grep -s '^ASSISTANT_PORT=' .env | cut -d= -f2 | grep . || echo 8080)/docs (профиль api)"
	@echo "Qdrant:     http://localhost:6333/dashboard"

down: ## остановить стек (данные в томах Docker сохраняются)
	docker compose down

# Утилиты ассистента работают на хосте через uv: им нужен только Qdrant из стека.
# --env-file .env отдаёт им ключи; переменная из shell перекрывает значение из файла.
RUN = uv run --env-file .env python -m assistant.cli

data: .env ## скачать датасет на ревизии DATASET_REVISION (проверка sha256)
	$(RUN) data

# YES=1 — подтвердить сборку дороже 50 ₽; ARGS — прочие флаги, например
# ARGS="--sample 10000" для сравнения моделей или ARGS=--no-quantization.
index: .env ## построить индекс заново: make index [YES=1] [ARGS="--sample 10000"]
	docker compose up -d --wait qdrant
	$(RUN) index $(if $(YES),--yes) $(ARGS)

# Для деплоя: ничего не делает, если индекс уже для этой ревизии и модели.
index-if-needed: .env ## построить индекс, только если его нет или он не тот
	docker compose up -d --wait qdrant
	$(RUN) index --if-needed $(if $(YES),--yes)

index-check: .env ## индекс подходит к настройкам? ничего не строит; код 3 — нужна сборка
	docker compose up -d --wait qdrant
	$(RUN) index --check

grocy-seed: .env ## демо-склад Grocy: продукты, остатки, пользователь без прав и его ключ (идемпотентно)
	docker compose --profile grocy run --rm --build grocy-seed

# Вопрос передаётся как "$$Q": shell подставляет значение переменной, но не
# выполняет его, поэтому кавычки и $ в вопросе ничего не сломают.
ask: .env ## ответ со ссылками: make ask Q="как включить OCR на русском"
	@test -n "$$Q" || { echo 'Укажите вопрос: make ask Q="как включить OCR на русском"'; exit 2; }
	$(RUN) ask "$$Q"

search: .env ## найти рецепты, без ответа: make search Q="курица без лука" [ARGS=--no-filters]
	@test -n "$$Q" || { echo 'Укажите вопрос: make search Q="курица без лука"'; exit 2; }
	$(RUN) search "$$Q" $(ARGS)

# Код выхода 1, если метрика ниже порога из CLAUDE.md. LABEL — суффикс имени отчёта,
# STAGE=retrieval — только поиск (разбор вопроса GigaChat, фильтр, векторы, без ответа).
# SPLIT=holdout — отложенная выборка (один раз в конце). ARGS — прочие флаги evals.run,
# например ARGS="--rescore evals/reports/raw/<отчёт>.json", ARGS=--no-judge или
# ARGS=--no-filters (поиск только векторами, базовая линия этапа 1).
# Ответы: ARGS="--mode auto" — режим «модель решает» (по умолчанию ANSWER_MODE).
# Прерванный прогон продолжается с --resume: команду печатает сам прогон.
eval: .env ## прогон эталонного набора, отчёт в evals/reports/<дата>.md
	docker compose up -d --wait qdrant
	uv run --env-file .env python -m evals.run $(if $(LABEL),--label $(LABEL)) $(if $(STAGE),--stage $(STAGE)) $(if $(SPLIT),--split $(SPLIT)) $(ARGS)

test: ## unit-тесты (pytest)
	uv run pytest

lint: ## ruff check, ruff format --check, mypy, shellcheck
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run shellcheck deploy/*.sh

format: ## автоформатирование и автоисправления ruff
	uv run ruff format .
	uv run ruff check --fix .
