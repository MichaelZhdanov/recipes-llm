# Python 3.14 и uv 0.13.5; тот же образ, что в Makefile и deploy/bootstrap.sh.
FROM ghcr.io/astral-sh/uv:0.13.5-python3.14-trixie-slim

# Байткод компилируется при установке: быстрее старт.
# Кеш uv не нужен в образе: без него слой меньше.
ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Сначала только зависимости: этот слой пересобирается, лишь когда меняется uv.lock.
# --locked: сборка падает, если uv.lock разошёлся с pyproject.toml.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY assistant ./assistant
RUN uv sync --locked --no-dev

# Сервис работает не от root. Каталог бота (том bot-data в compose) создаётся
# заранее с владельцем assistant: пустой том Docker берёт права из образа.
RUN useradd --system --uid 10001 --no-create-home assistant \
    && install -d -o assistant /app/data/bot
USER assistant

# Образ общий: по умолчанию — HTTP API, сервис bot в compose запускает
# python -m assistant.bot со своей проверкой здоровья.
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]

CMD ["/app/.venv/bin/uvicorn", "assistant.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
