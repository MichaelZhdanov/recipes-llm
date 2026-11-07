"""HTTP API ассистента.

Запуск: `uvicorn assistant.api:app_from_env --factory` — настройки из
окружения и JSON-логи; тесты собирают приложение `create_app` со своими.

- `GET /health` — процесс жив; Qdrant, ключи и LLM здесь не нужны.
- `POST /ask {"question": "…"}` — ответ со ссылками, тот же, что у `make ask`.
  У каждого ответа — заголовок `X-Request-ID`: та же метка стоит на всех
  записях лога по пути этого запроса.
- `GET /metrics` — метрики Prometheus: вопросы, время ответа, токены и цена
  (`assistant/metrics.py`).
  Если Qdrant, API эмбеддингов или LLM недоступны, ответ 503 с короткой
  причиной, подробности — в лог сервиса.

Входа здесь нет: порт сервиса опубликован только на 127.0.0.1, а
пользователи приходят через Telegram-бота (`assistant/bot/`).
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

import httpx2
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator

from assistant import __version__
from assistant.config import Settings
from assistant.generation.answer import Answer
from assistant.generation.gigachat import LlmError
from assistant.logs import REQUEST_ID, setup_logging
from assistant.metrics import Metrics, dependency_kind
from assistant.service import (
    MAX_QUESTION_CHARS,
    SEARCH_ERRORS,
    LazyAnswerer,
    QuestionAnswerer,
)

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    dataset_revision: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("вопрос пустой")
        return value.strip()


def create_app(
    settings: Settings | None = None, answerer: QuestionAnswerer | None = None
) -> FastAPI:
    """Собирает приложение. Глобального состояния нет: всё, что нужно, передаётся сюда.

    Без настроек они читаются из окружения. Тесты передают свои `Settings` и
    заглушку вместо ассистента; логи настраивает `app_from_env`, не эта функция.
    """
    config = settings if settings is not None else Settings()
    lazy: LazyAnswerer | None = None
    if answerer is None:
        answerer = lazy = LazyAnswerer(config)
    service = answerer
    metrics = Metrics(usage=lazy.usage if lazy is not None else lambda: None)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if lazy is not None:
            lazy.close()

    app = FastAPI(title="Recipes LLM", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Метка запроса для лога (assistant/logs.py) и заголовок X-Request-ID."""
        value = uuid.uuid4().hex[:12]
        token = REQUEST_ID.set(value)
        try:
            response = await call_next(request)
        finally:
            REQUEST_ID.reset(token)
        response.headers["X-Request-ID"] = value
        return response

    @app.get("/health")
    def health() -> HealthResponse:
        """Liveness: процесс жив и отвечает. Qdrant и ключи API здесь не проверяются."""
        return HealthResponse(
            status="ok", version=__version__, dataset_revision=config.dataset_revision
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics_text() -> Response:
        return Response(metrics.text(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/ask", responses={503: {"description": "поиск или LLM недоступны"}})
    def ask(request: AskRequest) -> Answer:
        """Ответ со ссылками на источники или «не нашёл» с подсказками."""
        started = time.monotonic()
        try:
            result = service.answer(request.question)
        except (*SEARCH_ERRORS, LlmError, httpx2.TransportError) as error:
            metrics.record(
                "answer",
                {
                    "source": "api",
                    "status": "error",
                    "error_kind": dependency_kind(error),
                },
            )
            raise _unavailable(error) from error
        metrics.record(
            "answer",
            {
                "source": "api",
                "status": result.answer.status,
                "reason": result.answer.reason,
                "search": result.trace.args,
                "seconds": time.monotonic() - started,
                "retrieval_seconds": result.trace.retrieval_seconds,
                "generation_seconds": result.trace.generation_seconds,
            },
        )
        return result.answer

    return app


def _unavailable(error: Exception) -> HTTPException:
    """503 с короткой причиной; подробности — в лог."""
    if isinstance(error, SEARCH_ERRORS):
        logger.error("поиск недоступен: %s", error)
        return HTTPException(503, "поиск недоступен")
    if isinstance(error, LlmError):
        logger.error("LLM недоступна: %s", error)
        return HTTPException(503, "модель недоступна")
    # Сеть до API эмбеддингов или LLM: повторы уже не помогли.
    logger.error("внешний сервис не отвечает: %s", error)
    return HTTPException(503, "внешний сервис не отвечает")


def app_from_env() -> FastAPI:
    """Для uvicorn (`--factory`): настройки из окружения, логи — JSON-строками."""
    settings = Settings()
    setup_logging(settings)
    return create_app(settings)
