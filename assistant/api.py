"""HTTP API ассистента.

Запуск: `uvicorn assistant.api:create_app --factory`.

- `GET /health` — процесс жив; Qdrant, ключи и GigaChat здесь не нужны.
- `POST /ask {"question": "…"}` — ответ со ссылками, тот же, что у `make ask`.
  Если Qdrant, API эмбеддингов или GigaChat недоступны, ответ 503 с короткой
  причиной, подробности — в лог сервиса.
"""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, Protocol

import httpx2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from assistant import __version__
from assistant.config import Settings
from assistant.generation.answer import Answer, AnswerResult, make_answerer
from assistant.generation.gigachat import LlmError, make_http_client
from assistant.retrieval.embeddings import EmbeddingError
from assistant.retrieval.store import IndexMismatchError, IndexMissingError

logger = logging.getLogger(__name__)

# Вопрос в чат поддержки, а не документ: длиннее 2000 символов — скорее
# вставленный лог целиком, его разбор — этап 3.
MAX_QUESTION_CHARS = 2000


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    paperless_ref: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("вопрос пустой")
        return value.strip()


class QuestionAnswerer(Protocol):
    def answer(self, question: str) -> AnswerResult: ...


# Ошибки зависимостей: сервис жив, но ответить сейчас не может.
_SEARCH_ERRORS = (
    EmbeddingError,
    IndexMissingError,
    IndexMismatchError,
    ResponseHandlingException,
)


class LazyAnswerer:
    """Ассистент создаётся при первом вопросе, а не при старте.

    Так `/health` отвечает, даже когда Qdrant ещё поднимается или ключа нет.
    Если создать не удалось, следующий вопрос попробует снова.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._answerer: QuestionAnswerer | None = None
        self._embeddings_http = httpx2.Client(timeout=30)
        self._llm_http = make_http_client()
        self._qdrant = QdrantClient(url=settings.qdrant_url, timeout=30)

    def answer(self, question: str) -> AnswerResult:
        with self._lock:
            if self._answerer is None:
                self._answerer = make_answerer(
                    self._settings,
                    embeddings_http=self._embeddings_http,
                    llm_http=self._llm_http,
                    qdrant=self._qdrant,
                )
        return self._answerer.answer(question)

    def close(self) -> None:
        self._embeddings_http.close()
        self._llm_http.close()
        self._qdrant.close()


def create_app(
    settings: Settings | None = None, answerer: QuestionAnswerer | None = None
) -> FastAPI:
    """Собирает приложение. Глобального состояния нет: всё, что нужно, передаётся сюда.

    Uvicorn с `--factory` вызывает функцию без аргументов, и настройки читаются
    из окружения. Тесты передают свои `Settings` и заглушку вместо ассистента.
    """
    config = settings if settings is not None else Settings()
    lazy: LazyAnswerer | None = None
    if answerer is None:
        answerer = lazy = LazyAnswerer(config)
    service = answerer

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if lazy is not None:
            lazy.close()

    app = FastAPI(title="Paperless Assistant", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> HealthResponse:
        """Liveness: процесс жив и отвечает. Qdrant и Paperless здесь не проверяются."""
        return HealthResponse(
            status="ok", version=__version__, paperless_ref=config.paperless_ref
        )

    @app.post("/ask", responses={503: {"description": "поиск или LLM недоступны"}})
    def ask(request: AskRequest) -> Answer:
        """Ответ со ссылками на документацию и код или «не нашёл» с подсказками."""
        try:
            return service.answer(request.question).answer
        except _SEARCH_ERRORS as error:
            logger.error("поиск недоступен: %s", error)
            raise HTTPException(503, "поиск по документации недоступен") from error
        except LlmError as error:
            logger.error("GigaChat недоступен: %s", error)
            raise HTTPException(503, "модель GigaChat недоступна") from error
        except httpx2.TransportError as error:
            # Сеть до API эмбеддингов или GigaChat: повторы уже не помогли.
            logger.error("внешний сервис не отвечает: %s", error)
            raise HTTPException(503, "внешний сервис не отвечает") from error

    return app
