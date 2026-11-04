"""HTTP API ассистента.

Запуск: `uvicorn assistant.api:create_app --factory`.

- `GET /` — простой веб-чат (`assistant/web/`), он ходит в `POST /ask`;
- `GET /health` — процесс жив; Qdrant, ключи и GigaChat здесь не нужны.
- `POST /ask {"question": "…"}` — ответ со ссылками, тот же, что у `make ask`.
  Если Qdrant, API эмбеддингов или GigaChat недоступны, ответ 503 с короткой
  причиной, подробности — в лог сервиса;
- `GET /login`, `POST /login`, `POST /logout`, `GET /session` — вход
  (`assistant/security/auth.py`). Без входа открыты только `/health` и страница
  входа; `NO_AUTH=1` выключает вход.
"""

import logging
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Protocol

import httpx2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from starlette.middleware.sessions import SessionMiddleware

from assistant import __version__
from assistant.config import Settings
from assistant.generation.answer import Answer, AnswerResult, make_answerer
from assistant.generation.gigachat import LlmError, make_http_client
from assistant.retrieval.embeddings import EmbeddingError
from assistant.retrieval.store import IndexMismatchError, IndexMissingError
from assistant.security.auth import (
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    LoginRequest,
    LoginRequired,
    check_config,
    credentials_match,
    fingerprint,
)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
# Страница чата берёт скрипты, стили и данные только со своего адреса: ни CDN,
# ни встроенных скриптов. Даже если в ответ модели попадёт HTML, браузер
# не выполнит чужой код (а сам чат вставляет текст через textContent).
WEB_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
WEB_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
    "/chat.css": ("chat.css", "text/css; charset=utf-8"),
}
# Страница входа: открыта без входа, стили общие с чатом.
LOGIN_FILES = {
    "/login": ("login.html", "text/html; charset=utf-8"),
    "/login.js": ("login.js", "text/javascript; charset=utf-8"),
}

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


class SessionInfo(BaseModel):
    # Кто вошёл; None — вход выключен (NO_AUTH=1) или выполнен выход.
    user: str | None


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


def _static(path: Path, media_type: str) -> Callable[[], FileResponse]:
    def serve() -> FileResponse:
        return FileResponse(path, media_type=media_type, headers=WEB_HEADERS)

    return serve


def create_app(
    settings: Settings | None = None, answerer: QuestionAnswerer | None = None
) -> FastAPI:
    """Собирает приложение. Глобального состояния нет: всё, что нужно, передаётся сюда.

    Uvicorn с `--factory` вызывает функцию без аргументов, и настройки читаются
    из окружения. Тесты передают свои `Settings` и заглушку вместо ассистента.
    """
    config = settings if settings is not None else Settings()
    if not config.no_auth:
        # Без пароля и ключа сессии сервис не стартует: открытым он не окажется.
        check_config(config)
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

    for route, (name, media_type) in WEB_FILES.items():
        app.add_api_route(
            route,
            _static(WEB_DIR / name, media_type),
            methods=["GET"],
            include_in_schema=False,
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

    @app.get("/session")
    def session() -> SessionInfo:
        """Кто вошёл: чат показывает кнопку «Выйти». Сюда без входа не попасть."""
        return SessionInfo(user=None if config.no_auth else config.web_login)

    if config.no_auth:
        logger.warning("вход выключен (NO_AUTH=1): чат и API открыты всем")
    else:
        _add_login(app, config)
    return app


def _add_login(app: FastAPI, settings: Settings) -> None:
    for route, (name, media_type) in LOGIN_FILES.items():
        app.add_api_route(
            route,
            _static(WEB_DIR / name, media_type),
            methods=["GET"],
            include_in_schema=False,
        )

    @app.post("/login", responses={401: {"description": "неверный логин или пароль"}})
    def login(request: Request, body: LoginRequest) -> SessionInfo:
        if not credentials_match(settings, body.login, body.password):
            logger.warning("неудачная попытка входа")
            raise HTTPException(401, "неверный логин или пароль")
        # Новая сессия, а не дописывание в старую: чужую cookie не «довходить».
        request.session.clear()
        request.session["auth"] = fingerprint(settings)
        return SessionInfo(user=settings.web_login)

    @app.post("/logout")
    def logout(request: Request) -> SessionInfo:
        request.session.clear()
        return SessionInfo(user=None)

    # Добавленный последним middleware — внешний: сначала SessionMiddleware
    # читает cookie, потом LoginRequired решает, пускать ли дальше.
    app.add_middleware(LoginRequired, settings=settings)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.web_session_secret.get_secret_value(),
        session_cookie=SESSION_COOKIE,
        max_age=SESSION_MAX_AGE,
        # strict: cookie не уходит с запросами с чужих сайтов, это защита от CSRF.
        same_site="strict",
        # Только по HTTPS; Chrome и Firefox делают исключение для localhost.
        https_only=True,
    )
