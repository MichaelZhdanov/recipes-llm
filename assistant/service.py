"""Ассистент для каналов — HTTP API и Telegram-бота.

Оба канала создают ассистента лениво и одинаково различают ошибки
зависимостей: Qdrant, API эмбеддингов или LLM сейчас недоступны — сервис
жив, но ответить не может.
"""

import threading
from collections.abc import Sequence
from typing import Protocol

import httpx2
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from assistant.config import Settings
from assistant.generation.answer import AnswerResult, Progress, make_answerer
from assistant.generation.gigachat import LlmError, Message
from assistant.generation.llm import make_http_client
from assistant.retrieval.embeddings import EmbeddingError
from assistant.retrieval.store import IndexMismatchError, IndexMissingError

# Вопрос, а не документ: длиннее 2000 символов вопросы не бывают.
MAX_QUESTION_CHARS = 2000

# Поиск недоступен: индекса нет, он не тот, Qdrant или API эмбеддингов не отвечают.
SEARCH_ERRORS = (
    EmbeddingError,
    IndexMissingError,
    IndexMismatchError,
    ResponseHandlingException,
)
# Все ошибки зависимостей: поиск, LLM и сеть до них, когда повторы уже не помогли.
DEPENDENCY_ERRORS = (*SEARCH_ERRORS, LlmError, httpx2.TransportError)


class QuestionAnswerer(Protocol):
    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
    ) -> AnswerResult: ...


class LazyAnswerer:
    """Ассистент создаётся при первом вопросе, а не при старте.

    Так `/health` и бот отвечают, даже когда Qdrant ещё поднимается или ключа
    нет. Если создать не удалось, следующий вопрос попробует снова.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._answerer: QuestionAnswerer | None = None
        self._embeddings_http = httpx2.Client(timeout=30)
        self._llm_http = make_http_client(settings)
        self._qdrant = QdrantClient(url=settings.qdrant_url, timeout=30)

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
    ) -> AnswerResult:
        with self._lock:
            if self._answerer is None:
                self._answerer = make_answerer(
                    self._settings,
                    embeddings_http=self._embeddings_http,
                    llm_http=self._llm_http,
                    qdrant=self._qdrant,
                )
        return self._answerer.answer(question, history, progress)

    def close(self) -> None:
        self._embeddings_http.close()
        self._llm_http.close()
        self._qdrant.close()
