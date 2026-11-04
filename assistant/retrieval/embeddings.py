"""Эмбеддинги через внешний API.

API эмбеддингов у разных компаний устроены почти одинаково: POST
{base}/embeddings, в теле модель и список текстов, в ответе data[].embedding
и usage.total_tokens. Отличия собраны в `Provider`, а маскирование, пакеты,
повторы и учёт токенов общие и живут в `HttpEmbedder`.

Свой тонкий адаптер на httpx2 вместо SDK: это один POST-запрос, а SDK тянут
лишние зависимости (у `voyageai` это requests, aiohttp, pillow и
langchain-text-splitters).

**Каждый текст перед отправкой проходит mask_pii.** Маскирование стоит здесь,
в адаптере, а не у вызывающего кода: так его нельзя случайно обойти.
Документация и код Paperless публичны, но вопросы пользователей — нет.
"""

import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx2
from pydantic import SecretStr

from assistant.config import (
    VOYAGE_API_URL,
    VOYAGE_ATLAS_API_URL,
    EmbeddingProvider,
    Settings,
)
from assistant.http import request_with_retries
from assistant.security.pii import mask_pii

InputType = Literal["query", "document"]


class Embedder(Protocol):
    model: str
    stats: EmbeddingStats

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbeddingStats:
    """Сколько потрачено и что замаскировано — для отчётов и оценки стоимости."""

    requests: int = 0
    tokens: int = 0
    masked: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class Provider:
    """Чем отличаются API эмбеддингов разных компаний."""

    name: str
    # Имя переменной с ключом и где его взять — для понятной ошибки без ключа.
    key_env: str
    key_help: str
    # Voyage помечает вопрос и документ полем input_type; OpenAI-совместимые API так не умеют.
    sends_input_type: bool
    # Подсказки к частым ошибкам: HTTP-статус → что проверить.
    hints: Mapping[int, str] = field(default_factory=dict)


# Ключ из одного кабинета на адресе другого даёт 403: подсказываем, что проверить.
_VOYAGE_AUTH_HINT = (
    "Проверьте VOYAGE_API_KEY и VOYAGE_API_URL: ключи dash.voyageai.com работают "
    f"с {VOYAGE_API_URL}, ключи MongoDB Atlas — с {VOYAGE_ATLAS_API_URL}"
)
VOYAGE = Provider(
    name="Voyage",
    key_env="VOYAGE_API_KEY",
    key_help="создайте ключ на dash.voyageai.com",
    sends_input_type=True,
    hints={401: _VOYAGE_AUTH_HINT, 403: _VOYAGE_AUTH_HINT},
)
# cloud.ru Evolution Foundation Models. Подсказки — по ошибкам, которые
# встретились при подключении (проверено 2026-11-04).
CLOUD_RU = Provider(
    name="cloud.ru",
    key_env="CLOUD_RU_API_KEY",
    key_help=(
        "в console.cloud.ru создайте сервисный аккаунт уровня «Проект» "
        "и в нём API-ключ для Foundation Models"
    ),
    sends_input_type=False,
    hints={
        401: (
            "Нужен API-ключ сервисного аккаунта уровня «Проект»: ключ аккаунта "
            "уровня организации и пара Key ID/Key Secret не подходят"
        ),
        402: (
            "Пополните баланс в console.cloud.ru: бонусы на Foundation Models "
            "не тратятся, а пополнение доходит до API за несколько минут"
        ),
    },
)
PROVIDERS: dict[EmbeddingProvider, Provider] = {"cloudru": CLOUD_RU, "voyage": VOYAGE}
# Модель, если EMBEDDING_MODEL пустой. Для cloud.ru выбрана по eval: лучшие
# recall@5 и MRR из четырёх моделей (evals/reports/2026-11-04-models.md).
# Voyage — модель из документации Anthropic.
DEFAULT_MODELS: dict[EmbeddingProvider, str] = {
    "cloudru": "ai-sage/Giga-Embeddings-instruct-3B",
    "voyage": "voyage-4",
}

# Instruct-модели могут лучше искать, если к вопросу дописать задачу;
# документы кодируются без неё. Формат — из карточки модели на Hugging Face,
# а включается инструкция, только если помогает на eval. Qwen3-Embedding
# на cloud.ru с инструкцией из своей карточки ищет хуже, чем без неё
# (recall@5 0.35 против 0.75), поэтому его вопрос уходит как есть.
# Voyage вместо инструкции получает input_type.
RETRIEVAL_TASK = (
    "Given a user question about Paperless-ngx, "
    "retrieve documentation sections or source code that answer it"
)
_QUERY_TEMPLATES = {
    "ai-sage/Giga-Embeddings-instruct": "Instruct: {task}\nQuery: {query}",
}


def query_template(model: str) -> str | None:
    """Шаблон вопроса для модели или None, если вопрос уходит как есть."""
    for prefix, template in _QUERY_TEMPLATES.items():
        if model.startswith(prefix):
            return template
    return None


def batches(
    texts: Sequence[str], max_texts: int, max_chars: int
) -> Iterator[list[str]]:
    batch: list[str] = []
    size = 0
    for text in texts:
        if batch and (len(batch) >= max_texts or size + len(text) > max_chars):
            yield batch
            batch, size = [], 0
        batch.append(text)
        size += len(text)
    if batch:
        yield batch


class HttpEmbedder:
    # С запасом от лимитов API (у Voyage 1000 текстов и 320K токенов в запросе).
    max_batch_texts = 128
    max_batch_chars = 200_000

    def __init__(
        self,
        *,
        provider: Provider,
        api_key: SecretStr,
        model: str,
        base_url: str,
        client: httpx2.Client,
        query_template: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise EmbeddingError(
                f"не задан {provider.key_env}: {provider.key_help} "
                f"и добавьте строку {provider.key_env}=… в .env"
            )
        self.provider = provider
        self.model = model
        self.query_template = query_template
        self.stats = EmbeddingStats()
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._client = client
        self._sleep = sleep

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def _embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batches(texts, self.max_batch_texts, self.max_batch_chars):
            masked = []
            for text in batch:
                result = mask_pii(text)
                self.stats.masked.update(result.counts)
                text = result.text
                # Инструкция дописывается после маскирования: в ней нет ПДн.
                if input_type == "query" and self.query_template:
                    text = self.query_template.format(task=RETRIEVAL_TASK, query=text)
                masked.append(text)
            vectors.extend(self._request(masked, input_type))
        return vectors

    def _request(self, inputs: list[str], input_type: InputType) -> list[list[float]]:
        body: dict[str, object] = {"input": inputs, "model": self.model}
        if self.provider.sends_input_type:
            body["input_type"] = input_type
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        response = request_with_retries(
            lambda: self._client.post(self._url, json=body, headers=headers),
            sleep=self._sleep,
        )
        self.stats.requests += 1
        if response.status_code != 200:
            # Тело ответа обрезаем: нам нужна причина, а не простыня.
            message = f"{self.provider.name} ответил HTTP {response.status_code}: {response.text[:300]}"
            hint = self.provider.hints.get(response.status_code)
            if hint:
                message += f". {hint}"
            raise EmbeddingError(message)
        data = response.json()
        items = sorted(data["data"], key=lambda item: item["index"])
        if len(items) != len(inputs):
            raise EmbeddingError(
                f"{self.provider.name} вернул {len(items)} векторов на {len(inputs)} текстов"
            )
        self.stats.tokens += int(data["usage"]["total_tokens"])
        return [item["embedding"] for item in items]


def make_embedder(settings: Settings, client: httpx2.Client) -> HttpEmbedder:
    if settings.embedding_provider == "cloudru":
        api_key, base_url = settings.cloud_ru_api_key, settings.cloud_ru_api_url
    else:
        api_key, base_url = settings.voyage_api_key, settings.voyage_api_url
    model = settings.embedding_model or DEFAULT_MODELS[settings.embedding_provider]
    return HttpEmbedder(
        provider=PROVIDERS[settings.embedding_provider],
        api_key=api_key,
        model=model,
        base_url=base_url,
        client=client,
        query_template=query_template(model),
    )
