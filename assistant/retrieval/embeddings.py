"""Эмбеддинги через Voyage AI (рекомендация Anthropic: своей модели эмбеддингов у неё нет).

Свой тонкий адаптер на httpx2 вместо SDK `voyageai`: это один POST-запрос,
а SDK тянет requests, aiohttp, pillow и langchain-text-splitters.

**Каждый текст перед отправкой проходит mask_pii.** Маскирование стоит здесь,
в адаптере, а не у вызывающего кода: так его нельзя случайно обойти.
Документация и код Paperless публичны, но вопросы пользователей — нет.

API: POST {base}/embeddings, до 1000 текстов и 320K токенов в запросе для
voyage-4. Батчи берём меньше (128 текстов, ≈200K символов), с запасом.
"""

import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import httpx2
from pydantic import SecretStr

from assistant.config import Settings
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


class VoyageEmbedder:
    max_batch_texts = 128
    max_batch_chars = 200_000

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str,
        base_url: str,
        client: httpx2.Client,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise EmbeddingError(
                "не задан VOYAGE_API_KEY: создайте ключ в MongoDB Atlas → AI Model APIs "
                "и добавьте строку VOYAGE_API_KEY=… в .env"
            )
        self.model = model
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
                masked.append(result.text)
            vectors.extend(self._request(masked, input_type))
        return vectors

    def _request(self, inputs: list[str], input_type: InputType) -> list[list[float]]:
        body = {"input": inputs, "model": self.model, "input_type": input_type}
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        response = request_with_retries(
            lambda: self._client.post(self._url, json=body, headers=headers),
            sleep=self._sleep,
        )
        self.stats.requests += 1
        if response.status_code != 200:
            # Тело ответа обрезаем: нам нужна причина, а не простыня.
            raise EmbeddingError(
                f"Voyage ответил HTTP {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        items = sorted(data["data"], key=lambda item: item["index"])
        if len(items) != len(inputs):
            raise EmbeddingError(
                f"Voyage вернул {len(items)} векторов на {len(inputs)} текстов"
            )
        self.stats.tokens += int(data["usage"]["total_tokens"])
        return [item["embedding"] for item in items]


def make_embedder(settings: Settings, client: httpx2.Client) -> VoyageEmbedder:
    return VoyageEmbedder(
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
        base_url=settings.voyage_api_url,
        client=client,
    )
