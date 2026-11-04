import hashlib
import math
import re
from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.retrieval.embeddings import Embedder, EmbeddingStats
from assistant.retrieval.store import VectorStore


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убирает переменные Settings из окружения перед каждым тестом.

    Иначе экспортированный в shell GIGACHAT_API_KEY или PAPERLESS_REF поменял бы
    значения по умолчанию, и тесты зависели бы от машины.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


class FakeEmbedder:
    """Детерминированный эмбеддер без сети: «мешок слов» с хешированием.

    Тексты с общими словами получают близкие векторы, этого хватает, чтобы
    проверить индекс и поиск, не обращаясь к Voyage.
    """

    dim = 64

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.stats = EmbeddingStats()

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.stats.requests += 1
        self.stats.tokens += sum(len(t.split()) for t in texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture
def fake_embedder() -> Embedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"))
