"""Поиск кусков по вопросу: векторный.

Вопрос уходит в API эмбеддингов только через эмбеддер, а маскирование ПДн
стоит внутри него. Из Qdrant берётся пул побольше (SEARCH_POOL), затем
убираются повторы по source_id: части одного длинного раздела не должны
занимать весь топ-5.
"""

from collections.abc import Collection

from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.store import (
    Hit,
    IndexManifest,
    IndexMismatchError,
    VectorStore,
    mismatch,
)

SEARCH_POOL = 20


def dedupe(hits: list[Hit]) -> list[Hit]:
    """Оставляет лучший кусок каждого источника, порядок по score сохраняется."""
    seen: set[str] = set()
    unique = []
    for hit in hits:
        if hit.chunk.source_id not in seen:
            seen.add(hit.chunk.source_id)
            unique.append(hit)
    return unique


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, *, ref: str) -> None:
        self.manifest: IndexManifest = store.manifest()
        if reason := mismatch(self.manifest, ref=ref, model=embedder.model):
            raise IndexMismatchError(f"{reason}: выполните make index")
        self._store = store
        self._embedder = embedder

    def search(
        self, question: str, k: int = 5, kinds: Collection[str] | None = None
    ) -> list[Hit]:
        vector = self._embedder.embed_query(question)
        hits = self._store.search(vector, limit=max(SEARCH_POOL, k * 4), kinds=kinds)
        return dedupe(hits)[:k]
