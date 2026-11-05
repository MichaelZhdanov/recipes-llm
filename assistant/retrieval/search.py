"""Поиск рецептов по вопросу: векторный.

Вопрос уходит в API эмбеддингов только через эмбеддер, а маскирование ПДн
стоит внутри него. Один рецепт — одна точка индекса, поэтому топ-k — это
k разных рецептов. Фильтры по ингредиентам добавит этап 2.
"""

from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.store import (
    Hit,
    IndexManifest,
    IndexMismatchError,
    VectorStore,
    mismatch,
)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, *, ref: str) -> None:
        self.manifest: IndexManifest = store.manifest()
        if reason := mismatch(self.manifest, ref=ref, model=embedder.model):
            raise IndexMismatchError(f"{reason}: выполните make index")
        self._store = store
        self._embedder = embedder

    def search(self, question: str, k: int = 5) -> list[Hit]:
        return self._store.search(self._embedder.embed_query(question), limit=k)
