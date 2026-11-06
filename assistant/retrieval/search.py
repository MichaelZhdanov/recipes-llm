"""Поиск рецептов по вопросу: фильтр по ингредиентам плюс векторы.

Вопрос уходит в API эмбеддингов только через эмбеддер, а маскирование ПДн
стоит внутри него. Один рецепт — одна точка индекса, поэтому топ-k — это
k разных рецептов. Условия «с чем и без чего» (`IngredientFilter`)
проверяет Qdrant, векторы ранжируют то, что прошло фильтр.
"""

from collections.abc import Sequence

from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.filters import IngredientFilter
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

    def search(
        self,
        question: str,
        k: int = 5,
        conditions: IngredientFilter | None = None,
        skip: Sequence[int] = (),
    ) -> list[Hit]:
        """skip — id рецептов, которых не должно быть в выдаче."""
        vector = self._embedder.embed_query(question)
        return self._store.search(vector, limit=k, conditions=conditions, skip=skip)

    def search_all(
        self,
        queries: Sequence[str],
        k: int = 5,
        conditions: IngredientFilter | None = None,
        skip: Sequence[int] = (),
    ) -> list[Hit]:
        """Поиск по нескольким запросам (search_v4: на вопрос по смыслу — 2–4
        блюда). Выдача — по очереди из каждого запроса, без повторов: так в
        ответе разные блюда, а не пять вариантов первого."""
        if len(queries) == 1:
            return self.search(queries[0], k, conditions, skip)
        lists = [self.search(q, k, conditions, skip) for q in queries]
        merged: list[Hit] = []
        seen: set[int] = set()
        for rank in range(k):
            for hits in lists:
                if rank < len(hits) and hits[rank].chunk.recipe_id not in seen:
                    seen.add(hits[rank].chunk.recipe_id)
                    merged.append(hits[rank])
        return merged[:k]

    def vocabulary(self) -> dict[str, int]:
        """Словарь ингредиентов индекса — для разбора вопроса."""
        return self._store.vocabulary()

    def recipes(self, ids: Sequence[int]) -> list[Chunk]:
        """Рецепты по id — найденные раньше в диалоге."""
        return self._store.recipes(ids)
