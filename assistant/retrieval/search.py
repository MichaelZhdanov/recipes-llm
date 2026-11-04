"""Поиск кусков по вопросу: точный по тексту ошибки и векторный.

1. Точный поиск (`exact.py`): если в вопросе есть текст ошибки или лога из
   кода, куски с совпавшим шаблоном идут первыми, не больше MAX_EXACT.
2. Векторный: вопрос уходит в API эмбеддингов только через эмбеддер, а
   маскирование ПДн стоит внутри него. Из Qdrant берётся пул побольше
   (SEARCH_POOL), затем убираются повторы по source_id: части одного
   длинного раздела не должны занимать весь топ-5.

Если точных совпадений нет, выдача ровно такая же, как у одного векторного
поиска: вопросы «как сделать» точный поиск не меняет.
"""

from collections.abc import Collection

from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.exact import MessageIndex
from assistant.retrieval.store import (
    Hit,
    IndexManifest,
    IndexMismatchError,
    VectorStore,
)

SEARCH_POOL = 20
MAX_EXACT = 2


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
    def __init__(
        self, store: VectorStore, embedder: Embedder, *, ref: str, exact: bool = True
    ) -> None:
        self.manifest: IndexManifest = store.manifest()
        if self.manifest.ref != ref or self.manifest.embedding_model != embedder.model:
            raise IndexMismatchError(
                f"индекс построен для {self.manifest.ref} и модели {self.manifest.embedding_model}, "
                f"а в настройках {ref} и {embedder.model}: выполните make index"
            )
        self._store = store
        self._embedder = embedder
        # exact=False — только векторный поиск: для замера вклада точного.
        self.messages = MessageIndex(store.chunks_with_messages() if exact else [])

    def search(
        self, question: str, k: int = 5, kinds: Collection[str] | None = None
    ) -> list[Hit]:
        exact = [
            Hit(chunk=found.chunk, score=1.0, exact=True, line=found.message.line)
            for found in self.messages.search(question, limit=MAX_EXACT)
            if kinds is None or found.chunk.kind in kinds
        ]
        vector = self._embedder.embed_query(question)
        hits = self._store.search(vector, limit=max(SEARCH_POOL, k * 4), kinds=kinds)
        return dedupe([*exact, *hits])[:k]
