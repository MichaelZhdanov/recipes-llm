"""Сборка индекса: рецепты → эмбеддинги → новая коллекция Qdrant → проверка → алиас.

Рецепты идут пачками: пачка превращается в точки индекса, получает
эмбеддинги и сразу пишется в новую коллекцию. Все векторы в памяти не
держатся — на 146 тыс. рецептов это были бы гигабайты.

Поиск всё это время работает по старому индексу. Алиас переключается
на новую коллекцию, только когда в ней все точки и каждая проверена.
Если что-то упало посередине (API эмбеддингов, Qdrant, Ctrl+C),
недостроенная коллекция удаляется, а старый индекс остаётся рабочим.
"""

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from assistant import __version__
from assistant.indexing.chunks import recipe_chunk
from assistant.indexing.dataset import Recipe
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.store import IndexManifest, VectorStore

# Цена за 1M входных токенов: сумма и валюта. Только для оценки в отчёте,
# не для биллинга. Voyage — прайс MongoDB Atlas, cloud.ru — каталог
# Foundation Models; обе проверены в ноябре 2026.
PRICE_PER_MTOK: dict[str, tuple[float, str]] = {
    "voyage-4-large": (0.12, "$"),
    "voyage-4": (0.06, "$"),
    "voyage-4-lite": (0.02, "$"),
    "ai-sage/Giga-Embeddings-instruct-3B": (32.22, "₽"),
    "ai-sage/Giga-Embeddings-instruct-480M": (3.32, "₽"),
    "BAAI/bge-m3": (0.61, "₽"),
    "Qwen/Qwen3-Embedding-0.6B": (0.854, "₽"),
}
# Рецептов в пачке: эмбеддер сам делит её на запросы к API.
BATCH = 512


@dataclass(frozen=True)
class BuildReport:
    manifest: IndexManifest
    tokens: int
    cost: str | None
    masked: Counter[str]
    seconds: float


def estimate_cost(model: str, tokens: int) -> str | None:
    """Оценка стоимости вида «≈3.2100 ₽»; None, если цены модели нет в таблице."""
    if model not in PRICE_PER_MTOK:
        return None
    price, currency = PRICE_PER_MTOK[model]
    return f"≈{tokens / 1_000_000 * price:.4f} {currency}"


def build_index(
    recipes: Sequence[Recipe],
    *,
    ref: str,
    commit: str | None,
    store: VectorStore,
    embedder: Embedder,
    log: Callable[[str], None],
    quantization: bool = True,
) -> BuildReport:
    """Строит индекс заново. ref и commit — версия данных, они попадут в паспорт."""
    if not recipes:
        raise ValueError("нет рецептов для индекса")
    started = time.monotonic()
    log(f"рецептов: {len(recipes)}")
    collection: str | None = None
    manifest: IndexManifest | None = None
    try:
        for start in range(0, len(recipes), BATCH):
            chunks = [recipe_chunk(r, ref) for r in recipes[start : start + BATCH]]
            vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
            if manifest is None:
                # Размерность известна только после первого ответа API.
                manifest = IndexManifest(
                    ref=ref,
                    commit=commit,
                    embedding_model=embedder.model,
                    dim=len(vectors[0]),
                    built_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    chunks={"recipe": len(recipes)},
                    assistant_version=__version__,
                    quantization="int8" if quantization else None,
                )
                collection = store.create(manifest)
            store.upsert(chunks, vectors, collection=collection)
            log(
                f"эмбеддинги: {start + len(chunks)}/{len(recipes)}, "
                f"токенов {embedder.stats.tokens}"
            )
        stored = store.verify(collection)
        if stored != len(recipes):
            raise RuntimeError(f"в индексе {stored} точек вместо {len(recipes)}")
    except BaseException:
        if collection is not None:
            store.drop(collection)
        raise
    assert collection is not None and manifest is not None
    store.activate(collection)

    return BuildReport(
        manifest=manifest,
        tokens=embedder.stats.tokens,
        cost=estimate_cost(embedder.model, embedder.stats.tokens),
        masked=embedder.stats.masked,
        seconds=time.monotonic() - started,
    )
