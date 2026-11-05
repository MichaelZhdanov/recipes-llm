"""Сборка индекса: куски → эмбеддинги → Qdrant → проверка.

Откуда берутся куски, решает вызывающий код: загрузчик рецептов появится
на этапе 1. Эмбеддинги считаются до того, как трогается коллекция. Если
API эмбеддингов упадёт на середине, старый индекс останется рабочим, а не
окажется удалённым наполовину.
"""

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from assistant import __version__
from assistant.indexing.chunks import Chunk
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
_PROGRESS_STEP = 512


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
    chunks: Sequence[Chunk],
    *,
    ref: str,
    commit: str | None,
    store: VectorStore,
    embedder: Embedder,
    log: Callable[[str], None],
) -> BuildReport:
    """Строит индекс заново. ref и commit — версия данных, они попадут в паспорт."""
    if not chunks:
        raise ValueError("нет рецептов для индекса")
    started = time.monotonic()
    counts = {"recipe": len(chunks)}
    log(f"рецептов: {len(chunks)}")

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), _PROGRESS_STEP):
        part = chunks[start : start + _PROGRESS_STEP]
        vectors.extend(embedder.embed_documents([c.embedding_text() for c in part]))
        log(
            f"эмбеддинги: {len(vectors)}/{len(chunks)}, токенов {embedder.stats.tokens}"
        )

    manifest = IndexManifest(
        ref=ref,
        commit=commit,
        embedding_model=embedder.model,
        dim=len(vectors[0]),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        chunks=dict(counts),
        assistant_version=__version__,
    )
    store.recreate(manifest)
    store.upsert(chunks, vectors)
    stored = store.verify()
    if stored != len(chunks):
        raise RuntimeError(f"в индексе {stored} точек вместо {len(chunks)}")

    return BuildReport(
        manifest=manifest,
        tokens=embedder.stats.tokens,
        cost=estimate_cost(embedder.model, embedder.stats.tokens),
        masked=embedder.stats.masked,
        seconds=time.monotonic() - started,
    )
