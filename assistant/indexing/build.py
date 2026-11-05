"""Сборка индекса: рецепты → эмбеддинги → новая коллекция Qdrant → проверка → алиас.

Рецепты идут пачками: пачка превращается в точки индекса, получает
эмбеддинги и сразу пишется в новую коллекцию. Все векторы в памяти не
держатся — на 146 тыс. рецептов это были бы гигабайты.

Поиск всё это время работает по старому индексу. Алиас переключается
на новую коллекцию, только когда в ней все точки и каждая проверена.
Если что-то упало посередине (API эмбеддингов, Qdrant, Ctrl+C),
недостроенная коллекция удаляется, а старый индекс остаётся рабочим.
"""

import math
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from assistant import __version__
from assistant.indexing.chunks import recipe_chunk
from assistant.indexing.dataset import DATASET_REPO, Recipe
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
# Грубая оценка до сборки: символов на токен у русского текста. Взята
# с запасом (меньше — значит дороже), точное число даёт usage из ответа API.
CHARS_PER_TOKEN = 3.0
# Сборка дороже этого просит подтверждения (решение автора 2026-11-05).
CONFIRM_ABOVE = {"₽": 50.0, "$": 0.5}
SAMPLE_SEED = 20261105


@dataclass(frozen=True)
class IndexSource:
    """Откуда рецепты: это попадает в паспорт индекса."""

    ref: str
    dataset: str = DATASET_REPO
    file_sha256: str = ""
    skipped: Mapping[str, int] = field(default_factory=dict)
    sample: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class Estimate:
    tokens: int
    requests: int
    # (сумма, валюта) или None, если цены модели нет в таблице.
    cost: tuple[float, str] | None

    @property
    def needs_confirmation(self) -> bool:
        return self.cost is not None and self.cost[0] > CONFIRM_ABOVE[self.cost[1]]

    def __str__(self) -> str:
        cost = (
            f"≈{self.cost[0]:.2f} {self.cost[1]}"
            if self.cost
            else "цена модели неизвестна"
        )
        return f"≈{self.tokens} токенов, {cost}, около {self.requests} запросов к API"


@dataclass(frozen=True)
class BuildReport:
    manifest: IndexManifest
    tokens: int
    cost: str | None
    masked: Counter[str]
    seconds: float


def price(model: str, tokens: int) -> tuple[float, str] | None:
    if model not in PRICE_PER_MTOK:
        return None
    per_million, currency = PRICE_PER_MTOK[model]
    return tokens / 1_000_000 * per_million, currency


def estimate_cost(model: str, tokens: int) -> str | None:
    """Оценка стоимости вида «≈3.2100 ₽»; None, если цены модели нет в таблице."""
    cost = price(model, tokens)
    return f"≈{cost[0]:.4f} {cost[1]}" if cost else None


def estimate(recipes: Sequence[Recipe], model: str, batch_texts: int) -> Estimate:
    """Сколько примерно стоит индекс, ещё до первого запроса к API."""
    chars = sum(len(recipe_chunk(r, "-").embedding_text()) for r in recipes)
    tokens = math.ceil(chars / CHARS_PER_TOKEN)
    return Estimate(
        tokens=tokens,
        requests=math.ceil(len(recipes) / batch_texts),
        cost=price(model, tokens),
    )


def pick_sample(recipes: Sequence[Recipe], size: int, seed: int) -> list[Recipe]:
    """Выборка с фиксированным сидом: у всех моделей одни и те же рецепты."""
    if size >= len(recipes):
        return list(recipes)
    picked = random.Random(seed).sample(list(recipes), size)
    return sorted(picked, key=lambda r: r.id)


def build_index(
    recipes: Sequence[Recipe],
    *,
    source: IndexSource,
    store: VectorStore,
    embedder: Embedder,
    log: Callable[[str], None],
    quantization: bool = True,
) -> BuildReport:
    """Строит индекс заново; source попадает в паспорт."""
    if not recipes:
        raise ValueError("нет рецептов для индекса")
    started = time.monotonic()
    log(f"рецептов: {len(recipes)}")
    collection: str | None = None
    manifest: IndexManifest | None = None
    try:
        for start in range(0, len(recipes), BATCH):
            chunks = [
                recipe_chunk(r, source.ref) for r in recipes[start : start + BATCH]
            ]
            vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
            if manifest is None:
                # Размерность известна только после первого ответа API.
                manifest = IndexManifest(
                    dataset=source.dataset,
                    ref=source.ref,
                    file_sha256=source.file_sha256,
                    recipes=len(recipes),
                    skipped=dict(source.skipped),
                    sample=source.sample,
                    seed=source.seed,
                    embedding_model=embedder.model,
                    dim=len(vectors[0]),
                    quantization="int8" if quantization else None,
                    built_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    assistant_version=__version__,
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
