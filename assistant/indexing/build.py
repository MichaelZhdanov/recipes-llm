"""Сборка индекса: исходники по тегу → куски → эмбеддинги → Qdrant.

Эмбеддинги считаются до того, как трогается коллекция. Если Voyage упадёт
на середине, старый индекс останется рабочим, а не окажется удалённым
наполовину.
"""

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx2

from assistant import __version__
from assistant.config import Settings
from assistant.indexing.chunks import CHANGELOG_PATH, Chunk
from assistant.indexing.code import chunk_code, message_coverage
from assistant.indexing.docs import chunk_docs
from assistant.indexing.sources import SourceTree, ensure_sources
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


def chunk_kind(chunk: Chunk) -> str:
    """Для статистики changelog считается отдельно: это больше половины кусков документации."""
    if chunk.path == CHANGELOG_PATH:
        return "changelog"
    return chunk.kind


def collect_chunks(tree: SourceTree, repo_url: str) -> list[Chunk]:
    return chunk_docs(tree.root, ref=tree.ref, repo_url=repo_url) + chunk_code(
        tree.root, ref=tree.ref, repo_url=repo_url
    )


def build_index(
    settings: Settings,
    *,
    http: httpx2.Client,
    store: VectorStore,
    embedder: Embedder,
    log: Callable[[str], None],
) -> BuildReport:
    started = time.monotonic()
    tree = ensure_sources(
        settings.paperless_repo_url,
        settings.paperless_ref,
        settings.paperless_source_path,
        http,
    )
    log(f"исходники {tree.ref} ({tree.commit or 'коммит неизвестен'}) в {tree.root}")
    chunks = collect_chunks(tree, settings.paperless_repo_url)
    counts = Counter(chunk_kind(c) for c in chunks)
    log(
        f"кусков: {len(chunks)} ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})"
    )

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), _PROGRESS_STEP):
        part = chunks[start : start + _PROGRESS_STEP]
        vectors.extend(embedder.embed_documents([c.embedding_text() for c in part]))
        log(
            f"эмбеддинги: {len(vectors)}/{len(chunks)}, токенов {embedder.stats.tokens}"
        )

    coverage = message_coverage(tree.root)
    log(
        f"шаблонов сообщений: {sum(len(c.messages) for c in chunks)}, "
        f"извлечено у {coverage.covered} из {coverage.calls} вызовов raise/логгера"
    )
    manifest = IndexManifest(
        ref=tree.ref,
        commit=tree.commit,
        embedding_model=embedder.model,
        dim=len(vectors[0]),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        chunks=dict(counts),
        assistant_version=__version__,
        messages={
            "templates": sum(len(c.messages) for c in chunks),
            "calls": coverage.calls,
            "covered": coverage.covered,
        },
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
