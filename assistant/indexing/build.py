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
from assistant.indexing.chunks import Chunk
from assistant.indexing.code import chunk_code
from assistant.indexing.docs import chunk_docs
from assistant.indexing.sources import SourceTree, ensure_sources
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.store import IndexManifest, VectorStore

# Цена за 1M токенов, $ (MongoDB Atlas, ноябрь 2026). Для оценки в отчёте, не для биллинга.
PRICE_PER_MTOK = {"voyage-4-large": 0.12, "voyage-4": 0.06, "voyage-4-lite": 0.02}
_PROGRESS_STEP = 512


@dataclass(frozen=True)
class BuildReport:
    manifest: IndexManifest
    tokens: int
    cost_usd: float | None
    masked: Counter[str]
    seconds: float


def chunk_kind(chunk: Chunk) -> str:
    """Для статистики changelog считается отдельно: это больше половины кусков документации."""
    if chunk.path == "docs/changelog.md":
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

    manifest = IndexManifest(
        ref=tree.ref,
        commit=tree.commit,
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

    price = PRICE_PER_MTOK.get(embedder.model)
    return BuildReport(
        manifest=manifest,
        tokens=embedder.stats.tokens,
        cost_usd=embedder.stats.tokens / 1_000_000 * price
        if price is not None
        else None,
        masked=embedder.stats.masked,
        seconds=time.monotonic() - started,
    )
