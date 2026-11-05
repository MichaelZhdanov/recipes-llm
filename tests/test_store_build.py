import json
from collections.abc import Sequence
from pathlib import Path

import httpx2
import pytest

from assistant.config import Settings
from assistant.indexing.build import build_index, estimate_cost
from assistant.indexing.chunks import Chunk
from assistant.indexing.docs import chunk_markdown
from assistant.indexing.sources import SOURCE_META_FILE
from assistant.retrieval.embeddings import Embedder, EmbeddingStats
from assistant.retrieval.store import IndexManifest, IndexMissingError, VectorStore

REPO = "https://github.com/example/fork"
DOCS = """\
# Configuration

## OCR settings

### PAPERLESS_OCR_LANGUAGE {#PAPERLESS_OCR_LANGUAGE}

Language used by tesseract for OCR, for example rus or deu.

## Email

### PAPERLESS_EMAIL_HOST {#PAPERLESS_EMAIL_HOST}

SMTP host used to send emails.
"""


def manifest(
    dim: int = 64, ref: str = "v3.4.2", model: str = "fake-model"
) -> IndexManifest:
    return IndexManifest(
        ref=ref,
        commit="abc",
        embedding_model=model,
        dim=dim,
        built_at="2026-11-04T00:00:00+00:00",
        chunks={"doc": 2},
        assistant_version="0.1.0",
    )


def docs_chunks() -> list[Chunk]:
    return chunk_markdown(
        DOCS, path="docs/configuration.md", ref="v3.4.2", repo_url=REPO
    )


def filled(store: VectorStore, embedder: Embedder) -> list[Chunk]:
    chunks = docs_chunks()
    store.recreate(manifest())
    store.upsert(chunks, embedder.embed_documents([c.embedding_text() for c in chunks]))
    return chunks


def test_manifest_is_stored_in_collection_metadata(store: VectorStore) -> None:
    store.recreate(manifest())

    assert store.manifest() == manifest()


def test_missing_index_gives_clear_error(store: VectorStore) -> None:
    with pytest.raises(IndexMissingError, match="make index"):
        store.manifest()


def test_upsert_verify_and_count(store: VectorStore, fake_embedder: Embedder) -> None:
    chunks = filled(store, fake_embedder)

    assert store.count() == len(chunks)
    assert store.verify() == len(chunks)
    assert store.has_source("docs/configuration.md#PAPERLESS_OCR_LANGUAGE")
    assert not store.has_source("docs/configuration.md#nope")


def test_search_returns_full_chunk_with_metadata(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    filled(store, fake_embedder)

    hits = store.search(
        fake_embedder.embed_query("tesseract OCR language rus"), limit=1
    )

    assert hits[0].chunk.source_id == "docs/configuration.md#PAPERLESS_OCR_LANGUAGE"
    assert hits[0].chunk.url.startswith(f"{REPO}/blob/v3.4.2/docs/configuration.md")
    assert 0 < hits[0].score <= 1


def test_search_can_filter_by_kind(store: VectorStore, fake_embedder: Embedder) -> None:
    filled(store, fake_embedder)

    assert store.search(fake_embedder.embed_query("OCR"), limit=5, kinds={"code"}) == []


def test_recreate_replaces_old_points(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    filled(store, fake_embedder)

    store.recreate(manifest(ref="v3.5.0"))

    assert store.count() == 0
    assert store.manifest().ref == "v3.5.0"


def make_tree(base: Path) -> None:
    root = base / "v3.4.2"
    (root / "docs").mkdir(parents=True)
    (root / "src/paperless").mkdir(parents=True)
    (root / "docs/configuration.md").write_text(DOCS)
    (root / "src/paperless/settings.py").write_text(
        'import os\n\nOCR_LANGUAGE = os.getenv("PAPERLESS_OCR_LANGUAGE", "eng")\n'
    )
    (root / SOURCE_META_FILE).write_text(
        json.dumps({"ref": "v3.4.2", "commit": "d939d87"})
    )


def no_network() -> httpx2.Client:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"неожиданный запрос {request.url}")

    return httpx2.Client(transport=httpx2.MockTransport(fail))


def test_build_index_end_to_end(
    tmp_path: Path, store: VectorStore, fake_embedder: Embedder
) -> None:
    make_tree(tmp_path)
    settings = Settings(paperless_source_path=tmp_path, paperless_repo_url=REPO)
    messages: list[str] = []

    report = build_index(
        settings,
        http=no_network(),
        store=store,
        embedder=fake_embedder,
        log=messages.append,
    )

    assert report.manifest.ref == "v3.4.2"
    assert report.manifest.commit == "d939d87"
    assert report.manifest.chunks == {"doc": 2, "code": 1}
    assert store.count() == 3
    assert store.manifest() == report.manifest
    assert any("кусков: 3" in m for m in messages)


class BrokenEmbedder:
    model = "fake-model"

    def __init__(self) -> None:
        self.stats = EmbeddingStats()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("Voyage недоступен")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("Voyage недоступен")


def test_failed_embedding_keeps_old_index(
    tmp_path: Path, store: VectorStore, fake_embedder: Embedder
) -> None:
    filled(store, fake_embedder)
    make_tree(tmp_path)

    with pytest.raises(RuntimeError, match="Voyage"):
        build_index(
            Settings(paperless_source_path=tmp_path, paperless_repo_url=REPO),
            http=no_network(),
            store=store,
            embedder=BrokenEmbedder(),
            log=lambda _: None,
        )

    assert store.count() == len(docs_chunks())
    assert store.manifest().ref == "v3.4.2"


def test_cost_is_estimated_in_model_currency() -> None:
    assert (
        estimate_cost("ai-sage/Giga-Embeddings-instruct-3B", 1_000_000) == "≈32.2200 ₽"
    )
    assert estimate_cost("voyage-4", 500_000) == "≈0.0300 $"
    assert estimate_cost("unknown-model", 10) is None
