import pytest

from assistant.cli import format_hits
from assistant.indexing.chunks import Chunk
from assistant.indexing.docs import chunk_markdown
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.search import Retriever, dedupe
from assistant.retrieval.store import (
    Hit,
    IndexManifest,
    IndexMismatchError,
    VectorStore,
)

REPO = "https://github.com/example/fork"
LONG = "\n\n".join(f"OCR language paragraph {i} " + "tesseract " * 30 for i in range(6))
DOCS = f"""\
# Configuration

## OCR

{LONG}

## Mail

Fetching mail from IMAP servers.

## Time zone

Set the time zone of the server.
"""


def build(store: VectorStore, embedder: Embedder, ref: str = "v3.4.2") -> list[Chunk]:
    chunks = chunk_markdown(
        DOCS, path="docs/configuration.md", ref=ref, repo_url=REPO, max_chars=300
    )
    vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
    store.recreate(
        IndexManifest(
            ref=ref,
            commit=None,
            embedding_model=embedder.model,
            dim=len(vectors[0]),
            built_at="2026-11-04T00:00:00+00:00",
            chunks={"doc": len(chunks)},
            assistant_version="0.1.0",
        )
    )
    store.upsert(chunks, vectors)
    return chunks


def test_long_section_parts_do_not_crowd_the_top(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    chunks = build(store, fake_embedder)
    assert sum(c.source_id.endswith("#ocr") for c in chunks) > 3

    hits = Retriever(store, fake_embedder, ref="v3.4.2").search(
        "OCR tesseract language", k=3
    )

    sources = [h.chunk.source_id for h in hits]
    assert sources[0] == "docs/configuration.md#ocr"
    assert len(sources) == len(set(sources)) == 3


def test_index_for_other_ref_is_refused(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    build(store, fake_embedder, ref="v3.4.1")

    with pytest.raises(IndexMismatchError, match=r"v3\.4\.1"):
        Retriever(store, fake_embedder, ref="v3.4.2")


def test_index_for_other_model_is_refused(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    build(store, fake_embedder)
    fake_embedder.model = "voyage-4-large"

    with pytest.raises(IndexMismatchError, match="make index"):
        Retriever(store, fake_embedder, ref="v3.4.2")


def test_dedupe_keeps_best_hit_per_source() -> None:
    def hit(source: str, score: float) -> Hit:
        chunk = chunk_markdown(
            f"## {source}\n\ntext\n", path="docs/a.md", ref="v", repo_url=REPO
        )[0]
        return Hit(chunk=chunk, score=score)

    hits = [hit("a", 0.9), hit("b", 0.8), hit("a", 0.7), hit("c", 0.6)]

    assert [(h.chunk.source_id, h.score) for h in dedupe(hits)] == [
        ("docs/a.md#a", 0.9),
        ("docs/a.md#b", 0.8),
        ("docs/a.md#c", 0.6),
    ]


def test_format_hits_shows_location_links_and_preview(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    build(store, fake_embedder)
    hits = Retriever(store, fake_embedder, ref="v3.4.2").search("time zone server", k=1)

    text = format_hits(hits)

    assert "doc: Configuration > Time zone" in text
    assert "docs/configuration.md:" in text
    assert f"{REPO}/blob/v3.4.2/docs/configuration.md?plain=1#L" in text
    assert "сайт: https://docs.paperless-ngx.com/configuration/#time-zone" in text
    assert "«## Time zone Set the time zone of the server.»" in text


def test_format_hits_when_nothing_found() -> None:
    assert format_hits([]) == "ничего не найдено"
