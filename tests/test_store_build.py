from collections.abc import Sequence

import pytest
from chunk_helpers import CHICKEN, RECIPES, ROWS
from qdrant_client import QdrantClient

from assistant.indexing.build import build_index, estimate_cost
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import Embedder, EmbeddingStats
from assistant.retrieval.store import IndexManifest, IndexMissingError, VectorStore


def manifest(
    dim: int = 64, ref: str = "v3.4.2", model: str = "fake-model"
) -> IndexManifest:
    return IndexManifest(
        ref=ref,
        commit="abc",
        embedding_model=model,
        dim=dim,
        built_at="2026-11-04T00:00:00+00:00",
        chunks={"recipe": 3},
        assistant_version="0.1.0",
    )


def filled(store: VectorStore, embedder: Embedder) -> list[Chunk]:
    store.recreate(manifest())
    store.upsert(
        RECIPES, embedder.embed_documents([c.embedding_text() for c in RECIPES])
    )
    return RECIPES


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


def test_search_returns_full_recipe(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    filled(store, fake_embedder)

    [hit] = store.search(fake_embedder.embed_query("курица рис лук"), limit=1)

    assert hit.chunk == CHICKEN
    assert hit.chunk.ingredients == ["курица", "рис", "лук репчатый", "соль"]
    assert 0 < hit.score <= 1


def test_recreate_replaces_old_points(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    filled(store, fake_embedder)

    store.recreate(manifest(ref="v3.5.0"))

    assert store.count() == 0
    assert store.manifest().ref == "v3.5.0"


def test_build_index_end_to_end(store: VectorStore, fake_embedder: Embedder) -> None:
    messages: list[str] = []

    report = build_index(
        ROWS,
        ref="v3.4.2",
        commit=None,
        store=store,
        embedder=fake_embedder,
        log=messages.append,
    )

    assert report.manifest.ref == "v3.4.2"
    assert report.manifest.chunks == {"recipe": 3}
    assert report.manifest.quantization == "int8"
    assert store.count() == 3
    assert store.manifest() == report.manifest
    assert store.verify() == 3
    assert any("рецептов: 3" in m for m in messages)


def test_nothing_to_index_is_an_error(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    with pytest.raises(ValueError, match="нет рецептов"):
        build_index(
            [], ref="v", commit=None, store=store, embedder=fake_embedder, log=print
        )


class BrokenEmbedder:
    model = "fake-model"

    def __init__(self) -> None:
        self.stats = EmbeddingStats()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("API эмбеддингов недоступен")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("API эмбеддингов недоступен")


def test_failed_embedding_keeps_old_index(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    filled(store, fake_embedder)

    with pytest.raises(RuntimeError, match="эмбеддингов"):
        build_index(
            ROWS,
            ref="v3.5.0",
            commit=None,
            store=store,
            embedder=BrokenEmbedder(),
            log=lambda _: None,
        )

    assert store.count() == len(RECIPES)
    assert store.manifest().ref == "v3.4.2"


def test_cost_is_estimated_in_model_currency() -> None:
    assert (
        estimate_cost("ai-sage/Giga-Embeddings-instruct-3B", 1_000_000) == "≈32.2200 ₽"
    )
    assert estimate_cost("voyage-4", 500_000) == "≈0.0300 $"
    assert estimate_cost("unknown-model", 10) is None


def collections(client: QdrantClient) -> list[str]:
    return sorted(c.name for c in client.get_collections().collections)


def test_rebuild_switches_alias_and_removes_old_collection(
    fake_embedder: Embedder,
) -> None:
    client = QdrantClient(":memory:")
    store = VectorStore(client)

    def build(ref: str) -> None:
        build_index(
            ROWS, ref=ref, commit=None, store=store, embedder=fake_embedder, log=print
        )

    build("v1")
    first = store.active()
    build("v2")

    assert store.active() != first
    assert collections(client) == [store.active()]
    assert store.manifest().ref == "v2"


class FailsOnSecondBatch:
    """Первая пачка проходит, вторая падает: коллекция уже создана и наполовину заполнена."""

    model = "fake-model"

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.stats = EmbeddingStats()
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("API эмбеддингов упал посередине")
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


def test_failure_in_the_middle_drops_the_unfinished_collection(
    fake_embedder: Embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("assistant.indexing.build.BATCH", 2)
    client = QdrantClient(":memory:")
    store = VectorStore(client)
    build_index(
        ROWS, ref="v1", commit=None, store=store, embedder=fake_embedder, log=print
    )
    old = store.active()

    with pytest.raises(RuntimeError, match="посередине"):
        build_index(
            ROWS,
            ref="v2",
            commit=None,
            store=store,
            embedder=FailsOnSecondBatch(fake_embedder),
            log=print,
        )

    assert collections(client) == [old]
    assert store.manifest().ref == "v1"
    assert store.count() == 3


def test_vectors_are_on_disk_and_int8_can_be_turned_off(
    fake_embedder: Embedder,
) -> None:
    client = QdrantClient(":memory:")
    store = VectorStore(client)

    report = build_index(
        ROWS,
        ref="v1",
        commit=None,
        store=store,
        embedder=fake_embedder,
        log=print,
        quantization=False,
    )

    params = client.get_collection(store.active() or "").config.params
    assert params.vectors.on_disk is True  # type: ignore[union-attr]
    assert report.manifest.quantization is None
