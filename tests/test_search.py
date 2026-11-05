import pytest
from chunk_helpers import CHICKEN, COCKTAIL, RECIPES
from eval_helpers import MANIFEST

from assistant.cli import format_hits
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import IndexMismatchError, VectorStore


def build(store: VectorStore, embedder: Embedder, ref: str = "v3.4.2") -> list[Chunk]:
    chunks = [c.model_copy(update={"ref": ref}) for c in RECIPES]
    vectors = embedder.embed_documents([c.embedding_text() for c in chunks])
    store.recreate(
        MANIFEST.model_copy(
            update={
                "ref": ref,
                "embedding_model": embedder.model,
                "dim": len(vectors[0]),
                "recipes": len(chunks),
            }
        )
    )
    store.upsert(chunks, vectors)
    return chunks


def test_search_finds_recipe_by_name_and_ingredients(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    build(store, fake_embedder)

    hits = Retriever(store, fake_embedder, ref="v3.4.2").search("курица рис", k=2)

    assert hits[0].chunk == CHICKEN
    # Один рецепт — одна точка: в топе разные рецепты.
    assert len({h.chunk.recipe_id for h in hits}) == 2


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


def test_format_hits_shows_name_link_and_ingredients(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    build(store, fake_embedder)
    hits = Retriever(store, fake_embedder, ref="v3.4.2").search(
        "молочно-клубничный коктейль", k=1
    )

    text = format_hits(hits)

    assert text.startswith("1. [")
    assert "] Густой молочно-клубничный коктейль\n" in text
    assert f"   {COCKTAIL.url}\n" in text
    assert text.endswith("   ингредиенты: молоко, клубника, сахар")


def test_format_hits_when_nothing_found() -> None:
    assert format_hits([]) == "ничего не найдено"
