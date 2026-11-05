"""Фильтр «с чем и без чего»: условия и их проверка в локальном Qdrant."""

import pytest
from chunk_helpers import CHICKEN, COCKTAIL, COOKIES, RECIPES
from eval_helpers import MANIFEST

from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.store import VectorStore


@pytest.fixture
def indexed(store: VectorStore, fake_embedder: Embedder) -> VectorStore:
    vectors = fake_embedder.embed_documents([c.embedding_text() for c in RECIPES])
    store.recreate(
        MANIFEST.model_copy(
            update={
                "embedding_model": fake_embedder.model,
                "dim": len(vectors[0]),
                "recipes": len(RECIPES),
            }
        )
    )
    store.upsert(RECIPES, vectors)
    return store


def found(
    store: VectorStore,
    embedder: Embedder,
    conditions: IngredientFilter | None,
    question: str = "курица рис",
) -> list[str]:
    hits = store.search(embedder.embed_query(question), limit=5, conditions=conditions)
    return [hit.chunk.name for hit in hits]


def test_without_conditions_all_recipes_are_ranked(
    indexed: VectorStore, fake_embedder: Embedder
) -> None:
    assert found(indexed, fake_embedder, None)[0] == CHICKEN.name
    assert len(found(indexed, fake_embedder, IngredientFilter())) == 3


def test_include_keeps_only_recipes_with_the_ingredient(
    indexed: VectorStore, fake_embedder: Embedder
) -> None:
    """Ближе всего к вопросу курица, но фильтр оставляет только рецепты с сахаром."""
    conditions = IngredientFilter.of(include=[["сахар"]])

    assert set(found(indexed, fake_embedder, conditions)) == {
        COCKTAIL.name,
        COOKIES.name,
    }


def test_group_is_alternatives_and_groups_are_all_required(
    indexed: VectorStore, fake_embedder: Embedder
) -> None:
    either = IngredientFilter.of(include=[["молоко", "курица"]])
    both = IngredientFilter.of(include=[["курица"], ["рис"]])
    impossible = IngredientFilter.of(include=[["курица"], ["сахар"]])

    assert set(found(indexed, fake_embedder, either)) == {COCKTAIL.name, CHICKEN.name}
    assert found(indexed, fake_embedder, both) == [CHICKEN.name]
    assert found(indexed, fake_embedder, impossible) == []


def test_exclude_drops_recipes_with_any_of_the_ingredients(
    indexed: VectorStore, fake_embedder: Embedder
) -> None:
    """«Курица с рисом, но без лука»: единственный рецепт с курицей — с луком."""
    no_onion = IngredientFilter.of(exclude=["лук репчатый"])
    no_milk_no_eggs = IngredientFilter.of(exclude=["молоко", "яйцо куриное"])

    assert CHICKEN.name not in found(indexed, fake_embedder, no_onion)
    assert found(indexed, fake_embedder, no_milk_no_eggs) == [CHICKEN.name]


def test_include_and_exclude_together(
    indexed: VectorStore, fake_embedder: Embedder
) -> None:
    conditions = IngredientFilter.of(include=[["сахар"]], exclude=["молоко"])

    assert found(indexed, fake_embedder, conditions) == [COOKIES.name]


def test_names_are_cleaned_and_empty_groups_are_not_conditions() -> None:
    """Модель на «что-нибудь к чаю» возвращает include `[[]]`: это «без условий»,
    а не «ничего не подходит»."""
    conditions = IngredientFilter.of(
        include=[[], [" Курица ", "курица", ""], ["Рис"], ["рис"]],
        exclude=["Лук репчатый", " "],
    )

    assert conditions == IngredientFilter(
        include=(("курица",), ("рис",)), exclude=("лук репчатый",)
    )
    assert not IngredientFilter.of(include=[[]], exclude=[""])
    assert IngredientFilter.of(include=[[]]).to_qdrant() is None
