from itertools import count

from chunk_helpers import COOKIES, RECIPES, ROWS
from eval_helpers import MANIFEST, case, found, result

from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from evals.retrieval_metrics import (
    exclude_violated,
    exclude_violations,
    fits,
    include_matches,
    include_precision,
    mark_available,
    name_hit,
    name_matches,
    primary,
    run_cases,
    run_score,
    unknown_ingredients,
)

CHICKEN_RICE = found("Курица с рисом", ("курица", "рис", "лук репчатый"))
FILLET_RICE = found("Плов", ("филе куриное", "рис"))
PANCAKES = found("Блины на молоке", ("молоко", "мука пшеничная"))


def test_name_matches_any_word_ignoring_case_and_yo() -> None:
    assert name_matches("Тёплый салат", ["теплый"])
    assert name_matches("БОРЩ украинский", ["щи", "борщ"])
    assert not name_matches("Окрошка", ["борщ"])


def test_include_needs_one_name_from_every_group() -> None:
    groups = [["курица", "филе куриное"], ["рис"]]

    assert include_matches(CHICKEN_RICE.ingredients, groups)
    assert include_matches(FILLET_RICE.ingredients, groups)
    assert not include_matches(PANCAKES.ingredients, groups)
    # Имя целиком: «рис» не находится внутри «рисовая мука».
    assert not include_matches(("курица", "рисовая мука"), groups)


def test_exclude_and_fits() -> None:
    no_onion = case("exclude-001", "exclude", expected_include=[["рис"]])

    assert exclude_violated(CHICKEN_RICE.ingredients, ["лук репчатый"])
    assert not fits(no_onion, CHICKEN_RICE.name, CHICKEN_RICE.ingredients)
    assert fits(no_onion, FILLET_RICE.name, FILLET_RICE.ingredients)


def test_scores_per_category() -> None:
    by_name = case("by_name-001", expected_name=["плов"])
    include = case("include-001", "include", expected_include=[["рис"]])
    exclude = case("exclude-001", "exclude")
    top = [CHICKEN_RICE, FILLET_RICE, PANCAKES, PANCAKES]

    results = [result(by_name, [top]), result(include, [top]), result(exclude, [top])]

    assert run_score(by_name, results[0].runs[0]) == 1.0
    assert run_score(include, results[1].runs[0]) == 0.5
    assert run_score(exclude, results[2].runs[0]) == 0.25
    assert name_hit(results) == 1.0
    assert include_precision(results) == 0.5
    assert exclude_violations(results) == 0.25
    assert primary(results) == 0.75


def test_case_without_any_matching_recipe_is_left_out() -> None:
    hit = result(case("by_name-001", expected_name=["курица"]), [[CHICKEN_RICE]])
    impossible = result(
        case("by_name-002", expected_name=["кенгуру"]),
        [[CHICKEN_RICE]],
        available=False,
    )

    assert name_hit([hit, impossible]) == 1.0
    assert primary([impossible]) is None


def test_mark_available_checks_indexed_recipes() -> None:
    results = [
        result(case("by_name-001", expected_name=["печенье"]), [[]], available=None),
        result(case("by_name-002", expected_name=["кенгуру"]), [[]], available=None),
        result(case("semantic-001", "semantic"), [[]], available=None),
    ]

    marked = mark_available(results, ROWS)

    assert [r.available for r in marked] == [True, False, None]


def test_unknown_ingredients_are_reported() -> None:
    cases = [
        case("include-001", "include", expected_include=[["Курица", "курятина"]]),
        case("exclude-001", "exclude", expected_exclude=["лук репчатый"]),
    ]

    assert unknown_ingredients(cases, {"курица", "Лук репчатый"}) == [
        "include-001: курятина"
    ]


def test_run_cases_measures_top_and_latency(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    store.recreate(MANIFEST)
    store.upsert(
        RECIPES, fake_embedder.embed_documents([c.embedding_text() for c in RECIPES])
    )
    retriever = Retriever(store, fake_embedder, ref="v3.4.2")
    ticks = count()

    [result_] = run_cases(
        [case("by_name-001", question="овсяное печенье", expected_name=["печенье"])],
        retriever,
        runs=3,
        clock=lambda: float(next(ticks)),
    )

    assert result_.latencies == [1.0, 1.0, 1.0]
    assert result_.runs[0].top[0].recipe_id == COOKIES.recipe_id
    assert result_.runs[0].top[0].ingredients == tuple(COOKIES.ingredients)
