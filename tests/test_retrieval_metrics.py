from itertools import count

from chunk_helpers import COOKIES, RECIPES, ROWS
from eval_helpers import MANIFEST, case, found, result

from assistant.generation.gigachat import LlmError
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import Hit, VectorStore
from assistant.tools.search_recipes import SearchArgs
from evals.dataset import Case
from evals.retrieval_metrics import (
    CaseResult,
    RetrievalRun,
    conditions,
    conditions_score,
    dropped_names,
    errors,
    exclude_violated,
    exclude_violations,
    fits,
    has_conditions,
    history,
    include_matches,
    include_precision,
    mark_available,
    name_hit,
    name_matches,
    primary,
    run_cases,
    run_score,
    unknown_ingredients,
    vector_search,
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
        vector_search(retriever),
        runs=3,
        clock=lambda: float(next(ticks)),
    )

    assert result_.latencies == [1.0, 1.0, 1.0]
    assert result_.runs[0].top[0].recipe_id == COOKIES.recipe_id
    assert result_.runs[0].top[0].ingredients == tuple(COOKIES.ingredients)


def test_conditions_score_checks_every_recipe_of_the_top() -> None:
    """«Курица без лука»: курица с рисом и луком не подходит, плов с филе — нет курицы."""
    kase = case(
        "exclude-001",
        "exclude",
        expected_include=[["курица"]],
        expected_exclude=["лук репчатый"],
    )
    no_onion = found("Курица в сметане", ("курица", "сметана"))

    assert (
        conditions_score(kase, RetrievalRun([no_onion, CHICKEN_RICE, FILLET_RICE], 0.1))
        == 1 / 3
    )
    assert conditions_score(kase, RetrievalRun([], 0.1)) == 0.0
    # У by_name и semantic условий для conditions@5 нет.
    assert (
        conditions_score(case("by_name-001"), RetrievalRun([CHICKEN_RICE], 0.1)) is None
    )


def test_follow_up_with_conditions_counts_and_without_does_not() -> None:
    with_onion = case(
        "follow_up-001",
        "follow_up",
        expected_exclude=["лук репчатый"],
        question="а без лука?",
    )
    quantity = case("follow_up-002", "follow_up", must_include=["200 г"])
    results = [
        result(with_onion, [[CHICKEN_RICE, FILLET_RICE]]),
        result(quantity, [[CHICKEN_RICE]]),
    ]

    assert has_conditions(with_onion) and not has_conditions(quantity)
    assert run_score(with_onion, results[0].runs[0]) == 0.5
    assert conditions(results) == 0.5


def test_conditions_skip_cases_without_answer_in_the_index() -> None:
    kase = case("include-001", "include", expected_include=[["рис"]])
    results = [
        result(kase, [[CHICKEN_RICE]]),
        result(
            case("include-002", "include", expected_include=[["кенгуру"]]),
            [[PANCAKES]],
            False,
        ),
    ]

    assert conditions(results) == 1.0


def test_parse_failure_is_an_empty_run_not_a_crash() -> None:
    def search(kase: Case) -> tuple[SearchArgs | None, list[Hit]]:
        raise LlmError("GigaChat не вызвал search_recipes (finish_reason: blacklist)")

    [result_] = run_cases([case("include-001", "include")], search, runs=2)

    assert [run.top for run in result_.runs] == [[], []]
    assert errors([result_]) == 2
    assert conditions([result_]) == 0.0


def test_dropped_names_are_collected_once_per_case() -> None:
    args = SearchArgs("плов", IngredientFilter(), dropped=("рис басмати",))
    runs = [RetrievalRun([], 0.1, args=args), RetrievalRun([], 0.1, args=args)]

    results = [CaseResult(case("include-001", "include"), runs)]

    assert dropped_names(results) == [("include-001", "рис басмати")]


def test_history_turns_become_messages() -> None:
    kase = case(
        "follow_up-001",
        "follow_up",
        expected_exclude=["лук репчатый"],
        turns=[
            {"role": "user", "text": "что из курицы?"},
            {"role": "assistant", "text": "Курица с рисом"},
        ],
    )

    assert history(kase) == [
        {"role": "user", "content": "что из курицы?"},
        {"role": "assistant", "content": "Курица с рисом"},
    ]
