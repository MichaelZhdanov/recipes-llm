"""Замер «что есть дома»: один разбор на кейс для обеих версий ответа, условия
кейса у показанных рецептов, задание судье без версии, отчёт v5 против v6."""

from collections.abc import Sequence

from chunk_helpers import CHICKEN, COOKIES

from assistant.generation.answer import Answer, AnswerResult, SourceLink, Trace
from assistant.generation.citations import SourceText
from assistant.generation.gigachat import Message
from assistant.retrieval.filters import IngredientFilter
from assistant.tools.search_recipes import SearchArgs
from evals.dataset import Case
from evals.pantry_eval import CachingPlanner, record, render, task

CASE = Case.model_validate(
    {
        "id": "pantry-099",
        "category": "pantry",
        "question": "есть курица, яйца и рис — что на ужин?",
        "expected_behavior": "answer",
        "expected_include": [["курица", "яйцо куриное", "рис"]],
    }
)


class Planner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(
        self, question: str, history: Sequence[Message] = (), note: str = ""
    ) -> SearchArgs:
        self.calls += 1
        return SearchArgs(question, IngredientFilter())


def result(status: str = "answer") -> AnswerResult:
    sources = [
        SourceText(1, CHICKEN, CHICKEN.text),
        SourceText(2, COOKIES, COOKIES.text),
    ]
    answer = Answer(
        status=status,
        reason=None if status == "answer" else "model_not_found",
        message="🍗 Вот идеи:" if status == "answer" else "Не подходит ни один целиком",
        statements=[],
        sources=[
            SourceLink(
                number=1, recipe_id=CHICKEN.recipe_id, name="К", url=CHICKEN.url
            ),
            SourceLink(
                number=2, recipe_id=COOKIES.recipe_id, name="П", url=COOKIES.url
            ),
        ]
        if status == "answer"
        else [],
        model="m",
        prompt_version="recipes_v6",
        dataset_revision="ffaf49ee",
    )
    return AnswerResult(answer, Trace([], sources, None, [], 0.0, 0.0))


def test_both_prompt_versions_share_one_parse_of_the_question() -> None:
    inner = Planner()
    planner = CachingPlanner(inner)  # type: ignore[arg-type]

    first = planner.plan("вопрос", [])
    second = planner.plan("вопрос", [])
    planner.plan("другой", [])

    assert first is second and inner.calls == 2


def test_record_checks_conditions_of_shown_recipes() -> None:
    row = record(CASE, result(), 1.0)

    # Курица с рисом — продукт из перечня; овсяное печенье с яйцом — тоже.
    assert row["fits"] == [True, True]
    assert [r["recipe_id"] for r in row["shown"]] == [
        CHICKEN.recipe_id,
        COOKIES.recipe_id,
    ]
    assert record(CASE, result("not_found"), 1.0)["shown"] == []


def test_task_hides_the_version_and_skips_refusals() -> None:
    rows = [
        {
            "id": CASE.id,
            "question": CASE.question,
            "dialog": [],
            "error": "",
            "recipes_v5": record(CASE, result("not_found"), 1.0),
            "recipes_v6": record(CASE, result(), 1.0),
        }
    ]

    items = task(rows)
    report = render(rows, {"date": "2026-11-07", "model": "m", "cost": "≈ 1 ₽"}, {})

    assert [i["item"] for i in items] == ["pantry-099:1"]
    assert "recipes_v" not in str(items)
    assert "| ответ, а не «не нашёл» | 0.00 (0/1" in report
    assert "| 1.00 (1/1" in report
    assert "Не подходит ни один целиком" in report
