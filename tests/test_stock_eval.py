"""Замер подбора по складу: запись кейса, задание судье без пометок, отчёт."""

from chunk_helpers import make_chunk

from assistant.generation.answer import Answer, AnswerResult, Trace
from assistant.retrieval.store import Hit
from evals.dataset import Case
from evals.stock_eval import record, render, task

OMELET = make_chunk("Омлет", {"Яйцо куриное": "3 шт", "Соль": None}, recipe_id=1)
PIE = make_chunk(
    "Пирог", {"Яйцо куриное": "1 шт", "Мука пшеничная": "200 г"}, recipe_id=2
)


def result() -> AnswerResult:
    answer = Answer(
        status="not_found",
        reason="no_sources",
        statements=[],
        sources=[],
        model="m",
        prompt_version="recipes_v5",
        dataset_revision="ffaf49ee",
        near_available=True,
    )
    trace = Trace(
        [Hit(OMELET, 0.9)],
        [],
        None,
        [],
        0.0,
        0.0,
        stock_near=((PIE, ("мука пшеничная",)),),
    )
    return AnswerResult(answer, trace)


def test_record_checks_stock_purity_and_conditions() -> None:
    case = Case.model_validate(
        {
            "id": "stock-099",
            "category": "stock",
            "question": "что-нибудь с яйцами",
            "expected_behavior": "answer",
            "expected_include": [["яйцо куриное"]],
        }
    )

    row = record(case, result(), 3.0)

    assert row["strict_pure"] and row["strict_fits"] == [True]
    assert row["near"] == [
        {
            "recipe_id": 2,
            "name": "Пирог",
            "ingredients": PIE.ingredients,
            "missing": ["мука пшеничная"],
        }
    ]
    # Судья не видит, какой рецепт строгий, а какой — «не хватает»: судит по вопросу.
    (item,) = task([row])
    assert [r["name"] for r in item["recipes"]] == ["Омлет", "Пирог"]
    assert all("missing" not in r for r in item["recipes"])
    verdicts = {
        "stock-099": {
            "recipes": [{"n": 1, "relevant": True}, {"n": 2, "relevant": False}]
        }
    }
    text = render([row], {"date": "d", "model": "m", "cost": "—"}, verdicts)
    assert "| нашлось целиком из склада | 1.00 (1/1" in text
    assert "| строгие рецепты по вопросу (судья) | 1.00 (1/1" in text
    assert "| расширенные рецепты по вопросу (судья) | 0.00 (0/1" in text
