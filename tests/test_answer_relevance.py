"""Рецепты итогового ответа по вопросу: какие кейсы судятся и как считается."""

from collections.abc import Sequence

from evals.answer_metrics import Rate
from evals.answer_relevance import Answered, answered, in_scope, report, score
from evals.dataset import Case
from evals.retrieval_metrics import Found


def case(case_id: str, category: str, **fields: object) -> Case:
    return Case.model_validate(
        {
            "id": case_id,
            "category": category,
            "question": "что подать к красному вину?",
            "expected_behavior": "answer",
            "turns": [{"role": "user", "text": "привет"}]
            if category == "follow_up"
            else [],
            **fields,
        }
    )


def record(
    case_id: str, status: str, recipe_ids: list[int], first_try: bool = False
) -> dict[str, object]:
    chunks = [
        {
            "recipe_id": 1,
            "name": "Антрекот в красном вине",
            "ingredients": ["говядина", "вино красное"],
        },
        {"recipe_id": 2, "name": "Сырная тарелка", "ingredients": ["сыр", "виноград"]},
    ]
    return {
        "id": case_id,
        "run": 0,
        "answer": {
            "status": status,
            "sources": [{"recipe_id": i} for i in recipe_ids],
        },
        "sources": [{"number": n, "chunk": c} for n, c in enumerate(chunks, start=1)],
        "first_try": {"query": "…"} if first_try else None,
    }


def test_only_meaning_questions_and_follow_ups_without_conditions() -> None:
    assert in_scope(case("semantic-011", "semantic"))
    assert in_scope(case("follow_up-009", "follow_up"))
    assert not in_scope(
        case("follow_up-001", "follow_up", expected_exclude=["лук репчатый"])
    )
    assert not in_scope(case("include-001", "include", expected_include=[["курица"]]))
    # «Сколько муки во втором?» — про рецепт прошлого ответа.
    assert not in_scope(case("follow_up-003", "follow_up", must_include=["200 г"]))


class FakeJudge:
    """Подходит только «Сырная тарелка»."""

    def judge(self, question: str, recipes: Sequence[Found]) -> list[bool | None]:
        return [r.name == "Сырная тарелка" for r in recipes]


def test_answer_recipes_are_judged_and_not_found_has_none() -> None:
    cases = {
        "semantic-011": case("semantic-011", "semantic"),
        "semantic-012": case("semantic-012", "semantic"),
        "semantic-013": case("semantic-013", "semantic"),
    }
    records = [
        record("semantic-011", "answer", [1, 2]),
        record("semantic-012", "answer", [2], first_try=True),
        record("semantic-013", "not_found", [1]),
        {**record("semantic-011", "answer", [1]), "run": 1},  # не первый прогон
    ]

    runs = answered(records, cases)
    scored = score("после", runs, FakeJudge())  # type: ignore[arg-type]
    text = report([scored], (Rate(3, 3), Rate(3, 3)), "судья")

    assert [a.case_id for a in runs] == ["semantic-011", "semantic-012", "semantic-013"]
    assert runs[2].recipes == [] and runs[1].new_search
    assert str(scored.relevant()).startswith("0.67 (2/3")
    assert str(scored.all_relevant()).startswith("0.50 (1/2")
    assert "| «не нашёл» | 1 из 3 |" in text
    assert "| ответов с новым поиском | 1 |" in text
    assert "❌ Антрекот в красном вине, ✅ Сырная тарелка" in text
    assert isinstance(runs[0], Answered)
