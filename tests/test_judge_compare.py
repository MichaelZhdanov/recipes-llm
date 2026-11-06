"""Сравнение судей на разметке автора: какие ответы судить и как считать совпадение."""

from evals.judge_compare import MARKINGS, agreement, answers_to_judge

V7 = "2026-11-06-judge-v7-manual.yaml"
ALWAYS_V2 = MARKINGS[V7]["always"]


def mark(case_id: str, statement: int, supported: bool) -> dict[str, object]:
    return {
        "mode": "always",
        "id": case_id,
        "run": 0,
        "statement": statement,
        "supported": supported,
    }


def test_each_answer_is_judged_once_even_with_several_marks() -> None:
    marks = {V7: [mark("a", 1, True), mark("a", 2, False), mark("b", 1, True)]}

    assert answers_to_judge(marks) == [(ALWAYS_V2, "a", 0), (ALWAYS_V2, "b", 0)]


def test_agreement_counts_both_sides_and_missing_verdicts() -> None:
    marks = [mark("a", 1, True), mark("a", 2, False), mark("b", 1, False)]
    verdicts: dict[tuple[str, str, int, int], bool | None] = {
        (ALWAYS_V2, "a", 0, 0): True,
        (ALWAYS_V2, "a", 0, 1): True,
        (ALWAYS_V2, "b", 0, 0): None,
    }

    # совпало 1 из 3; автор «нет» — 2, судья поймал 0; автор «да» — судья тоже 1
    assert agreement(marks, MARKINGS[V7], verdicts) == (1, 3, 0, 2, 1)
