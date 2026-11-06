"""Цикл «два судьи → исправление → судьи»: правило «нет у любого», номера, удаление."""

from collections.abc import Mapping, Sequence

from eval_helpers import CHUNK

from assistant.generation.citations import (
    CheckedStatement,
    Citation,
    SourceText,
    Statement,
    check_statement,
)
from assistant.generation.revise import Remark
from evals.judge import Claim, Verdict
from evals.loop import _author_agreement, run_case

RECIPE = "Сырники\nТворог — 250 г\nМука пшеничная — 1 ст. л.\nСахар"
SOURCES = [SourceText(1, CHUNK.model_copy(update={"recipe_id": 7}), RECIPE)]


def statement(text: str, quote: str = "Творог — 250 г") -> CheckedStatement:
    return check_statement(
        Statement(text=text, citations=[Citation(source=1, quote=quote)]), SOURCES
    )


class FakeJudge:
    """Говорит «нет» утверждениям, в тексте которых есть одно из слов `rejects`."""

    def __init__(self, model: str, rejects: Sequence[str]) -> None:
        self.model = model
        self.rejects = rejects
        self.seen: list[list[str]] = []

    def judge(
        self,
        question: str,
        claims: Sequence[Claim],
        facts: Sequence[str],
        *,
        clarity: bool = True,
    ) -> Verdict:
        self.seen.append([c.text for c in claims])
        marks: list[bool | None] = [
            not any(word in c.text for word in self.rejects) for c in claims
        ]
        comments = ["нет в рецепте" if not m else "ок" for m in marks]
        return Verdict(marks, [], comments, [])


class FakeReviser:
    """Убирает слово «яйцо»; «жарить» исправить не может — удаляет."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def revise(
        self,
        question: str,
        sources: Sequence[SourceText],
        statements: Mapping[int, CheckedStatement],
        remarks: Sequence[Remark],
    ) -> dict[int, CheckedStatement | None]:
        self.calls.append([r.number for r in remarks])
        result: dict[int, CheckedStatement | None] = {}
        for remark in remarks:
            text = statements[remark.number].text
            if "жарить" in text:
                result[remark.number] = None
            else:
                result[remark.number] = statement(text.replace(" и яйца", ""))
        return result


def test_one_no_is_enough_and_only_fixed_statements_are_judged_again() -> None:
    strict = FakeJudge("strict", ["яйца", "жарить"])
    lenient = FakeJudge("lenient", [])
    statements = {
        1: statement("Сырники из творога и яйца: творог 250 г."),
        2: statement("Сырники надо жарить: творог 250 г."),
        3: statement("В них есть сахар.", "Сахар"),
    }

    loop = run_case(
        "by_name-004",
        0,
        "Сырники",
        SOURCES,
        statements,
        [strict, lenient],
        FakeReviser(),
    )

    assert loop.flagged_first == {1, 2}
    assert loop.final == {
        1: "Сырники из творога: творог 250 г.",
        2: None,
        3: "В них есть сахар.",
    }
    assert loop.revisions == {1: 1}
    # Второй круг — только исправленное утверждение 1, одобренное 3 больше не судят.
    assert strict.seen[1] == ["Сырники из творога: творог 250 г."]


def test_still_rejected_after_all_cycles_is_removed() -> None:
    stubborn = FakeJudge("stubborn", ["Сырники"])
    reviser = FakeReviser()

    loop = run_case(
        "by_name-004",
        0,
        "Сырники",
        SOURCES,
        {1: statement("Сырники из творога и яйца: творог 250 г.")},
        [stubborn],
        reviser,
        cycles=2,
    )

    assert loop.final == {1: None}
    assert len(loop.rounds) == 3  # первый круг и два круга после исправлений
    assert reviser.calls == [[1], [1]]


def test_clean_answer_has_one_round_and_no_revision() -> None:
    reviser = FakeReviser()

    loop = run_case(
        "x",
        0,
        "?",
        SOURCES,
        {1: statement("В них есть сахар.", "Сахар")},
        [FakeJudge("a", []), FakeJudge("b", [])],
        reviser,
    )

    assert len(loop.rounds) == 1 and reviser.calls == []
    assert loop.final == {1: "В них есть сахар."}


def test_agreement_with_author_counts_both_sides() -> None:
    marks = [
        {"mode": "auto", "id": "a", "run": 0, "statement": 1, "supported": False},
        {"mode": "auto", "id": "a", "run": 0, "statement": 2, "supported": True},
        {"mode": "auto", "id": "b", "run": 1, "statement": 1, "supported": True},
    ]
    verdict: dict[tuple[str, str, int, int], bool | None] = {
        ("auto", "a", 0, 0): False,
        ("auto", "a", 0, 1): False,
        ("auto", "b", 1, 0): None,
    }

    assert _author_agreement(marks, verdict) == ((1, 3), (1, 1), (0, 2))
