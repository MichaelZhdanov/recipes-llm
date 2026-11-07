"""Замер передачи человеку: F1 как в боте (фраза или модель) и только модель."""

from collections.abc import Sequence
from typing import Any

from assistant.generation.gigachat import Message
from assistant.retrieval.filters import IngredientFilter
from assistant.tools.search_recipes import SearchArgs
from evals.dataset import Case
from evals.escalation_eval import Outcome, predict, render, score


def case(question: str, reason: str | None, **fields: Any) -> Case:
    return Case.model_validate(
        {
            "id": "escalation-099",
            "category": "escalation",
            "question": question,
            "expected_behavior": "escalate" if reason else "answer",
            "escalation_reason": reason,
            **fields,
        }
    )


class Planner:
    def __init__(self, args: SearchArgs) -> None:
        self.args = args
        self.histories: list[list[Message]] = []

    def plan(
        self, question: str, history: Sequence[Message] = (), note: str = ""
    ) -> SearchArgs:
        self.histories.append(list(history))
        return self.args


def test_phrase_counts_as_a_request_even_if_the_model_missed_it() -> None:
    plain = SearchArgs("человек", IngredientFilter())
    turns = [{"role": "user", "text": "блины"}, {"role": "assistant", "text": "Вот…"}]
    planner = Planner(plain)

    outcome = predict(case("позовите оператора", "request", turns=turns), planner)  # type: ignore[arg-type]

    assert outcome.phrase and outcome.model is None and outcome.system == "request"
    assert planner.histories[0][0] == {"role": "user", "content": "блины"}


def test_allergy_case_checks_flag_and_excluded_names() -> None:
    args = SearchArgs(
        "торт без орехов",
        IngredientFilter.of(exclude=["орехи грецкие", "миндаль"]),
        allergy=True,
    )
    allergic = case("аллергия на орехи", None, expected_exclude=["орехи грецкие"])

    outcome = predict(allergic, Planner(args))  # type: ignore[arg-type]

    assert outcome.system is None and outcome.allergy and outcome.exclude_ok is True


def test_score_counts_both_ways_and_reason_accuracy() -> None:
    outcomes = [
        Outcome("a", "", "health", "health", False),
        Outcome("b", "", "health", None, False),
        Outcome("c", "", "request", None, True),  # фразой — как в боте поймано
        Outcome("d", "", None, "cooking_failed", False),  # зря
        Outcome("e", "", None, None, False),
        Outcome("f", "", "data_issue", "unclear_recipe", False),  # повод не тот
    ]

    bot = score(outcomes)
    model = score(outcomes, model_only=True)

    assert (bot.tp, bot.fp, bot.fn, bot.tn) == (3, 1, 1, 1)
    assert bot.reason_right == 2  # health и request
    assert (bot.health.k, bot.health.n) == (1, 2)
    assert (model.tp, model.fn) == (2, 2)
    assert round(bot.f1, 2) == 0.75
    text = render(
        outcomes,
        {
            "date": "2026-11-07",
            "model": "m",
            "effort": "low",
            "median": 1.0,
            "cost": "—",
        },
    )
    assert "порог не пройден" in text and "`b` ❌" in text


def test_allergy_flag_is_expected_only_where_allergy_is_named() -> None:
    named = Outcome("a", "у сына алергия на мёд", None, None, False, allergy=True)
    diet = Outcome("b", "сижу на диете, без майонеза", None, None, False, allergy=True)

    text = render(
        [named, diet],
        {"date": "d", "model": "m", "effort": "", "median": 0.0, "cost": "—"},
    )

    assert named.allergy_named and not diet.allergy_named
    assert "где аллергия названа: 1.00 (1/1" in text
    assert "аллергии в вопросе нет): 1" in text
