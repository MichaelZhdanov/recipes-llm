"""Проверка ответа Sonnet перед отправкой и одна доработка Haiku."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from eval_helpers import CHUNK

from assistant.generation.citations import (
    CheckedStatement,
    Citation,
    SourceText,
    Statement,
    check_statement,
)
from assistant.generation.gigachat import (
    Completion,
    FunctionCallMode,
    LlmStats,
    Message,
    Usage,
)
from assistant.generation.prompt import load_prompt
from assistant.generation.review import PROMPT_VERSION, Reviewer, review_answer
from assistant.generation.revise import Reviser
from evals.judge import JUDGE_PROMPT

RECIPE = "Сырники\nТворог — 250 г\nМука пшеничная — 1 ст. л.\nСахар"
SOURCES = [SourceText(1, CHUNK.model_copy(update={"recipe_id": 7}), RECIPE)]


def statement(text: str, quote: str = "Творог — 250 г") -> CheckedStatement:
    return check_statement(
        Statement(text=text, citations=[Citation(source=1, quote=quote)]), SOURCES
    )


EGG = statement("🧀 Сырники из творога и яйца: творог 250 г.")
SUGAR = statement("🍯 В них есть сахар.", "Сахар")


class Scripted:
    """LLM, которая отвечает заготовками по очереди и запоминает, что видела."""

    def __init__(self, replies: list[dict[str, object]]) -> None:
        self.replies = [json.dumps(r, ensure_ascii=False) for r in replies]
        self.calls: list[list[Message]] = []
        self.stats = LlmStats()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Mapping[str, object] | None = None,
        functions: Sequence[Mapping[str, object]] | None = None,
        function_call: FunctionCallMode | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        usage = Usage(100, 20, 0, 120)
        self.stats.add(usage)
        return Completion(self.replies.pop(0), "stop", model, usage)


def verdicts(*marks: bool) -> dict[str, object]:
    return {
        "statements": [
            {
                "number": n,
                "comment": "нет яйца в рецепте" if not ok else "ок",
                "supported": ok,
            }
            for n, ok in enumerate(marks, start=1)
        ]
    }


def fix(number: int, text: str, quote: str = "Творог — 250 г") -> dict[str, object]:
    return {
        "statements": [
            {
                "number": number,
                "drop": False,
                "text": text,
                "citations": [{"source": 1, "quote": quote}],
            }
        ]
    }


def test_review_rubric_is_the_judge_rubric_without_facts() -> None:
    """Проверка в пути ответа и судья eval судят по одним правилам."""
    judge = (Path("evals/prompts") / f"{JUDGE_PROMPT}.md").read_text(encoding="utf-8")
    facts = judge[
        judge.index("Если дан список фактов") : judge.index("Отвечай только JSON")
    ]

    assert load_prompt(PROMPT_VERSION) == judge.replace(facts, "")


def test_all_yes_goes_to_the_user_without_revision() -> None:
    sonnet = Scripted([verdicts(True, True, True)])
    haiku = Scripted([])

    review = review_answer(
        "Сырники",
        SOURCES,
        [EGG, SUGAR],
        "🧀 Вот сырники:",
        Reviewer(sonnet),
        Reviser(haiku, model="h"),
    )

    assert [s.text for s in review.final] == [EGG.text, SUGAR.text]
    assert review.intro_kept and review.flagged == [] and haiku.calls == []
    assert 'kind="intro"' in sonnet.calls[0][1]["content"]


def test_no_is_revised_once_with_the_remark_and_checked_again() -> None:
    sonnet = Scripted([verdicts(False, True), verdicts(True)])
    haiku = Scripted([fix(1, "🧀 Сырники из творога: творог 250 г.")])

    review = review_answer(
        "Сырники",
        SOURCES,
        [EGG, SUGAR],
        "",
        Reviewer(sonnet),
        Reviser(haiku, model="h"),
    )

    assert [s.text for s in review.final] == [
        "🧀 Сырники из творога: творог 250 г.",
        SUGAR.text,
    ]
    assert "нет яйца в рецепте" in haiku.calls[0][1]["content"]
    # Повторно Sonnet видит только исправленное утверждение.
    assert SUGAR.text not in sonnet.calls[1][1]["content"]
    assert len(sonnet.calls) == 2


def test_still_no_after_revision_is_removed_and_no_second_round() -> None:
    sonnet = Scripted([verdicts(False, True), verdicts(False)])
    haiku = Scripted([fix(1, "🧀 Сырники с яйцом: творог 250 г.")])

    review = review_answer(
        "Сырники",
        SOURCES,
        [EGG, SUGAR],
        "",
        Reviewer(sonnet),
        Reviser(haiku, model="h"),
    )

    assert [s.text for s in review.final] == [SUGAR.text]
    assert len(haiku.calls) == 1 and len(sonnet.calls) == 2


def test_intro_with_no_is_dropped_without_revision() -> None:
    sonnet = Scripted([verdicts(True, False)])
    haiku = Scripted([])

    review = review_answer(
        "Сырники",
        SOURCES,
        [SUGAR],
        "🧀 Сырники без сахара:",
        Reviewer(sonnet),
        Reviser(haiku, model="h"),
    )

    assert not review.intro_kept and [s.text for s in review.final] == [SUGAR.text]
    assert haiku.calls == []


def test_unparsable_review_means_unverified_and_revision() -> None:
    sonnet = Scripted([{"не": "та схема"}, verdicts(True)])
    haiku = Scripted([fix(1, "🍯 В них есть сахар.", "Сахар")])
    reviewer = Reviewer(sonnet)

    review = review_answer(
        "Сырники", SOURCES, [SUGAR], "", reviewer, Reviser(haiku, model="h")
    )

    assert reviewer.failures == 1
    assert review.flagged == [1] and [s.text for s in review.final] == [SUGAR.text]
    assert (
        review.review_usage.prompt_tokens == 200
        and review.revise_usage.prompt_tokens == 100
    )
