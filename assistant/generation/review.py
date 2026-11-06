"""Проверка ответа перед отправкой пользователю (решение автора 2026-11-06).

Haiku ответил, код проверил цитаты. Дальше Sonnet по той же рубрике, что
судья eval (`review_v1` — это `evals/prompts/judge_v8.md` без пункта о фактах
must_include; тест сверяет, чтобы они не разошлись), решает, подтверждают ли
рецепты каждое утверждение и вступление:

1. Всё «да» — ответ уходит пользователю как есть.
2. Утверждения с «нет» — один раз Haiku на доработку с конкретными
   замечаниями Sonnet (`revise.py`); исправленные — снова Sonnet.
3. Что и после этого «нет» — удаляется. Вступление с «нет» убирается без
   доработки. Не осталось ни одного утверждения — «не нашёл».

Второго круга нет: цикл не бесконечный, задержка предсказуемая (решение
автора). В eval-эксперименте настоящие ошибки исправлялись с первого раза
(`evals/reports/2026-11-06-loop.md`).
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

from assistant.generation.citations import CheckedStatement, SourceText
from assistant.generation.gigachat import ChatModel, LlmStats, Usage
from assistant.generation.prompt import load_prompt
from assistant.generation.revise import Remark, Reviser

PROMPT_VERSION = "review_v1"
MODEL = "claude-sonnet-5-5"
EFFORT = "high"
MAX_TOKENS = 2000
# temperature Claude на нашей версии API не принимает; адаптер её не отправляет.
_TEMPERATURE = 1e-4

SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "number": {"type": "integer"},
                    "comment": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["number", "comment", "supported"],
            },
        },
    },
    "required": ["statements"],
}


@dataclass(frozen=True)
class Item:
    """Что проверяется: текст, рецепты целиком и вступление ли это."""

    text: str
    recipes: list[str]
    intro: bool = False


@dataclass(frozen=True)
class Mark:
    supported: bool | None
    comment: str = ""


def recipes_of(statement: CheckedStatement, sources: Sequence[SourceText]) -> list[str]:
    """Рецепты, на которые ведут подтверждённые цитаты утверждения, целиком."""
    by_number = {s.number: s for s in sources}
    cited = dict.fromkeys(c.source for c in statement.citations if c.valid)
    return [f"Рецепт целиком:\n{by_number[n].text}" for n in cited]


def statements_block(items: Sequence[Item]) -> list[str]:
    """Утверждения с рецептами в том виде, в каком их видит и судья eval."""
    parts: list[str] = []
    for number, item in enumerate(items, start=1):
        kind = ' kind="intro"' if item.intro else ""
        parts += [
            "",
            f'<statement number="{number}"{kind}>',
            f"<text>{item.text}</text>",
        ]
        parts += [f"<fragment>\n{recipe}\n</fragment>" for recipe in item.recipes]
        parts.append("</statement>")
    return parts


def review_message(question: str, items: Sequence[Item]) -> str:
    head = [
        f"Вопрос пользователя: {question}",
        "",
        "Утверждения ответа и их фрагменты:",
    ]
    return "\n".join([*head, *statements_block(items)])


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    comment: str
    supported: bool


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    statements: list[_Verdict]


class Reviewer:
    def __init__(
        self,
        llm: ChatModel,
        *,
        model: str = MODEL,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = load_prompt(prompt_version)
        # Ответы, которые не разобрались: такие утверждения считаются непроверенными.
        self.failures = 0

    def check(self, question: str, items: Sequence[Item]) -> list[Mark]:
        if not items:
            return []
        completion = self.llm.chat(
            [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": review_message(question, items)},
            ],
            model=self.model,
            temperature=_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_schema", "schema": SCHEMA, "strict": True},
        )
        try:
            parsed = _Answer.model_validate_json(completion.content)
        except ValidationError:
            self.failures += 1
            return [Mark(None) for _ in items]
        found = {v.number: v for v in parsed.statements}
        return [
            Mark(v.supported, v.comment) if (v := found.get(n)) else Mark(None)
            for n in range(1, len(items) + 1)
        ]


@dataclass(frozen=True)
class Review:
    """Что сделала проверка — для журнала, eval и разбора."""

    # Номер утверждения (с 1) → вердикт первой проверки.
    first: dict[int, Mark]
    intro: Mark | None
    # Номер → текст исправления Haiku; None — Haiku не исправил или цитата не прошла код.
    revised: dict[int, str | None]
    # Номер → вердикт повторной проверки исправленного.
    second: dict[int, Mark]
    # Что увидит пользователь: одобренные и исправленные утверждения.
    final: list[CheckedStatement]
    intro_kept: bool
    seconds: float
    # Токены проверки (Sonnet) и доработки (Haiku): для цены ответа.
    review_usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    revise_usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))

    @property
    def flagged(self) -> list[int]:
        return [n for n, m in self.first.items() if m.supported is not True]


def _usage_since(stats: LlmStats, before: LlmStats) -> Usage:
    prompt = stats.prompt_tokens - before.prompt_tokens
    completion = stats.completion_tokens - before.completion_tokens
    cached = stats.precached_prompt_tokens - before.precached_prompt_tokens
    return Usage(prompt, completion, cached, prompt + completion)


def _snapshot(stats: LlmStats) -> LlmStats:
    return LlmStats(
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        precached_prompt_tokens=stats.precached_prompt_tokens,
    )


def review_answer(
    question: str,
    sources: Sequence[SourceText],
    statements: Sequence[CheckedStatement],
    intro: str,
    reviewer: Reviewer,
    reviser: Reviser,
    clock: Callable[[], float] = time.monotonic,
) -> Review:
    started = clock()
    before_review, before_revise = (
        _snapshot(reviewer.llm.stats),
        _snapshot(reviser.llm.stats),
    )
    numbered: Mapping[int, CheckedStatement] = dict(enumerate(statements, start=1))
    items = [Item(s.text, recipes_of(s, sources)) for s in statements]
    if intro:
        all_recipes = list(dict.fromkeys(r for item in items for r in item.recipes))
        items.append(Item(intro, all_recipes, intro=True))
    marks = reviewer.check(question, items)
    first = dict(zip(numbered, marks[: len(statements)], strict=True))
    intro_mark = marks[len(statements)] if intro else None
    flagged = [n for n, mark in first.items() if mark.supported is not True]

    revised: dict[int, CheckedStatement | None] = {}
    if flagged:
        remarks = [
            Remark(n, [first[n].comment] if first[n].comment else []) for n in flagged
        ]
        revised = reviser.revise(question, sources, numbered, remarks)
    current = {n: s for n, s in numbered.items() if n not in flagged}
    fixed = {n: s for n, s in revised.items() if s is not None}
    second: dict[int, Mark] = {}
    if fixed:
        again = reviewer.check(
            question, [Item(s.text, recipes_of(s, sources)) for s in fixed.values()]
        )
        second = dict(zip(fixed, again, strict=True))
        current.update({n: s for n, s in fixed.items() if second[n].supported is True})
    return Review(
        first=first,
        intro=intro_mark,
        revised={
            n: (fix.text if (fix := revised.get(n)) is not None else None)
            for n in flagged
        },
        second=second,
        final=[current[n] for n in sorted(current)],
        intro_kept=intro_mark is not None and intro_mark.supported is True,
        seconds=clock() - started,
        review_usage=_usage_since(reviewer.llm.stats, before_review),
        revise_usage=_usage_since(reviser.llm.stats, before_revise),
    )
