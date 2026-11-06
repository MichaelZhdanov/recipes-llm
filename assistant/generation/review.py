"""Проверка ответа перед отправкой пользователю (решение автора 2026-11-06).

Haiku ответил, код проверил цитаты. Дальше Sonnet по той же рубрике, что
судья eval (`review_v3` — это `evals/prompts/judge_v10.md` без пункта о фактах
must_include; тест сверяет, чтобы они не разошлись), решает, подтверждают ли
рецепты каждое утверждение и вступление:

1. Всё «да» — ответ уходит пользователю как есть.
2. Утверждения с «нет» — один раз Haiku на доработку с конкретными
   замечаниями Sonnet (`revise.py`); исправленные — снова Sonnet.
3. Что и после этого «нет» — удаляется. Вступление с «нет» убирается без
   доработки. Не осталось ни одного утверждения — «не нашёл».

Отдельно Sonnet отмечает, отвечает ли рецепт утверждения на вопрос
(`review_v4`, решение автора 2026-11-06): на «что подать к вину» блюдо,
тушённое в вине, верно описано, но не ответ. Такие утверждения доработкой
текста не исправить — рецепты те же, — поэтому они удаляются сразу, а если
по вопросу не осталось ничего, ассистент один раз ищет заново (`answer.py`).

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

PROMPT_VERSION = "review_v4"
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
                    "answers_question": {"type": "boolean"},
                },
                "required": ["number", "comment", "supported", "answers_question"],
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
    # Отвечает ли рецепт утверждения на вопрос; None — не проверялось
    # (проверки до review_v4) или ответ не разобрался.
    relevant: bool | None = None


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


def review_message(
    question: str, items: Sequence[Item], context: Sequence[str] = ()
) -> str:
    """context — прошлые вопросы пользователя: уточнение («а без лука?») без
    них не понять, а проверка решает, отвечает ли рецепт на вопрос."""
    head = []
    if context:
        head += ["Ранее в диалоге пользователь спрашивал:"]
        head += [f"- {_escape(q)}" for q in context]
        head.append("")
    head += [
        f"Вопрос пользователя: {question}",
        "",
        "Утверждения ответа и их фрагменты:",
    ]
    return "\n".join([*head, *statements_block(items)])


def _escape(text: str) -> str:
    """Реплики пользователя: тегом из них не выйти в блок утверждений."""
    return text.replace("<", "&lt;")


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    comment: str
    supported: bool
    answers_question: bool = True


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

    def check(
        self, question: str, items: Sequence[Item], context: Sequence[str] = ()
    ) -> list[Mark]:
        if not items:
            return []
        completion = self.llm.chat(
            [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": review_message(question, items, context)},
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
            Mark(v.supported, v.comment, v.answers_question)
            if (v := found.get(n))
            else Mark(None)
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
    # Номер утверждения → почему его рецепт не отвечает на вопрос; такие
    # утверждения удалены без доработки текста.
    irrelevant: dict[int, str] = field(default_factory=dict)
    # Токены проверки (Sonnet) и доработки (Haiku): для цены ответа.
    review_usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))
    revise_usage: Usage = field(default_factory=lambda: Usage(0, 0, 0, 0))

    @property
    def flagged(self) -> list[int]:
        """Утверждения, которые ушли на доработку текста."""
        return [
            n
            for n, m in self.first.items()
            if m.supported is not True and n not in self.irrelevant
        ]


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
    reviser: Reviser | None,
    clock: Callable[[], float] = time.monotonic,
    context: Sequence[str] = (),
) -> Review:
    """reviser=None — без доработки: «нет» сразу удаляется (после нового поиска
    доработка уже была). context — прошлые вопросы пользователя."""
    started = clock()
    before_review = _snapshot(reviewer.llm.stats)
    before_revise = _snapshot(reviser.llm.stats) if reviser else LlmStats()
    numbered: Mapping[int, CheckedStatement] = dict(enumerate(statements, start=1))
    items = [Item(s.text, recipes_of(s, sources)) for s in statements]
    if intro:
        all_recipes = list(dict.fromkeys(r for item in items for r in item.recipes))
        items.append(Item(intro, all_recipes, intro=True))
    marks = reviewer.check(question, items, context)
    first = dict(zip(numbered, marks[: len(statements)], strict=True))
    intro_mark = marks[len(statements)] if intro else None
    # Рецепт не по вопросу — текстом не исправить: удаляется сразу.
    irrelevant = {n: m.comment for n, m in first.items() if m.relevant is False}
    flagged = [
        n
        for n, mark in first.items()
        if mark.supported is not True and n not in irrelevant
    ]

    revised: dict[int, CheckedStatement | None] = {}
    if flagged and reviser is not None:
        remarks = [
            Remark(n, [first[n].comment] if first[n].comment else []) for n in flagged
        ]
        revised = reviser.revise(question, sources, numbered, remarks)
    current = {
        n: s for n, s in numbered.items() if n not in flagged and n not in irrelevant
    }
    fixed = {n: s for n, s in revised.items() if s is not None}
    second: dict[int, Mark] = {}
    if fixed:
        again = reviewer.check(
            question,
            [Item(s.text, recipes_of(s, sources)) for s in fixed.values()],
            context,
        )
        second = dict(zip(fixed, again, strict=True))
        current.update(
            {
                n: s
                for n, s in fixed.items()
                if second[n].supported is True and second[n].relevant is not False
            }
        )
    return Review(
        first=first,
        intro=intro_mark,
        # Без reviser доработки не было — и записывать нечего.
        revised={
            n: (fix.text if (fix := revised.get(n)) is not None else None)
            for n in (flagged if reviser is not None else [])
        },
        second=second,
        final=[current[n] for n in sorted(current)],
        intro_kept=intro_mark is not None and intro_mark.supported is True,
        seconds=clock() - started,
        review_usage=_usage_since(reviewer.llm.stats, before_review),
        revise_usage=(
            _usage_since(reviser.llm.stats, before_revise)
            if reviser
            else Usage(0, 0, 0, 0)
        ),
        irrelevant=irrelevant,
    )
