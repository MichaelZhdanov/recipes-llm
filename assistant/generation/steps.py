"""Шаги приготовления для рецепта без живой страницы (решение автора 2026-11-06).

Обычно под рецептом — ссылка «способ приготовления» на povarenok.ru. Если
страница не отвечает 200, шагов у бота нет: в данных только название и
строки «Ингредиент — количество». Тогда по кнопке «🍳 Шаги»:

1. Haiku пишет шаги по названию и строкам (`steps_v1`).
2. Sonnet проверяет их по стандарту автора (`steps_review_v1`): только
   ингредиенты рецепта (плюс вода, соль, перец, масло «по вкусу»), каждая
   строка использована, количества не противоречат, время и температура —
   только из данных, ничего опасного, не против названия.
3. «Нет» — один раз Haiku переписывает по замечаниям, Sonnet проверяет снова.
4. «Нет» и после этого — шаги не показываются: лучше честное «не смог
   надёжно расписать», чем правдоподобная ошибка.

Как и в ответе, модели не знают друг о друге: в промптах нет имён моделей.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import httpx2
from pydantic import BaseModel, ConfigDict, ValidationError

from assistant.config import Settings
from assistant.generation.answer import escape_source_text
from assistant.generation.claude import Claude
from assistant.generation.gigachat import DETERMINISTIC_TEMPERATURE, ChatModel
from assistant.generation.llm import llm_model, make_chat_model
from assistant.generation.prompt import load_prompt
from assistant.generation.review import EFFORT as REVIEW_EFFORT
from assistant.generation.review import MODEL as REVIEW_MODEL

PROMPT_VERSION = "steps_v1"
REVIEW_PROMPT_VERSION = "steps_review_v1"
MAX_TOKENS = 1500

WRITE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "possible": {"type": "boolean"},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["possible", "steps"],
}
# Сначала нарушения, потом вердикт: модель решает, уже перечислив их.
REVIEW_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "problems": {"type": "array", "items": {"type": "string"}},
        "approved": {"type": "boolean"},
    },
    "required": ["problems", "approved"],
}

Status = Literal["approved", "rejected", "impossible", "failed"]


class _Written(BaseModel):
    model_config = ConfigDict(extra="ignore")
    possible: bool
    steps: list[str]


class _Checked(BaseModel):
    model_config = ConfigDict(extra="ignore")
    problems: list[str]
    approved: bool


@dataclass(frozen=True)
class Check:
    # None — ответ проверки не разобрался: шаги считаются непроверенными.
    approved: bool | None
    problems: list[str]


@dataclass(frozen=True)
class Steps:
    """Что получилось: шаги для показа и всё, что нужно eval и журналу.

    - approved — шаги прошли проверку (сразу или после доработки);
    - rejected — проверка не пропустила и доработку;
    - impossible — модель сама сказала, что надёжно расписать нельзя;
    - failed — ответ модели не разобрался.
    """

    status: Status
    seconds: float
    # Что показать пользователю; пусто, если не approved.
    steps: list[str] = field(default_factory=list)
    draft: list[str] = field(default_factory=list)
    first: Check | None = None
    revised: list[str] | None = None
    second: Check | None = None


def _escape(text: str) -> str:
    """Текст модели: из блока данных тегом не выйти."""
    return text.replace("<", "&lt;")


def _numbered(steps: Sequence[str]) -> str:
    return "\n".join(f"{n}. {_escape(step)}" for n, step in enumerate(steps, start=1))


def write_message(
    recipe: str, draft: Sequence[str] = (), problems: Sequence[str] = ()
) -> str:
    parts = [f"<recipe>\n{escape_source_text(recipe)}\n</recipe>"]
    if draft:
        parts.append(f"<draft>\n{_numbered(draft)}\n</draft>")
        listed = "\n".join(f"- {_escape(p)}" for p in problems)
        parts.append(f"<problems>\n{listed}\n</problems>")
    return "\n\n".join(parts)


def review_message(recipe: str, steps: Sequence[str]) -> str:
    return (
        f"<recipe>\n{escape_source_text(recipe)}\n</recipe>\n\n"
        f"<steps>\n{_numbered(steps)}\n</steps>"
    )


class StepsWriter:
    def __init__(
        self, llm: ChatModel, *, model: str, prompt_version: str = PROMPT_VERSION
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = load_prompt(prompt_version)

    def write(
        self, recipe: str, draft: Sequence[str] = (), problems: Sequence[str] = ()
    ) -> list[str] | None:
        """Шаги; [] — расписать надёжно нельзя; None — ответ не разобрался."""
        completion = self.llm.chat(
            [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": write_message(recipe, draft, problems)},
            ],
            model=self.model,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "schema": WRITE_SCHEMA,
                "strict": True,
            },
        )
        try:
            parsed = _Written.model_validate_json(completion.content)
        except ValidationError:
            return None
        steps = [step.strip() for step in parsed.steps if step.strip()]
        return steps if parsed.possible else []


class StepsReviewer:
    def __init__(
        self,
        llm: ChatModel,
        *,
        model: str = REVIEW_MODEL,
        prompt_version: str = REVIEW_PROMPT_VERSION,
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = load_prompt(prompt_version)

    def check(self, recipe: str, steps: Sequence[str]) -> Check:
        completion = self.llm.chat(
            [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": review_message(recipe, steps)},
            ],
            model=self.model,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "schema": REVIEW_SCHEMA,
                "strict": True,
            },
        )
        try:
            parsed = _Checked.model_validate_json(completion.content)
        except ValidationError:
            return Check(None, [])
        return Check(parsed.approved, [p.strip() for p in parsed.problems if p.strip()])


def make_steps(
    recipe: str,
    writer: StepsWriter,
    reviewer: StepsReviewer,
    clock: Callable[[], float] = time.monotonic,
) -> Steps:
    """recipe — текст рецепта: название и строки, как в индексе."""
    started = clock()

    def elapsed() -> float:
        return clock() - started

    draft = writer.write(recipe)
    if draft is None:
        return Steps("failed", elapsed())
    if not draft:
        return Steps("impossible", elapsed())
    first = reviewer.check(recipe, draft)
    if first.approved is True:
        return Steps("approved", elapsed(), steps=draft, draft=draft, first=first)
    # Без замечаний дорабатывать нечего: проверка не разобралась или
    # сказала «нет», не объяснив, — такие шаги не показываем.
    if not first.problems:
        return Steps("rejected", elapsed(), draft=draft, first=first)
    revised = writer.write(recipe, draft, first.problems)
    if not revised:
        return Steps("rejected", elapsed(), draft=draft, first=first, revised=revised)
    second = reviewer.check(recipe, revised)
    approved = second.approved is True
    return Steps(
        "approved" if approved else "rejected",
        elapsed(),
        steps=revised if approved else [],
        draft=draft,
        first=first,
        revised=revised,
        second=second,
    )


def make_steps_models(
    settings: Settings, client: httpx2.Client
) -> tuple[StepsWriter, StepsReviewer]:
    """Пишет модель ассистента, проверяет Sonnet с effort high — как ответ."""
    writer = StepsWriter(make_chat_model(settings, client), model=llm_model(settings))
    checker = Claude(
        api_key=settings.anthropic_api_key,
        api_url=settings.anthropic_api_url,
        client=client,
        effort=REVIEW_EFFORT,
    )
    return writer, StepsReviewer(checker)
