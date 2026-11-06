"""Эталонный набор: схема кейса и загрузка `evals/dataset/*.yaml`.

Схема та же, что в CLAUDE.md (раздел «Верификация»). Лишние поля и опечатки
в ключах — ошибка: молча проигнорированное `expected_includes` с лишней «s»
дало бы кейс, который никогда не засчитывается.

Рядом с основным набором лежат ещё два, у каждого свой загрузчик:
- `holdout/` — отложенная выборка. Обычный `load_cases` её не видит:
  её смотрят один раз в конце, когда модель и промпт уже выбраны;
- `judge/canaries.yaml` — «канарейки» для проверки модели-судьи:
  утверждение, фрагмент и правильный вердикт `supported`;
- `judge/clarity_canaries.yaml` — канарейки для оценки грамотности
  `clear`: смысл подтверждён, а текст хороший или плохой;
- `judge/relevance_canaries.yaml` — канарейки судьи релевантности:
  вопрос, рецепт и правильный ответ `relevant`.
"""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_DIR = Path(__file__).parent / "dataset"
HOLDOUT_DIR = DATASET_DIR / "holdout"
CANARIES_FILE = DATASET_DIR / "judge" / "canaries.yaml"
CLARITY_CANARIES_FILE = DATASET_DIR / "judge" / "clarity_canaries.yaml"
RELEVANCE_CANARIES_FILE = DATASET_DIR / "judge" / "relevance_canaries.yaml"
STEPS_CANARIES_FILE = DATASET_DIR / "judge" / "steps_canaries.yaml"

Category = Literal[
    "include",
    "exclude",
    "by_name",
    "semantic",
    "how_to_cook",
    "out_of_scope",
    "chit_chat",
    "escalation",
    "security",
    "follow_up",
]
Behavior = Literal["answer", "abstain", "escalate", "no_search"]
# Поводы передать разговор человеку — правила эскалации из PLAN.md (F5).
EscalationReason = Literal[
    "request",
    "not_found_twice",
    "failure",
    "cooking_failed",
    "unclear_recipe",
    "health",
    "data_issue",
]


class Turn(BaseModel):
    """Реплика из начала диалога (кейсы follow_up)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z_]+-\d{3}$")
    category: Category
    question: str = Field(min_length=1)
    # Как написан вопрос: обычно, разговорно или с опечатками.
    style: Literal["plain", "colloquial", "typo"] = "plain"
    # Реплики до вопроса: у follow_up вопрос понятен только из них.
    turns: list[Turn] = []
    expected_behavior: Behavior
    # Группы альтернатив из словаря датасета: [[курица, филе куриное], [рис]].
    # В подходящем рецепте есть хотя бы одно имя из каждой группы.
    expected_include: list[list[str]] = []
    # Этих ингредиентов в подходящем рецепте нет.
    expected_exclude: list[str] = []
    # by_name: хотя бы одно слово должно быть в названии рецепта.
    expected_name: list[str] = []
    escalation_reason: EscalationReason | None = None
    must_include: list[str] = []
    must_not_include: list[str] = []
    notes: str = ""

    @model_validator(mode="after")
    def _conditions_match_category(self) -> Self:
        required = {
            "include": ("expected_include", self.expected_include),
            "exclude": ("expected_exclude", self.expected_exclude),
            "by_name": ("expected_name", self.expected_name),
            "follow_up": ("turns", self.turns),
        }
        if self.category in required:
            field, value = required[self.category]
            if not value:
                raise ValueError(f"{self.id}: у кейса {self.category} пустое {field}")
        if any(not group for group in self.expected_include):
            raise ValueError(f"{self.id}: пустая группа в expected_include")
        return self

    @model_validator(mode="after")
    def _behavior_matches_category(self) -> Self:
        fixed = {"out_of_scope": "abstain", "chit_chat": "no_search"}
        if self.category in fixed and self.expected_behavior != fixed[self.category]:
            raise ValueError(
                f"{self.id}: у {self.category} expected_behavior — {fixed[self.category]}"
            )
        escalates = self.expected_behavior == "escalate"
        if escalates != (self.escalation_reason is not None):
            raise ValueError(
                f"{self.id}: escalation_reason задаётся тогда и только тогда, "
                "когда expected_behavior — escalate"
            )
        return self


class Canary(BaseModel):
    """Пара «утверждение — фрагмент» с известным ответом для модели-судьи.

    Судья должен сказать, подтверждает ли фрагмент утверждение. Канарейки
    с `supported: false` расходятся с фрагментом тонко (другое имя параметра,
    другое значение по умолчанию): если судья их пропускает, его оценкам
    groundedness верить нельзя.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^canary-\d{3}$")
    statement: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    supported: bool
    notes: str = ""


class ClarityCanary(BaseModel):
    """Утверждение, которое фрагмент подтверждает, с известной оценкой грамотности.

    `clear: false` — смысл верный, но текст плохой: неверный перевод термина
    («Docker-объём» вместо «том»), ошибка, обрывок разметки. Если судья такое
    пропускает, его метрике грамотности верить нельзя.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^clarity-\d{3}$")
    statement: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    clear: bool
    notes: str = ""


class RelevanceCanary(BaseModel):
    """Вопрос и рецепт с известным ответом «подходит или нет» для судьи релевантности.

    Канарейки с `relevant: false` расходятся с вопросом тонко (в постном
    блюде сливочное масло, «к чаю» — пирог с мясом): если судья их принимает,
    его метрике semantic_relevance верить нельзя.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^relevance-\d{3}$")
    question: str = Field(min_length=1)
    name: str = Field(min_length=1)
    ingredients: list[str] = Field(min_length=1)
    relevant: bool
    notes: str = ""


class StepsCanary(BaseModel):
    """Рецепт и шаги приготовления, написанные ИИ, с известным ответом «можно показать».

    Шаги пишутся, когда страницы рецепта на сайте нет. Канарейки с
    `approved: false` расходятся со стандартом тонко — один лишний
    ингредиент, выдуманные «40 минут при 180°», опасный шаг: если проверка
    шагов их пропускает, показывать её шаги пользователю нельзя.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^steps-\d{3}$")
    # Рецепт целиком: название и все строки «Ингредиент — количество», через \n.
    recipe: str = Field(min_length=1)
    steps: str = Field(min_length=1)
    approved: bool
    notes: str = ""


def _read_list(file: Path) -> list[object]:
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{file}: ожидается YAML-список")
    return raw


def _check_unique(ids: list[str]) -> None:
    repeated = sorted({item for item in ids if ids.count(item) > 1})
    if repeated:
        raise ValueError(f"повторяются id: {repeated}")


def load_cases(directory: Path = DATASET_DIR) -> list[Case]:
    """Кейсы из `*.yaml` в каталоге. Подкаталоги (holdout, judge) не читаются."""
    cases = [
        Case.model_validate(item)
        for file in sorted(directory.glob("*.yaml"))
        for item in _read_list(file)
    ]
    _check_unique([case.id for case in cases])
    return cases


# Канарейки, которые устарели по решению автора. YAML субагента вручную не
# правится, поэтому они не удаляются из файла, а пропускаются при загрузке.
RETIRED_CANARIES = {
    "canary-043": "2026-11-06, автор: зачем нужен общеизвестный продукт "
    "(«желатин — чтобы застыло») — не ошибка, а общеизвестное о продукте",
    "relevance-036": "2026-11-06, автор: кофейная выпечка к кофе подходит — "
    "напиток как вкусовая добавка не мешает; не подходит то, что на напитке "
    "готовят (мясо, тушённое в вине)",
}


def load_canaries(file: Path = CANARIES_FILE, *, retired: bool = False) -> list[Canary]:
    """Канарейки опоры на рецепт; устаревшие (RETIRED_CANARIES) — только с retired=True."""
    canaries = [Canary.model_validate(item) for item in _read_list(file)]
    _check_unique([canary.id for canary in canaries])
    return [c for c in canaries if retired or c.id not in RETIRED_CANARIES]


def load_clarity_canaries(file: Path = CLARITY_CANARIES_FILE) -> list[ClarityCanary]:
    canaries = [ClarityCanary.model_validate(item) for item in _read_list(file)]
    _check_unique([canary.id for canary in canaries])
    return canaries


def load_relevance_canaries(
    file: Path = RELEVANCE_CANARIES_FILE, *, retired: bool = False
) -> list[RelevanceCanary]:
    """Канарейки релевантности; устаревшие (RETIRED_CANARIES) — только с retired=True."""
    canaries = [RelevanceCanary.model_validate(item) for item in _read_list(file)]
    _check_unique([canary.id for canary in canaries])
    return [c for c in canaries if retired or c.id not in RETIRED_CANARIES]


def load_steps_canaries(file: Path = STEPS_CANARIES_FILE) -> list[StepsCanary]:
    canaries = [StepsCanary.model_validate(item) for item in _read_list(file)]
    _check_unique([canary.id for canary in canaries])
    return canaries
