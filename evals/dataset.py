"""Эталонный набор: схема кейса и загрузка `evals/dataset/*.yaml`.

Схема та же, что в CLAUDE.md (раздел «Верификация»). Лишние поля и опечатки
в ключах — ошибка: молча проигнорированное `expected_source` без «s» дало бы
кейс, который никогда не засчитывается.

Рядом с основным набором лежат ещё два, у каждого свой загрузчик:
- `holdout/` — отложенная выборка. Обычный `load_cases` её не видит:
  её смотрят один раз в конце, когда модель и промпт уже выбраны;
- `judge/canaries.yaml` — «канарейки» для проверки модели-судьи:
  утверждение, фрагмент и правильный вердикт.
"""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_DIR = Path(__file__).parent / "dataset"
HOLDOUT_DIR = DATASET_DIR / "holdout"
CANARIES_FILE = DATASET_DIR / "judge" / "canaries.yaml"

Category = Literal[
    "howto", "config", "error", "out_of_scope", "security", "tool", "escalation"
]
# Категории, где кейс проверяет поиск: у них обязан быть хотя бы один источник.
RETRIEVAL_CATEGORIES = frozenset({"howto", "config", "error"})


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z_]+-\d{3}$")
    category: Category
    role: Literal["user", "engineer"]
    question: str = Field(min_length=1)
    expected_behavior: str = Field(pattern=r"^(answer|abstain|escalate|tool:[a-z_]+)$")
    expected_sources: list[str] = []
    must_include: list[str] = []
    must_not_include: list[str] = []
    notes: str = ""

    @model_validator(mode="after")
    def _retrieval_cases_have_sources(self) -> Self:
        if self.category in RETRIEVAL_CATEGORIES and not self.expected_sources:
            raise ValueError(
                f"{self.id}: у кейса категории {self.category} нет expected_sources"
            )
        return self

    @model_validator(mode="after")
    def _out_of_scope_means_abstain(self) -> Self:
        # Ответа нет ни в доке, ни в коде: правильное поведение одно — отказ,
        # а источник, на который он «должен» сослаться, противоречил бы кейсу.
        if self.category == "out_of_scope" and (
            self.expected_behavior != "abstain" or self.expected_sources
        ):
            raise ValueError(
                f"{self.id}: у out_of_scope expected_behavior — abstain, "
                "а expected_sources пустой"
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


def load_canaries(file: Path = CANARIES_FILE) -> list[Canary]:
    canaries = [Canary.model_validate(item) for item in _read_list(file)]
    _check_unique([canary.id for canary in canaries])
    return canaries
