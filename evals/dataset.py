"""Эталонный набор: схема кейса и загрузка `evals/dataset/*.yaml`.

Схема та же, что в CLAUDE.md (раздел «Верификация»). Лишние поля и опечатки
в ключах — ошибка: молча проигнорированное `expected_source` без «s» дало бы
кейс, который никогда не засчитывается.
"""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASET_DIR = Path(__file__).parent / "dataset"

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


def load_cases(directory: Path = DATASET_DIR) -> list[Case]:
    cases: list[Case] = []
    for file in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"{file}: ожидается YAML-список кейсов")
        cases.extend(Case.model_validate(item) for item in raw)
    ids = [case.id for case in cases]
    repeated = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if repeated:
        raise ValueError(f"повторяются id кейсов: {repeated}")
    return cases
