"""Промпты: модели не знают друг о друге (решение автора 2026-11-06).

Ответ пишет одна модель, проверяет другая того же вендора. Если в промпте
есть имя модели или провайдера, проверяющая может узнать «свою» и судить
мягче. Поэтому ни в одном промпте ассистента и судьи eval имён нет.
"""

import re
from pathlib import Path

import pytest

from assistant.generation.prompt import PROMPTS_DIR

EVAL_PROMPTS = Path(__file__).parent.parent / "evals" / "prompts"
NAMES = re.compile(
    r"haiku|sonnet|opus|claude|anthropic|gigachat|гигачат|deepseek|glm|qwen|gpt|openai",
    re.IGNORECASE,
)
PROMPTS = sorted([*PROMPTS_DIR.glob("*.md"), *EVAL_PROMPTS.glob("*.md")])


def test_prompts_are_found() -> None:
    assert len(PROMPTS) > 10


@pytest.mark.parametrize("path", PROMPTS, ids=lambda p: p.name)
def test_prompt_names_no_model_or_provider(path: Path) -> None:
    found = NAMES.findall(path.read_text(encoding="utf-8"))

    assert found == [], f"{path.name}: {sorted(set(found))}"
