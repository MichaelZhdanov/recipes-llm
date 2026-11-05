"""Промпты ассистента: файлы `prompts/<версия>.md`, версия пишется в отчёт eval."""

from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")
