"""Описание функции для API без поля few_shot_examples (Anthropic, OpenAI).

Функции заданы в формате GigaChat: у него есть `few_shot_examples` — примеры
вопросов и аргументов. У других API такого поля нет, поэтому примеры идут
в описание функции текстом.
"""

import json
from collections.abc import Mapping


def describe(function: Mapping[str, object]) -> str:
    description = str(function.get("description", ""))
    examples = function.get("few_shot_examples")
    if isinstance(examples, list) and examples:
        lines = [
            f"- «{e['request']}» → {json.dumps(e['params'], ensure_ascii=False)}"
            for e in examples
            if isinstance(e, dict)
        ]
        description += "\n\nПримеры вопросов и аргументов:\n" + "\n".join(lines)
    return description
