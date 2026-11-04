"""Тип вопроса: разбор ошибки или обычный вопрос.

Правило одно и без LLM: если точный поиск нашёл в тексте вопроса шаблон
сообщения из кода, это ошибка, и ответ строится по промпту разбора ошибок.
Иначе — обычный вопрос, всё как на этапе 2. Тип «диагностика» (живой
Paperless) появится на этапе 5.
"""

from collections.abc import Sequence
from typing import Literal

from assistant.retrieval.store import Hit

QuestionType = Literal["error", "general"]


def route(hits: Sequence[Hit]) -> QuestionType:
    return "error" if any(hit.exact for hit in hits) else "general"
