"""Инструмент search_recipes: описание для LLM, разбор вопроса, проверка аргументов.

Путь вопроса:
1. LLM ассистента (по умолчанию Claude Haiku 5.5, `assistant/generation/llm.py`)
   получает вопрос, до HISTORY_LIMIT прошлых реплик и словарь
   ингредиентов индекса с числом рецептов у каждого имени (промпт
   `search_v2`) и вызывает search_recipes — это режим «всегда ищем».
   GigaChat вызов задаётся принудительно; Claude принудительный вызов не
   принимает (или теряет размышления), там об этом говорит промпт.
2. Код проверяет аргументы: имена ингредиентов — только из словаря, чужие
   отбрасываются и запоминаются в `dropped` (их считает eval). Пустой
   `query` заменяется вопросом.
3. Поиск: фильтр Qdrant по `conditions`, векторы по `query`
   (`Retriever.search`).

Модель не выдумывает фильтр, которого нет в данных: даже если она напишет
«куриное мясо», в Qdrant уйдут только имена, которые реально есть в рецептах.
"""

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from assistant.generation.gigachat import (
    DETERMINISTIC_TEMPERATURE,
    ChatModel,
    LlmError,
    Message,
)
from assistant.generation.prompt import load_prompt
from assistant.retrieval.filters import IngredientFilter

NAME = "search_recipes"
# v1 — только имена; v2 — имена с частотами, альтернативы одного продукта
# в одной группе, query без опечаток (evals/reports/2026-11-05-stage2.md).
PROMPT_VERSION = "search_v2"
# Число рецептов, которое модель может скопировать вместе с именем из словаря.
_COUNT_SUFFIX = re.compile(r"\s*[—-]\s*\d+$")
# Сколько последних реплик диалога видит разбор (решение 30).
HISTORY_LIMIT = 10
# Самый длинный честный разбор — около 500 токенов (на «постное» — несколько
# десятков продуктов). Больше — это зацикливание: GigaChat повторяет имена,
# пока не кончится лимит, и возвращает пустой ответ с finish_reason: length.
# 1000 обрывает цикл за ≈ 20 с, 3000 — за минуту (evals/reports/2026-11-05-stage2.md).
MAX_TOKENS = 1000
# Зацикливание случайно: тот же запрос в следующий раз обычно проходит
# (проверено на своих вопросах; repetition_penalty и maxItems его не убирают).
LENGTH_ATTEMPTS = 3

SEARCH_RECIPES: dict[str, object] = {
    "name": NAME,
    "description": (
        "Найти рецепты в базе povarenok.ru: по смыслу вопроса и с условиями "
        "по ингредиентам — что должно быть в рецепте и чего быть не должно."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "что ищет пользователь, одной фразой, с учётом диалога",
            },
            "include": {
                "type": "array",
                "description": (
                    "группы имён из словаря: в рецепте хотя бы одно имя из каждой группы"
                ),
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "exclude": {
                "type": "array",
                "description": "имена из словаря, которых в рецепте быть не должно",
                "items": {"type": "string"},
            },
        },
        "required": ["query", "include", "exclude"],
    },
    # Примеры для GigaChat (поле её API). Без них модель не заполняла include
    # вовсе — проверено на своих вопросах, не из эталонного набора. Продукты
    # в примерах в наборе не встречаются, чтобы не подсказывать ответы на кейсы.
    "few_shot_examples": [
        {
            "request": "что-нибудь из нута и болгарского перца, только без кинзы",
            "params": {
                "query": "блюдо из нута с болгарским перцем без кинзы",
                "include": [
                    ["нут"],
                    [
                        "перец болгарский",
                        "перец сладкий",
                        "перец сладкий красный",
                        "перец сладкий желтый",
                        "перец сладкий зеленый",
                    ],
                ],
                "exclude": ["кинза"],
            },
        },
        {
            "request": "сладкое, но ваниль не переношу",
            "params": {
                "query": "десерт без ванили",
                "include": [],
                "exclude": [
                    "ваниль",
                    "ванилин",
                    "ванильный сахар",
                    "ванильная эссенция",
                    "сахар ванильный",
                ],
            },
        },
        {
            "request": "что-нибудь с вишней, только без алкоголя",
            "params": {
                "query": "блюдо с вишней без алкоголя",
                "include": [["вишня"]],
                "exclude": [
                    "вино белое сухое",
                    "вино красное сухое",
                    "коньяк",
                    "ликер",
                    "водка",
                    "ром",
                    "пиво светлое",
                    "шампанское",
                    "бренди",
                    "виски",
                ],
            },
        },
        {
            "request": "хочу рататуй",
            "params": {"query": "рататуй", "include": [], "exclude": []},
        },
    ],
}


@dataclass(frozen=True)
class SearchArgs:
    query: str
    conditions: IngredientFilter
    # Имена из аргументов модели, которых нет в словаре: в фильтр не попали.
    dropped: tuple[str, ...] = ()


def _strings(value: object) -> list[str]:
    """Список строк из того, что прислала модель: строка — список из одной."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def parse_arguments(
    arguments: Mapping[str, object], vocabulary: Collection[str], *, question: str
) -> SearchArgs:
    """Аргументы модели → запрос и условия. Имена не из словаря отбрасываются."""
    dropped: list[str] = []

    def known(names: list[str]) -> list[str]:
        result = []
        for name in names:
            normalized = _COUNT_SUFFIX.sub("", name.strip().lower())
            if normalized in vocabulary:
                result.append(normalized)
            elif normalized:
                dropped.append(normalized)
        return result

    raw_include = arguments.get("include")
    # Плоский список ["курица", "рис"] — это каждое имя своей группой.
    groups = raw_include if isinstance(raw_include, list) else []
    include = [known(_strings(group)) for group in groups]
    exclude = known(_strings(arguments.get("exclude")))
    query = arguments.get("query")
    return SearchArgs(
        query=query.strip() if isinstance(query, str) and query.strip() else question,
        conditions=IngredientFilter.of(include, exclude),
        dropped=tuple(dict.fromkeys(dropped)),
    )


def system_prompt(vocabulary: Mapping[str, int]) -> str:
    """Промпт со словарём: «имя — число рецептов», по алфавиту, чтобы похожие
    имена стояли рядом («краб», «крабовое мясо», «крабовые палочки»)."""
    lines = (f"{name} — {count}" for name, count in sorted(vocabulary.items()))
    # replace, а не format: в промпте могут появиться фигурные скобки JSON.
    return load_prompt(PROMPT_VERSION).replace("{vocabulary}", "\n".join(lines))


class SearchPlanner:
    """Разбор вопроса в аргументы search_recipes принудительным вызовом функции."""

    def __init__(
        self, llm: ChatModel, *, model: str, vocabulary: Mapping[str, int]
    ) -> None:
        self._llm = llm
        self._model = model
        self._vocabulary = frozenset(vocabulary)
        self._system: Message = {"role": "system", "content": system_prompt(vocabulary)}
        # Модели с версией из ответов API и число повторов: пишутся в отчёт eval.
        self.served: set[str] = set()
        self.retries = 0

    def plan(self, question: str, history: Sequence[Message] = ()) -> SearchArgs:
        messages: list[Message] = [
            self._system,
            *history[-HISTORY_LIMIT:],
            {"role": "user", "content": question},
        ]
        for attempt in range(LENGTH_ATTEMPTS):
            if attempt:
                self.retries += 1
            completion = self._llm.chat(
                messages,
                model=self._model,
                temperature=DETERMINISTIC_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                functions=[SEARCH_RECIPES],
                function_call={"name": NAME},
            )
            self.served.add(completion.model)
            if completion.finish_reason != "length":
                break
        call = completion.function_call
        if call is None or call["name"] != NAME:
            raise LlmError(
                f"модель не вызвала {NAME} (finish_reason: {completion.finish_reason})"
            )
        return parse_arguments(call["arguments"], self._vocabulary, question=question)
