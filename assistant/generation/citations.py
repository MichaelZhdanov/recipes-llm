"""Формат ответа модели и проверка цитат кодом, без LLM.

Модель отвечает списком утверждений. У каждого утверждения есть цитаты:
номер рецепта из промпта и строка, скопированная из него дословно.
Здесь проверяется, что каждая цитата действительно есть в рецепте.
Утверждение без подтверждённой цитаты удаляется целиком: лучше ответить
меньше, чем ответить выдумкой (принцип 1 из CLAUDE.md).

Правила для цитаты (решение автора 2026-11-05):
- номер рецепта есть в промпте;
- цитата — **целая строка** текста рецепта: название или строка
  ингредиента («Молоко — 250 мл»). Не кусок строки: «молок» из «Молоко —
  250 мл» не подтверждает ничего. Сравнение терпит оформление (`normalize`):
  тире любого вида, кавычки, разметку Markdown, пробелы, регистр и «ё».

Правило для утверждения — **числа**: каждое число из утверждения должно
стоять в процитированных строках. Так модель не может назвать количество,
которого нет в рецепте («500 мл», когда в рецепте 250). Единицу правило не
сверяет: «250 г» при «250 мл» в рецепте пройдёт.

Номер процитированной строки (первая — название, дальше по строке на
ингредиент) сохраняется: по нему видно, на что опирается утверждение.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from assistant.indexing.chunks import Chunk

# --- Формат ответа модели (JSON по схеме, см. prompts/recipes_v1.md) ---


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: int
    quote: str


class Statement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citations: list[Citation]


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # answer — утверждения с цитатами; not_found — ответа в рецептах нет;
    # chat — болтовня («привет», «спасибо»), без поиска фактов.
    status: Literal["answer", "not_found", "chat"]
    # Короткий текст без фактов о рецептах: ответ на болтовню или подсказка,
    # что спросить, при not_found. При answer — пустой.
    message: str = ""
    statements: list[Statement]


# Та же схема в JSON Schema для `response_format`. Написана руками, а не через
# ModelAnswer.model_json_schema(): та выдаёт $defs и $ref, а их поддержку
# в GigaChat мы не проверяли. Claude требует additionalProperties: false
# у каждого объекта. Совпадение с моделью проверяет тест.
ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["answer", "not_found", "chat"]},
        "message": {"type": "string"},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source": {"type": "integer"},
                                "quote": {"type": "string"},
                            },
                            "required": ["source", "quote"],
                        },
                    },
                },
                "required": ["text", "citations"],
            },
        },
    },
    "required": ["status", "message", "statements"],
}


# --- Что видела модель ---


@dataclass(frozen=True)
class SourceText:
    """Источник под номером из промпта и его текст ровно в том виде, в каком он ушёл в модель.

    Текст уже прошёл маскирование ПДн и экранирование, поэтому цитата
    сверяется именно с ним. Ни то ни другое не меняет переводы строк,
    так что номера строк совпадают с текстом рецепта.
    """

    number: int
    chunk: Chunk
    text: str


# --- Результат проверки ---


@dataclass(frozen=True)
class CheckedCitation:
    source: int
    quote: str
    # None — цитата подтверждена, иначе причина отказа.
    error: str | None = None
    # Строки текста рецепта (с 1), где найден фрагмент.
    start_line: int | None = None
    end_line: int | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class CheckedStatement:
    text: str
    citations: list[CheckedCitation]
    # Числа из утверждения, которых нет в процитированных строках.
    unknown_numbers: list[str]

    @property
    def kept(self) -> bool:
        return not self.unknown_numbers and any(c.valid for c in self.citations)


# --- Нормализация и сравнение строк ---

_TRANSLATE = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "\u00a0": " ",
        "ё": "е",
        "Ё": "Е",
        # Кавычки не несут смысла для проверки: модель их то добавляет, то теряет.
        "“": None,
        "”": None,
        "„": None,
        "«": None,
        "»": None,
        "‘": None,
        "’": None,
        '"': None,
        "'": None,
        "`": None,
    }
)
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)\s]*\)")
# Курсив: `_слово_` или `*слово*` на границе слова.
_EMPHASIS = re.compile(r"(?<![\w*])([_*])(?=\S)(.+?)(?<=\S)\1(?![\w*])")
_EDGE_PUNCTUATION = ".,;:!?- "
# Число: целое, десятичное через точку или запятую, дробь «1/2».
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?(?:/\d+)?")


def normalize(text: str) -> str:
    """Выравнивает оформление, которое модель может передать по-своему:
    тире, кавычки, «ё», разметку Markdown. Переводы строк не трогает."""
    text = text.translate(_TRANSLATE).replace("**", "")
    text = _MARKDOWN_LINK.sub(r"\1", text)
    return _EMPHASIS.sub(r"\2", text)


def line_key(text: str) -> str:
    """Строка для сравнения: оформление выровнено, пробелы схлопнуты,
    знаки по краям и регистр не важны."""
    return " ".join(normalize(text).split()).strip(_EDGE_PUNCTUATION).casefold()


def check_citation(
    citation: Citation, sources: Sequence[SourceText]
) -> CheckedCitation:
    def fail(error: str) -> CheckedCitation:
        return CheckedCitation(citation.source, citation.quote, error=error)

    by_number = {source.number: source for source in sources}
    source = by_number.get(citation.source)
    if source is None:
        return fail(f"рецепта {citation.source} нет в промпте")
    if "\n" in citation.quote.strip():
        return fail("цитата — одна строка рецепта, а не несколько")
    key = line_key(citation.quote)
    if not key:
        return fail("пустая цитата")
    lines = [line_key(line) for line in source.text.split("\n")]
    if key in lines:
        number = lines.index(key) + 1
        return CheckedCitation(
            citation.source, citation.quote, start_line=number, end_line=number
        )
    if any(key in line for line in lines):
        return fail("это часть строки, а цитата — строка целиком")
    return fail("такой строки нет в рецепте")


def numbers_in(text: str) -> list[str]:
    """Числа текста без повторов: «1,5» и «1.5» — одно число."""
    return sorted({n.replace(",", ".") for n in _NUMBER.findall(text)})


def check_statement(
    statement: Statement, sources: Sequence[SourceText]
) -> CheckedStatement:
    citations = [check_citation(c, sources) for c in statement.citations]
    by_number = {source.number: source for source in sources}
    cited_lines = [
        by_number[c.source].text.split("\n")[c.start_line - 1]
        for c in citations
        if c.valid and c.start_line is not None
    ]
    known = set(numbers_in("\n".join(cited_lines)))
    unknown = [n for n in numbers_in(statement.text) if n not in known]
    return CheckedStatement(statement.text, citations, unknown)


def check_answer(
    answer: ModelAnswer, sources: Sequence[SourceText]
) -> list[CheckedStatement]:
    return [check_statement(s, sources) for s in answer.statements]
