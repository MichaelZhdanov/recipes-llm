"""Формат ответа модели и проверка ссылок кодом, без LLM.

Модель отвечает списком утверждений. У каждого утверждения есть цитаты:
номер источника из промпта и фрагмент, скопированный из него дословно.
Здесь проверяется, что каждая цитата действительно есть в источнике.
Утверждение без подтверждённой цитаты удаляется целиком: лучше ответить
меньше, чем ответить выдумкой (принцип 1 из CLAUDE.md).

Правила для цитаты:
- номер источника есть в промпте;
- в цитате не меньше MIN_QUOTE_WORDS слов и MIN_QUOTE_CHARS символов
  (короткий фрагмент вроде «the» найдётся где угодно) и не больше
  MAX_QUOTE_CHARS, иначе ссылка на строки теряет смысл;
- фрагмент найден в тексте, который видела модель. Сравниваются слова,
  а оформление выравнивается с обеих сторон одинаково (`normalize`):
  - пробелы и переводы строк между словами любые, регистр не важен;
  - типографские тире приводятся к `-`, кавычки любого вида не
    учитываются;
  - разметка Markdown не учитывается: `**`, курсив `_x_` и `*x*`,
    обратные кавычки, ссылки `[текст](адрес)`, в том числе разбитые
    переносом строки;
  - знаки препинания по краям цитаты не учитываются: модель ставит
    точку после обрезанного предложения;
  - подчёркивание внутри слова не трогается: в `max_tokens` оно часть
    имени;
  - у кода не важен перенос строк у скобок и висячая запятая: модель
    сворачивает вызов из нескольких строк в одну. Префикс строки
    (`f"…"`, `r"…"`) не учитывается.
  Слова, числа, имена и их порядок должны совпасть;
- у документации совпадение не может целиком лежать в строке заголовка:
  заголовок только называет тему, а не подтверждает утверждение.

Правило для утверждения: имена с `_` в обратных кавычках (`max_tokens`)
должны встречаться в тексте процитированных источников. Так ловится
правдоподобное, но несуществующее имя. Для рецептов его дополнит проверка
количеств: «250 мл» из утверждения должно быть в рецепте (этап 3).

По месту совпадения считаются точные строки, и ссылка в ответе ведёт
ровно на них, а не на весь кусок.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from assistant.indexing.chunks import Chunk

MIN_QUOTE_WORDS = 3
MIN_QUOTE_CHARS = 20
MAX_QUOTE_CHARS = 400


# --- Формат ответа модели (JSON по схеме, см. prompts/answer_v1.md) ---


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

    status: Literal["answer", "not_found"]
    statements: list[Statement]


# Та же схема в JSON Schema для `response_format` GigaChat. Написана руками,
# а не через ModelAnswer.model_json_schema(): та выдаёт $defs и $ref, а их
# поддержку в GigaChat мы не проверяли. Совпадение с моделью проверяет тест.
ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answer", "not_found"]},
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
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
    "required": ["status", "statements"],
}


# --- Что видела модель ---


@dataclass(frozen=True)
class SourceText:
    """Источник под номером из промпта и его текст ровно в том виде, в каком он ушёл в модель.

    Текст уже прошёл маскирование ПДн и экранирование, поэтому цитата
    сверяется именно с ним. Ни то ни другое не меняет переводы строк,
    так что номера строк совпадают с исходным файлом.
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
    # Строки файла, где найден фрагмент.
    start_line: int | None = None
    end_line: int | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class CheckedStatement:
    text: str
    citations: list[CheckedCitation]
    # Имена из утверждения, которых нет в процитированных источниках.
    unknown_names: list[str]

    @property
    def kept(self) -> bool:
        return not self.unknown_names and any(c.valid for c in self.citations)


# --- Нормализация и поиск ---

_TRANSLATE = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "\u00a0": " ",
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
# Текст ссылки может переноситься на следующую строку, адрес — нет.
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)\s]*\)")
# Курсив: `_слово_` или `*слово*` на границе слова. Подчёркивание внутри
# слова (max_tokens) сюда не попадает.
_EMPHASIS = re.compile(r"(?<![\w*])([_*])(?=\S)(.+?)(?<=\S)\1(?![\w*])")
# Префикс строки Python: f"…", r'…', rb"…". Он не часть текста, а без
# кавычек приклеился бы к слову: f"Error" → «fError». Перед префиксом
# не может стоять буква или кавычка, поэтому ["fr", "de"] не трогается.
_STRING_PREFIX = re.compile(r"(?<![\w\"'])[rRbBuUfF]{1,2}(?=[\"'])")
_EDGE_PUNCTUATION = ".,;:!?"
# Токены цитаты: скобка или кусок без пробелов и скобок.
_TOKEN = re.compile(r"[()\[\]{}]|[^\s()\[\]{}]+")
_OPENING = frozenset("([{")
_CLOSING = frozenset(")]}")
# Висячая запятая перед закрывающей скобкой: `f(a, b,)` то же, что `f(a, b)`.
_TRAILING_COMMA = re.compile(r",\s*(?=[)\]}])")


def normalize(text: str) -> str:
    """Выравнивает оформление, которое модель может передать по-своему.

    Переводы строк не добавляются и не удаляются: по ним считаются строки.
    """
    text = _STRING_PREFIX.sub("", text)
    text = text.translate(_TRANSLATE).replace("**", "")
    text = _MARKDOWN_LINK.sub(r"\1", text)
    return _EMPHASIS.sub(r"\2", text)


def _separator(previous: str, token: str) -> str:
    """Что может стоять между соседними токенами цитаты в источнике.

    Модель переносит код по-своему: сворачивает вызов из нескольких строк
    в одну или наоборот. Поэтому у скобок пробелы и переводы строк
    необязательны, а перед закрывающей скобкой может быть висячая запятая.
    Между словами пробел обязателен, как и раньше.
    """
    if token in _CLOSING:
        return r",?\s*"
    if previous in _OPENING | _CLOSING or token in _OPENING:
        return r"\s*"
    return r"\s+"


def _quote_pattern(quote: str) -> re.Pattern[str] | None:
    text = normalize(quote).strip().strip(_EDGE_PUNCTUATION)
    tokens = _TOKEN.findall(_TRAILING_COMMA.sub("", text))
    if not tokens:
        return None
    body = re.escape(tokens[0]) + "".join(
        _separator(previous, token) + re.escape(token)
        for previous, token in pairwise(tokens)
    )
    # Цитата начинается и кончается на границе слова: «false» не совпадёт
    # с началом «falsehood», а «anguage» — с серединой «language».
    start = r"(?<!\w)" if re.match(r"\w", tokens[0]) else ""
    end = r"(?!\w)" if re.search(r"\w$", tokens[-1]) else ""
    return re.compile(start + body + end, re.IGNORECASE)


_DOC_HEADING = re.compile(r"^\s{0,3}(#{1,6}\s|<h[1-6][\s>])")
_FENCE = re.compile(r"^\s*(```|~~~)")


def heading_lines(text: str) -> set[int]:
    """Номера строк (с 0) с заголовками Markdown вне блоков кода."""
    headings = set()
    in_code = False
    for number, line in enumerate(text.split("\n")):
        if _FENCE.match(line):
            in_code = not in_code
        elif not in_code and _DOC_HEADING.match(line):
            headings.add(number)
    return headings


def check_citation(
    citation: Citation, sources: Sequence[SourceText]
) -> CheckedCitation:
    def fail(error: str) -> CheckedCitation:
        return CheckedCitation(citation.source, citation.quote, error=error)

    by_number = {source.number: source for source in sources}
    source = by_number.get(citation.source)
    if source is None:
        return fail(f"источника {citation.source} нет в промпте")
    quote = citation.quote.strip()
    if len(quote.split()) < MIN_QUOTE_WORDS or len(quote) < MIN_QUOTE_CHARS:
        return fail("слишком короткая цитата")
    if len(quote) > MAX_QUOTE_CHARS:
        return fail("слишком длинная цитата")
    pattern = _quote_pattern(quote)
    if pattern is None:
        return fail("пустая цитата")

    text = normalize(source.text)
    # Заголовки ищутся в исходном тексте: нормализация убирает обратные
    # кавычки, и ограды блоков кода ``` перестали бы распознаваться.
    headings = heading_lines(source.text) if source.chunk.kind == "doc" else set()
    found_only_in_heading = False
    for match in pattern.finditer(text):
        first = text.count("\n", 0, match.start())
        last = first + match.group().count("\n")
        if set(range(first, last + 1)) <= headings:
            found_only_in_heading = True
            continue
        return CheckedCitation(
            citation.source,
            citation.quote,
            start_line=source.chunk.start_line + first,
            end_line=source.chunk.start_line + last,
        )
    if found_only_in_heading:
        return fail("цитата совпадает только с заголовком")
    return fail("фрагмента нет в источнике")


# Имена, которые модель может выдумать: в обратных кавычках и с подчёркиванием.
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_SNAKE_NAME = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")


def names_in(statement: str) -> list[str]:
    """Имена из утверждения, которые обязаны быть в процитированном источнике."""
    names: set[str] = set()
    for span in _BACKTICKED.findall(statement):
        names.update(_SNAKE_NAME.findall(span))
    return sorted(names)


def _has_name(text: str, name: str) -> bool:
    """Имя целиком: OCR_LANGUAGE не находится внутри OCR_LANGUAGES."""
    return (
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
        is not None
    )


def check_statement(
    statement: Statement, sources: Sequence[SourceText]
) -> CheckedStatement:
    citations = [check_citation(c, sources) for c in statement.citations]
    cited = {c.source for c in citations if c.valid}
    cited_text = "\n".join(s.text for s in sources if s.number in cited)
    unknown = [
        name for name in names_in(statement.text) if not _has_name(cited_text, name)
    ]
    return CheckedStatement(statement.text, citations, unknown)


def check_answer(
    answer: ModelAnswer, sources: Sequence[SourceText]
) -> list[CheckedStatement]:
    return [check_statement(s, sources) for s in answer.statements]
