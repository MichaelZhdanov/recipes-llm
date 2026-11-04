"""Точный поиск ошибки: какой шаблон сообщения из кода совпал с текстом вопроса.

Шаблоны строит `make index` (`assistant/indexing/messages.py`), они лежат в
кусках кода. Совпадение — это когда все постоянные куски шаблона встречаются
в тексте вопроса по порядку:
- регистр и пробелы не важны, переводы строк тоже;
- между кусками — подстановка не длиннее MAX_HOLE символов, иначе куски
  «склеились» бы через полстраницы вставленного лога;
- среди нескольких совпадений лучше то, где постоянного текста больше.

Если пользователь вставил сообщение не целиком, полного совпадения нет.
Тогда запасной вариант: кусок шаблона не короче MIN_PARTIAL символов,
найденный в тексте подстрокой; из таких берётся самый длинный.

Шаблон с короткими кусками («File {path} not found») совпал бы с обычным
вопросом, поэтому в поиск идут только шаблоны, у которых самый длинный кусок
не короче MIN_LONGEST_PART символов.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from assistant.indexing.chunks import Chunk
from assistant.indexing.messages import Message

MAX_HOLE = 200
MIN_PARTIAL = 25
MIN_LONGEST_PART = 12
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class ExactHit:
    chunk: Chunk
    message: Message
    # Сколько постоянного текста шаблона совпало: длиннее — точнее.
    matched: int
    # True — совпал весь шаблон, False — только один длинный кусок.
    full: bool


@dataclass(frozen=True)
class _Entry:
    chunk: Chunk
    message: Message
    pattern: re.Pattern[str]
    # Нормализованные куски, длинные — для запасного поиска подстрокой.
    parts: tuple[str, ...]


def _entry(chunk: Chunk, message: Message) -> _Entry | None:
    parts = [normalize(part) for part in message.parts if normalize(part)]
    if not parts or max(len(p) for p in parts) < MIN_LONGEST_PART:
        return None
    hole = rf".{{0,{MAX_HOLE}}}?"
    pattern = re.compile(hole.join(re.escape(part) for part in parts))
    return _Entry(chunk, message, pattern, tuple(parts))


class MessageIndex:
    """Шаблоны сообщений всех кусков кода, загруженные один раз."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._entries = [
            entry
            for chunk in chunks
            for message in chunk.messages
            if (entry := _entry(chunk, message)) is not None
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def search(self, text: str, limit: int = 2) -> list[ExactHit]:
        """Лучшие совпадения, не больше одного на кусок кода."""
        question = normalize(text)
        hits = [
            ExactHit(e.chunk, e.message, sum(len(p) for p in e.message.parts), True)
            for e in self._entries
            if e.pattern.search(question)
        ]
        if not hits:
            for e in self._entries:
                found = [p for p in e.parts if len(p) >= MIN_PARTIAL and p in question]
                if found:
                    hits.append(
                        ExactHit(e.chunk, e.message, max(map(len, found)), False)
                    )
        best: dict[tuple[str, int], ExactHit] = {}
        for hit in sorted(hits, key=lambda h: -h.matched):
            best.setdefault((hit.chunk.path, hit.chunk.start_line), hit)
        return list(best.values())[:limit]
