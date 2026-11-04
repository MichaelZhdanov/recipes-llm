"""Глоссарий: как переводить термины документации в ответах по-русски.

Модель переводит английские термины по-своему и иногда неверно для
контекста: Docker volume становится «объёмом», а не «томом». Глоссарий
(`glossary.tsv`, одобрен автором) фиксирует переводы:
- термины Paperless взяты как есть из официального перевода интерфейса
  (`src/locale/ru_RU/LC_MESSAGES/django.po` тега). Это слова, которые
  пользователь видит на экране, тест сверяет их с .po;
- термины документации и инфраструктуры (том, образ, реверс-прокси) —
  короткий список, которого в переводе интерфейса нет.

В промпт попадают не все термины, а только те, что встречаются в
найденных источниках: так промпт не раздувается, а модель видит перевод
именно тех слов, которые ей придётся переводить.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

GLOSSARY_FILE = Path(__file__).parent / "glossary.tsv"


@dataclass(frozen=True)
class Term:
    en: str
    ru: str
    # msgid в django.po, из которого взят перевод; None — термина в переводе нет.
    po_msgid: str | None
    # В каком смысле применять перевод: «том Docker, не „объём“».
    note: str

    @property
    def pattern(self) -> re.Pattern[str]:
        """Термин целым словом, с окончанием множественного числа: volume, volumes."""
        words = r"\s+".join(re.escape(word) for word in self.en.split())
        return re.compile(rf"(?<![\w-]){words}(?:s|es)?(?![\w-])", re.IGNORECASE)


def parse_glossary(text: str) -> list[Term]:
    terms = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("en\t"):
            continue
        en, ru, po_msgid, *note = line.split("\t")
        terms.append(
            Term(
                en.strip(),
                ru.strip(),
                None if po_msgid in ("", "-") else po_msgid,
                note[0].strip() if note else "",
            )
        )
    return terms


@cache
def load_glossary(path: Path = GLOSSARY_FILE) -> tuple[Term, ...]:
    return tuple(parse_glossary(path.read_text(encoding="utf-8")))


def terms_in(texts: Iterable[str], glossary: Sequence[Term]) -> list[Term]:
    """Термины глоссария, которые встречаются в текстах, в порядке глоссария."""
    joined = "\n".join(texts)
    return [term for term in glossary if term.pattern.search(joined)]


def render_terms(terms: Sequence[Term]) -> str:
    """`- volume — том (том Docker, не «объём»)`, по строке на термин."""
    return "\n".join(
        f"- {t.en} — {t.ru}" + (f" ({t.note})" if t.note else "") for t in terms
    )
