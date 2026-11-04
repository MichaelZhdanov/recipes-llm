"""Кандидаты в глоссарий из официального русского перевода Paperless-ngx.

`uv run python scripts/glossary_candidates.py data/paperless-src/v3.4.2`

Перевод интерфейса Paperless делает сообщество в Crowdin, он лежит в теге:
`src/locale/ru_RU/LC_MESSAGES/django.po`. Это те слова, которые русский
пользователь видит в интерфейсе. Скрипт оставляет из перевода термины:
- строка из `models.py`, то есть название сущности или поля Paperless;
- 1–3 слова, без плейсхолдеров и знаков препинания;
- термин встречается в документации (без changelog), отсортировано
  по частоте.

Это только кандидаты. Глоссарий (`assistant/generation/glossary.tsv`)
собирается из них вручную и одобряется автором: в переводе Crowdin есть
ошибки, например `never` → «некогда».
"""

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PO_FILE = Path("src/locale/ru_RU/LC_MESSAGES/django.po")
_STRING = re.compile(r'^(msgid|msgstr) "(.*)"$')
_TERM = re.compile(r"^[A-Za-z][A-Za-z -]*$")


@dataclass(frozen=True)
class Entry:
    msgid: str
    msgstr: str
    references: tuple[str, ...]


def parse_po(text: str) -> Iterator[Entry]:
    """Простой разбор .po: пары msgid/msgstr и ссылки `#:` над ними.

    Многострочные строки склеиваются. Пустой msgid (шапка файла) и пустой
    перевод пропускаются.
    """
    references: list[str] = []
    current: dict[str, str] = {}
    key = ""

    def flush() -> Iterator[Entry]:
        if current.get("msgid") and current.get("msgstr"):
            yield Entry(current["msgid"], current["msgstr"], tuple(references))

    for line in [*text.splitlines(), ""]:
        if line.startswith("#:"):
            if "msgstr" in current:
                yield from flush()
                current, references = {}, []
            references += line[2:].split()
        elif match := _STRING.match(line):
            key = match.group(1)
            if key == "msgid" and "msgstr" in current:
                yield from flush()
                current, references = {}, []
            current[key] = match.group(2)
        elif line.startswith('"') and key:
            current[key] += line.strip()[1:-1]
        elif not line.strip():
            yield from flush()
            current, references, key = {}, [], ""


def is_term(entry: Entry) -> bool:
    return (
        any("models.py" in ref for ref in entry.references)
        and bool(_TERM.match(entry.msgid))
        and len(entry.msgid.split()) <= 3
    )


def frequency(term: str, docs: str) -> int:
    return len(
        re.findall(rf"(?<![\w-]){re.escape(term)}(?![\w-])", docs, re.IGNORECASE)
    )


def candidates(source_root: Path) -> list[tuple[int, str, str]]:
    entries = parse_po((source_root / PO_FILE).read_text(encoding="utf-8"))
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((source_root / "docs").glob("*.md"))
        if path.name != "changelog.md"
    )
    seen: dict[str, tuple[int, str, str]] = {}
    for entry in entries:
        if not is_term(entry):
            continue
        key = entry.msgid.lower()
        count = frequency(entry.msgid, docs)
        if count and key not in seen:
            seen[key] = (count, entry.msgid.lower(), entry.msgstr)
    return sorted(seen.values(), key=lambda item: (-item[0], item[1]))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    for count, term, translation in candidates(Path(argv[1])):
        print(f"{count}\t{term}\t{translation}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
