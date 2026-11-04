"""Нарезка документации Paperless (Markdown) по заголовкам.

Кусок — это раздел от заголовка до следующего заголовка любого уровня.
Устроено так:
- строки с `#` внутри блоков ``` — это код, а не заголовки;
- якорь берётся из явного `{#id}`, иначе вычисляется как в Python-Markdown
  (toc.slugify), повторы получают суффиксы `_1`, `_2`;
- раздел без текста (заголовок, сразу за которым подзаголовок) куском не
  становится, но остаётся в пути заголовков;
- длинный раздел делится на части по пустым строкам, у частей общий source_id;
- текст до первого заголовка (введение страницы) — отдельный кусок, его
  заголовок берётся из front matter (`title:`).
"""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from assistant.indexing.chunks import Chunk, blob_url, docs_site_url

DEFAULT_MAX_CHARS = 2000

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# В setup.md три заголовка набраны HTML: <h3 id="migration_ng">…</h3>.
_HTML_HEADING = re.compile(r'^<h([1-6])\s+id="([^"]+)"[^>]*>(.*?)</h\1>\s*$')
# Курсив: сайт строит якорь по отрисованному тексту, где `_…_` уже нет.
_EMPHASIS = re.compile(r"(?<!\w)[_*](.+?)[_*](?!\w)")
_EXPLICIT_ID = re.compile(r"\s*\{\s*#([^\s}]+)[^}]*\}\s*$")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_FRONT_MATTER_TITLE = re.compile(r"^title:\s*(.+?)\s*$")


def slugify(value: str) -> str:
    """Тот же алгоритм, что markdown.extensions.toc.slugify (separator="-")."""
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _unique(anchor: str, used: set[str]) -> str:
    """Как toc.unique: повтор `x` становится `x_1`, затем `x_2`."""
    while anchor in used or not anchor:
        match = re.match(r"^(.*)_([0-9]+)$", anchor)
        anchor = (
            f"{match.group(1)}_{int(match.group(2)) + 1}" if match else f"{anchor}_1"
        )
    used.add(anchor)
    return anchor


def clean_heading(raw: str) -> str:
    """`[`PAPERLESS_REDIS=<url>`](#PAPERLESS_REDIS) {#PAPERLESS_REDIS}` → `PAPERLESS_REDIS=<url>`."""
    text = _EXPLICIT_ID.sub("", raw)
    text = _LINK.sub(r"\1", text)
    # HTML-теги не вырезаем: `<url>` и `<lang>` в заголовках — плейсхолдеры, а не теги.
    text = text.replace("`", "").replace("~~", "").replace("**", "")
    return _EMPHASIS.sub(r"\1", text).strip()


@dataclass
class _Section:
    level: int
    title: str
    anchor: str
    breadcrumb: str
    start: int  # номер строки заголовка (1-based); у введения — первая строка после front matter
    lines: list[str]


def _front_matter(lines: list[str]) -> tuple[int, str | None]:
    """Сколько строк занимает front matter и какой в нём title."""
    if not lines or lines[0].strip() != "---":
        return 0, None
    title = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return index + 1, title
        match = _FRONT_MATTER_TITLE.match(line)
        if match:
            title = match.group(1).strip("'\"")
    return 0, None


def _headings(lines: list[str], skip: int) -> dict[int, tuple[int, str]]:
    """Номер строки → (уровень, сырой текст) для настоящих заголовков, вне блоков ```."""
    found: dict[int, tuple[int, str]] = {}
    in_fence = False
    for number, line in enumerate(lines[skip:], start=skip + 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if match := _HEADING.match(line):
            found[number] = (len(match.group(1)), match.group(2))
        elif match := _HTML_HEADING.match(line):
            # Приводим к виду с явным якорем, дальше он обрабатывается как обычный.
            found[number] = (
                int(match.group(1)),
                f"{match.group(3)} {{#{match.group(2)}}}",
            )
    return found


def _sections(text: str, path: str) -> list[_Section]:
    lines = text.splitlines()
    skip, fm_title = _front_matter(lines)
    page_title = fm_title or Path(path).stem
    headings = _headings(lines, skip)
    has_h1 = any(level == 1 for level, _ in headings.values())

    used: set[str] = set()
    stack: list[tuple[int, str]] = []
    intro = _Section(0, page_title, "", page_title, skip + 1, [])
    sections = [intro]
    for number, line in enumerate(lines[skip:], start=skip + 1):
        if number not in headings:
            sections[-1].lines.append(line)
            continue
        level, raw = headings[number]
        title = clean_heading(raw)
        explicit = _EXPLICIT_ID.search(raw)
        anchor = _unique(explicit.group(1) if explicit else slugify(title), used)
        stack = [item for item in stack if item[0] < level] + [(level, title)]
        crumbs = [title for _, title in stack]
        if not has_h1:
            crumbs.insert(0, page_title)
        sections.append(
            _Section(level, title, anchor, " > ".join(crumbs), number, [line])
        )
    return sections


def _split(lines: list[str], start: int, max_chars: int) -> list[tuple[int, list[str]]]:
    """Делит строки раздела на части примерно по max_chars.

    Граница — пустая строка вне блока ```. Если пустых строк нет (длинный
    список в changelog), часть закрывается на любой строке вне блока, когда
    стала вдвое длиннее лимита. Блок кода целиком остаётся в одной части.
    """
    parts: list[tuple[int, list[str]]] = []
    current: list[str] = []
    current_start = start
    size = 0
    in_fence = False
    for offset, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        current.append(line)
        size += len(line) + 1
        paragraph_end = not line.strip() and size >= max_chars
        if not in_fence and (paragraph_end or size >= 2 * max_chars):
            parts.append((current_start, current))
            current, size = [], 0
            current_start = start + offset + 1
    if any(line.strip() for line in current):
        parts.append((current_start, current))
    return parts


def chunk_markdown(
    text: str, *, path: str, ref: str, repo_url: str, max_chars: int = DEFAULT_MAX_CHARS
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _sections(text, path):
        body = section.lines[1:] if section.level else section.lines
        if not any(line.strip() for line in body):
            continue
        source_id = f"{path}#{section.anchor}" if section.anchor else path
        for start, part in _split(section.lines, section.start, max_chars):
            # Пустые строки по краям части не нужны ни в тексте, ни в номерах строк.
            while part and not part[0].strip():
                part, start = part[1:], start + 1
            while part and not part[-1].strip():
                part = part[:-1]
            end = start + len(part) - 1
            chunks.append(
                Chunk(
                    kind="doc",
                    path=path,
                    start_line=start,
                    end_line=end,
                    title=section.title,
                    breadcrumb=section.breadcrumb,
                    source_id=source_id,
                    url=blob_url(repo_url, ref, path, start, end),
                    site_url=docs_site_url(path, section.anchor),
                    ref=ref,
                    text="\n".join(part),
                )
            )
    return chunks


def chunk_docs(root: Path, *, ref: str, repo_url: str) -> list[Chunk]:
    """Все `docs/*.md` дерева исходников, включая changelog."""
    chunks: list[Chunk] = []
    for file in sorted((root / "docs").glob("*.md")):
        path = file.relative_to(root).as_posix()
        chunks.extend(
            chunk_markdown(
                file.read_text(encoding="utf-8"), path=path, ref=ref, repo_url=repo_url
            )
        )
    return chunks
