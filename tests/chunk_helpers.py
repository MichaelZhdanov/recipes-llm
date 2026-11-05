"""Куски индекса для тестов.

Нарезчик документации ушёл вместе с Paperless, а тестам поиска, сборки
индекса и ответов нужны куски с настоящими номерами строк и заголовками.
На этапе 1 кусок станет рецептом: тогда меняется эта фабрика и проверки
полей куска, а не каждый тест.
"""

import re

from assistant.indexing.chunks import Chunk

_HEADING = re.compile(r"^(#{1,6}) (.+?)(?: \{#([\w-]+)\})?\s*$")


def slug(title: str) -> str:
    return re.sub(r"[^\w]+", "-", title.lower()).strip("-")


def make_chunk(
    text: str,
    *,
    source_id: str = "docs/a.md#a",
    start_line: int = 1,
    kind: str = "doc",
    ref: str = "v3.4.2",
) -> Chunk:
    """Один кусок: путь и заголовок берутся из source_id вида `путь#якорь`."""
    path, _, anchor = source_id.partition("#")
    end_line = start_line + text.count("\n")
    return Chunk.model_validate(
        {
            "kind": kind,
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "title": anchor or path,
            "breadcrumb": anchor or path,
            "source_id": source_id,
            "url": f"https://x/{path}#L{start_line}-L{end_line}",
            "site_url": None,
            "ref": ref,
            "text": text,
        }
    )


def markdown_sections(
    markdown: str, *, path: str, ref: str = "v3.4.2", repo_url: str = "https://x"
) -> list[Chunk]:
    """Разделы Markdown как куски: заголовок с текстом до следующего заголовка.

    Якорь — из `{#id}` или из текста заголовка, путь заголовков — от `#`.
    Заголовок без текста (и раздел первого уровня) куском не становится.
    """
    lines = markdown.split("\n")
    headings = [(n, m) for n, line in enumerate(lines) if (m := _HEADING.match(line))]
    chunks: list[Chunk] = []
    trail: dict[int, str] = {}
    for index, (start, match) in enumerate(headings):
        level, title, anchor = len(match[1]), match[2], match[3]
        trail = {k: v for k, v in trail.items() if k < level} | {level: title}
        end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        body = lines[start:end]
        while body and not body[-1].strip():
            body.pop()
        if level == 1 or len(body) == 1:
            continue
        first, last = start + 1, start + len(body)
        chunks.append(
            Chunk(
                kind="doc",
                path=path,
                start_line=first,
                end_line=last,
                title=title,
                breadcrumb=" > ".join(trail[k] for k in sorted(trail)),
                source_id=f"{path}#{anchor or slug(title)}",
                url=f"{repo_url}/blob/{ref}/{path}?plain=1#L{first}-L{last}",
                site_url=None,
                ref=ref,
                text="\n".join(body),
            )
        )
    return chunks
