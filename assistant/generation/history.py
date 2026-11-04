"""Пометка утверждений, которые опираются только на историю изменений.

Запись changelog говорит, что поменялось в какой-то версии, а не как всё
устроено сейчас. Например, «Take ownership of k8s-at-home Helm chart»
(1.10.2) отменена в 1.12.0: «Chore: remove helm chart code». Модель
цитирует запись честно, проверка ссылок её пропускает, но в ответе это
звучит как текущее состояние.

Поэтому код дописывает к такому утверждению версию записи:
«Команда поддерживает Helm-чарт (история изменений: paperless-ngx 1.10.2).». Пользователь видит, что это
история, а не текущая документация. Сам факт так не исправляется: это
задача поиска, которому нужно найти и позднюю запись.

Пометка ставится, только если все подтверждённые цитаты утверждения взяты
из changelog. Если есть и текущая документация, утверждение опирается на неё.
"""

import re
from collections.abc import Sequence

from assistant.indexing.chunks import CHANGELOG_PATH, Chunk

_VERSION = re.compile(r"\d+\.\d+")


def changelog_version(breadcrumb: str) -> str | None:
    """Версия записи по пути заголовков.

    «Changelog > paperless-ngx 1.10.2 > Features» → «paperless-ngx 1.10.2»;
    в истории исходного проекта версия — отдельный заголовок:
    «Changelog > Paperless > 0.0.3» → «Paperless 0.0.3».
    """
    parts = breadcrumb.split(" > ")
    for index, part in enumerate(parts):
        if _VERSION.search(part):
            if part[0].isdigit() and index > 0:
                return f"{parts[index - 1]} {part}"
            return part
    return None


def history_note(chunks: Sequence[Chunk]) -> str:
    """Пометка для утверждения с цитатами из этих кусков; пустая, если не нужна."""
    if not chunks or any(chunk.path != CHANGELOG_PATH for chunk in chunks):
        return ""
    versions = [v for c in chunks if (v := changelog_version(c.breadcrumb))]
    if not versions:
        return " (история изменений)"
    return f" (история изменений: {', '.join(dict.fromkeys(versions))})"


def mark(text: str, chunks: Sequence[Chunk]) -> str:
    """Утверждение с пометкой перед финальным знаком препинания, если она нужна."""
    note = history_note(chunks)
    if not note:
        return text
    text = text.rstrip()
    if text and text[-1] in ".!?":
        return text[:-1] + note + text[-1]
    return text + note
