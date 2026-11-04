"""Нарезка Python-кода Paperless по функциям и классам (модуль ast).

Куски:
- функция верхнего уровня — `path::name`;
- метод — `path::Class.method`, с декораторами;
- остальное тело класса (строка `class …:`, docstring, поля, `class Meta`) —
  `path::Class`; если атрибуты идут и после методов, это несколько частей
  с общим source_id;
- код уровня модуля блоками до MODULE_BLOCK_LINES строк — `path:L<a>-L<b>`.
  Так в индекс попадает `settings/__init__.py`, где почти всё на уровне модуля.
  Блоки из одних импортов пропускаются: для поиска это шум.

Комментарии прямо над кодом входят в кусок: в Paperless они часто объясняют
настройку («# The next 3 settings can also be set using just PAPERLESS_URL»).
Тесты и миграции не индексируются.
"""

import ast
import logging
from pathlib import Path

from assistant.indexing.chunks import Chunk, blob_url

log = logging.getLogger(__name__)

EXCLUDED_DIRS = frozenset({"tests", "migrations"})
MODULE_BLOCK_LINES = 60

_Def = ast.FunctionDef | ast.AsyncFunctionDef


def _start(node: ast.stmt) -> int:
    decorators: list[ast.expr] = getattr(node, "decorator_list", [])
    return min([d.lineno for d in decorators] + [node.lineno])


def _end(node: ast.stmt) -> int:
    return node.end_lineno or node.lineno


def _is_preamble(node: ast.stmt) -> bool:
    """Импорт или строка-docstring."""
    if isinstance(node, ast.Import | ast.ImportFrom):
        return True
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


class _Cutter:
    """Собирает куски одного файла и следит, чтобы комментарии не перекрывали соседа."""

    def __init__(self, source: str, path: str, ref: str, repo_url: str) -> None:
        self.lines = source.splitlines()
        self.path, self.ref, self.repo_url = path, ref, repo_url
        self.chunks: list[Chunk] = []
        self.last_end = 0

    def _with_comments(self, start: int) -> int:
        while start - 1 > self.last_end and self.lines[start - 2].strip().startswith(
            "#"
        ):
            start -= 1
        return start

    def add(self, start: int, end: int, title: str, source_id: str) -> None:
        start = self._with_comments(start)
        self.last_end = end
        self.chunks.append(
            Chunk(
                kind="code",
                path=self.path,
                start_line=start,
                end_line=end,
                title=title,
                breadcrumb=f"{self.path} > {title}",
                source_id=source_id,
                url=blob_url(self.repo_url, self.ref, self.path, start, end),
                site_url=None,
                ref=self.ref,
                text="\n".join(self.lines[start - 1 : end]),
            )
        )

    def module_block(self, block: list[ast.stmt]) -> None:
        # Docstring модуля и импорты в начале блока для поиска — шум.
        while block and _is_preamble(block[0]):
            block = block[1:]
        if not block:
            return
        start, end = self._with_comments(_start(block[0])), _end(block[-1])
        self.add(
            start,
            end,
            f"код модуля (строки {start}-{end})",
            f"{self.path}:L{start}-L{end}",
        )

    def class_def(self, node: ast.ClassDef) -> None:
        """Методы — отдельные куски; подряд идущие не-методы — части `Class`.

        Первая часть начинается со строки `class …:` (вместе с декораторами и
        базовыми классами), даже если сразу за ней идёт метод.
        """
        class_id = f"{self.path}::{node.name}"
        part_start: int | None = _start(node)
        # Пока в части только заголовок класса: он кончается перед первым элементом тела.
        part_end = _start(node.body[0]) - 1

        for item in node.body:
            if not isinstance(item, _Def):
                part_start = part_start if part_start is not None else _start(item)
                part_end = _end(item)
                continue
            if part_start is not None and part_end >= part_start:
                self.add(part_start, part_end, node.name, class_id)
            part_start = None
            qualname = f"{node.name}.{item.name}"
            self.add(_start(item), _end(item), qualname, f"{self.path}::{qualname}")
        if part_start is not None and part_end >= part_start:
            self.add(part_start, part_end, node.name, class_id)


def chunk_python(source: str, *, path: str, ref: str, repo_url: str) -> list[Chunk]:
    cutter = _Cutter(source, path, ref, repo_url)
    block: list[ast.stmt] = []
    for node in ast.parse(source).body:
        if isinstance(node, _Def | ast.ClassDef):
            cutter.module_block(block)
            block = []
            if isinstance(node, ast.ClassDef):
                cutter.class_def(node)
            else:
                cutter.add(_start(node), _end(node), node.name, f"{path}::{node.name}")
            continue
        if block and _end(node) - _start(block[0]) >= MODULE_BLOCK_LINES:
            cutter.module_block(block)
            block = []
        block.append(node)
    cutter.module_block(block)
    return cutter.chunks


def is_indexed(path: Path) -> bool:
    return path.name != "conftest.py" and not EXCLUDED_DIRS & set(path.parts)


def chunk_code(root: Path, *, ref: str, repo_url: str) -> list[Chunk]:
    """Все `src/**/*.py` дерева исходников, кроме тестов и миграций."""
    chunks: list[Chunk] = []
    for file in sorted((root / "src").rglob("*.py")):
        relative = file.relative_to(root)
        if not is_indexed(relative):
            continue
        try:
            chunks.extend(
                chunk_python(
                    file.read_text(encoding="utf-8"),
                    path=relative.as_posix(),
                    ref=ref,
                    repo_url=repo_url,
                )
            )
        except SyntaxError as error:
            log.warning("пропускаю %s: %s", relative, error)
    return chunks
