"""Шаблоны текстов ошибок и логов из кода Paperless (модуль ast).

Пользователь присылает строку из лога или ошибку задачи, например
«Not consuming scan.pdf: Given ASN 42 already exists!». Векторный поиск такую
строку почти не находит: она короткая и полна подстановок. Поэтому при
индексации из кода извлекаются шаблоны сообщений, а поиск потом проверяет,
какой шаблон совпал с текстом вопроса.

Откуда берутся сообщения:
- `raise X("…")`;
- `log.warning("…")` и другие уровни логгера;
- `self._fail(короткое, "подробное")` — ошибка обработки документа;
- строки, присвоенные переменной, которая затем уходит в такой вызов
  (`log_msg = f"…"`, `log_msg += "…"`), в пределах одной функции.

Шаблон — постоянные куски текста по порядку. Между ними «дырки»: подстановки
f-строк `{…}`, `%s` у логгера, `{}` у `.format()`. Шаблоны, где постоянного
текста меньше MIN_LITERAL символов, не берутся: «Error: {e}» совпадёт
с чем угодно.
"""

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

MIN_LITERAL = 15
LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical"}
)
_GETTEXT = frozenset({"_", "gettext", "gettext_lazy", "ngettext", "pgettext"})
# %s, %d, %r, %(name)s и т. п. — подстановки логгера; %% — просто знак процента.
_PERCENT = re.compile(r"%(?:\([^)]*\))?[-#0 +]*\d*(?:\.\d+)?[sdirfxXeEgGc]")
_FORMAT_FIELD = re.compile(r"\{[^{}]*\}")


class Message(BaseModel):
    """Шаблон сообщения: где в файле и какой у него постоянный текст."""

    model_config = ConfigDict(frozen=True)

    # Строка, с которой начинается текст сообщения.
    line: int
    # raise | log.warning | log.error | … | _fail
    kind: str
    # Постоянные куски по порядку; между соседними — подстановка.
    parts: list[str]
    # Есть ли подстановка до первого куска и после последнего:
    # f"Unsupported mime type {mime_type}" — дырка в конце.
    head_hole: bool = False
    tail_hole: bool = False

    @property
    def literal_length(self) -> int:
        return sum(len(part.strip()) for part in self.parts)


def _split(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Куски текста вокруг подстановок; пустая строка между ними — сама подстановка."""
    parts: list[str] = []
    for index, piece in enumerate(pattern.split(text)):
        if index:
            parts.append("")
        if piece:
            parts.append(piece)
    return parts


def template_parts(node: ast.expr) -> list[str] | None:
    """Постоянные куски текста выражения или None, если текста там нет.

    Пустая строка в списке — граница: подстановка, после которой кусок
    не продолжается. Соседние куски склеиваются в `_merge`.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if "%" not in node.value:
            return [node.value]
        return [part.replace("%%", "%") for part in _split(node.value, _PERCENT)]
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("")  # подстановка {…}
        return parts
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _GETTEXT
        and node.args
    ):
        return template_parts(node.args[0])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
    ):
        return _split(node.func.value.value, _FORMAT_FIELD)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return template_parts(node.left)  # "…%s…" % value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # "Execute: " + " ".join(args): часть без текста — тоже подстановка.
        left = template_parts(node.left)
        right = template_parts(node.right)
        if left is None and right is None:
            return None
        return [
            *(left if left is not None else [""]),
            *(right if right is not None else [""]),
        ]
    return None


def _merge(parts: list[str]) -> list[str]:
    """Склеивает соседние постоянные куски и выкидывает пустые границы.

    `f"Given ASN " f"{asn} already"` даёт ["Given ASN ", "", " already"]:
    два куска с дыркой между ними.
    """
    merged: list[str] = []
    current = ""
    for part in parts:
        if part == "":
            if current.strip():
                merged.append(current)
            current = ""
        else:
            current += part
    if current.strip():
        merged.append(current)
    return merged


@dataclass(frozen=True)
class _Call:
    """Вызов, который выводит сообщение: что именно выводится и как."""

    kind: str
    argument: ast.expr


def _message_calls(node: ast.AST) -> Iterator[_Call]:
    if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and node.exc.args:
        yield _Call("raise", node.exc.args[0])
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in LOG_METHODS and node.args:
            yield _Call(f"log.{node.func.attr}", node.args[0])
        elif node.func.attr == "_fail" and len(node.args) > 1:
            # Первый аргумент — короткий код для интерфейса, текст — второй.
            yield _Call("_fail", node.args[1])


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Узлы функции или модуля без вложенных функций и классов: у них своя область."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ):
            stack.extend(ast.iter_child_nodes(node))


def _scopes(tree: ast.Module) -> Iterator[ast.AST]:
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node


@dataclass(frozen=True)
class Extraction:
    messages: list[Message]
    # Сколько вызовов с текстом сообщения нашлось и у скольких извлёкся шаблон.
    calls: int
    covered: int


def extract_messages(source: str) -> Extraction:
    """Шаблоны сообщений файла, по одному на текст, в порядке строк."""
    tree = ast.parse(source)
    found: dict[tuple[int, tuple[str, ...]], Message] = {}
    calls = covered = 0
    for scope in _scopes(tree):
        nodes = list(_scope_nodes(scope))
        # Что присваивается переменным в этой области: имя → [(узел значения)].
        assigned: dict[str, list[ast.expr]] = {}
        for node in nodes:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, []).append(node.value)
            elif (
                isinstance(node, ast.AugAssign | ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                assigned.setdefault(node.target.id, []).append(node.value)
        for node in nodes:
            for call in _message_calls(node):
                calls += 1
                values = (
                    assigned.get(call.argument.id, [])
                    if isinstance(call.argument, ast.Name)
                    else [call.argument]
                )
                templates = []
                for value in values:
                    parts = template_parts(value)
                    if parts is None:
                        continue
                    message = Message(
                        line=value.lineno,
                        kind=call.kind,
                        parts=_merge(parts),
                        head_hole=bool(parts) and parts[0] == "",
                        tail_hole=bool(parts) and parts[-1] == "",
                    )
                    if message.literal_length >= MIN_LITERAL:
                        templates.append(message)
                if templates:
                    covered += 1
                for message in templates:
                    found.setdefault((message.line, tuple(message.parts)), message)
    messages = sorted(found.values(), key=lambda m: (m.line, m.parts))
    return Extraction(messages=messages, calls=calls, covered=covered)
