from itertools import pairwise
from pathlib import Path

from assistant.indexing.chunks import Chunk
from assistant.indexing.code import (
    MODULE_BLOCK_LINES,
    chunk_code,
    chunk_python,
    is_indexed,
)

PATH = "src/documents/example.py"

SOURCE = '''\
"""Модуль-пример."""

import os
from pathlib import Path

# Порог по умолчанию берётся из окружения.
THRESHOLD = int(os.getenv("PAPERLESS_THRESHOLD", "5"))


def helper(value):
    return value * 2


class Consumer(Base):
    """Обрабатывает документы."""

    name = "consumer"

    # Главный метод.
    @property
    def ready(self):
        return True

    async def run(self, path):
        def inner():
            return path
        return inner()

    class Meta:
        ordering = ["id"]
'''


def chunks_by_id(source: str = SOURCE) -> dict[str, list[Chunk]]:
    result: dict[str, list[Chunk]] = {}
    for chunk in chunk_python(
        source, path=PATH, ref="v3.4.2", repo_url="https://github.com/x/fork"
    ):
        result.setdefault(chunk.source_id, []).append(chunk)
    return result


def test_functions_methods_and_classes_become_chunks() -> None:
    ids = list(chunks_by_id())

    assert ids == [
        f"{PATH}:L6-L7",
        f"{PATH}::helper",
        f"{PATH}::Consumer",
        f"{PATH}::Consumer.ready",
        f"{PATH}::Consumer.run",
    ]


def test_import_only_block_is_skipped_and_comment_above_code_is_kept() -> None:
    module = chunks_by_id()[f"{PATH}:L6-L7"][0]

    assert module.start_line == 6
    assert module.text.startswith("# Порог по умолчанию")
    assert "import os" not in module.text


def test_method_includes_decorator_comment_and_nested_function() -> None:
    ready = chunks_by_id()[f"{PATH}::Consumer.ready"][0]
    run = chunks_by_id()[f"{PATH}::Consumer.run"][0]

    assert ready.text.splitlines()[:2] == ["    # Главный метод.", "    @property"]
    assert "def inner" in run.text
    assert run.title == "Consumer.run"
    assert run.breadcrumb == f"{PATH} > Consumer.run"


def test_class_attributes_before_and_after_methods_are_kept() -> None:
    parts = chunks_by_id()[f"{PATH}::Consumer"]

    assert len(parts) == 2
    assert parts[0].text.startswith("class Consumer(Base):")
    assert 'name = "consumer"' in parts[0].text
    assert "class Meta:" in parts[1].text


def test_class_that_starts_with_method_keeps_its_signature() -> None:
    source = "class Plugin(Base):\n    def run(self):\n        pass\n"

    header = chunks_by_id(source)[f"{PATH}::Plugin"][0]

    assert header.text == "class Plugin(Base):"


def test_long_module_level_code_is_split_into_blocks() -> None:
    source = (
        "\n".join(f"SETTING_{i} = {i}" for i in range(MODULE_BLOCK_LINES * 2 + 5))
        + "\n"
    )

    blocks = list(chunks_by_id(source).values())

    assert len(blocks) == 3
    assert all(len(b[0].text.splitlines()) <= MODULE_BLOCK_LINES + 1 for b in blocks)


def test_chunks_do_not_overlap_and_match_file_lines() -> None:
    chunks = [c for parts in chunks_by_id().values() for c in parts]
    lines = SOURCE.splitlines()

    ordered = sorted(chunks, key=lambda c: c.start_line)
    for previous, current in pairwise(ordered):
        assert current.start_line > previous.end_line
    for chunk in chunks:
        assert "\n".join(lines[chunk.start_line - 1 : chunk.end_line]) == chunk.text
        assert chunk.url.endswith(f"#L{chunk.start_line}-L{chunk.end_line}")
        assert chunk.site_url is None


def test_tests_migrations_and_conftest_are_not_indexed() -> None:
    assert is_indexed(Path("src/documents/consumer.py"))
    assert not is_indexed(Path("src/documents/tests/test_consumer.py"))
    assert not is_indexed(Path("src/documents/migrations/0001_initial.py"))
    assert not is_indexed(Path("src/conftest.py"))


def test_file_with_syntax_error_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "src/pkg").mkdir(parents=True)
    (tmp_path / "src/pkg/good.py").write_text("def ok():\n    return 1\n")
    (tmp_path / "src/pkg/broken.py").write_text("def broken(:\n")

    chunks = chunk_code(tmp_path, ref="v3.4.2", repo_url="https://github.com/x/fork")

    assert [c.source_id for c in chunks] == ["src/pkg/good.py::ok"]
