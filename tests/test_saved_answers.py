"""Гейт CI: ссылки в сохранённых ответах модели перепроверяются текущим кодом.

Ключа GigaChat в CI нет, поэтому проверяются ответы, которые модель уже
дала на eval (`tests/fixtures/answers.json`, выгрузка — `evals/fixtures.py`).
Тексты источников восстанавливаются из исходников Paperless на теге.
Локально без них тест пропускается (`make sources`), в CI он обязателен:
там задано REQUIRE_PAPERLESS_SOURCES=1.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from assistant.indexing.chunks import Chunk
from assistant.indexing.sources import SOURCE_META_FILE, SourceTree
from evals.answer_metrics import Rate
from evals.fixtures import FIXTURES_FILE, chunks_by_lines, export, recheck
from evals.report import ANSWER_THRESHOLDS

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/MichaelZhdanov/paperless-ngx-rag"


def fixtures() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURES_FILE.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def chunks() -> dict[tuple[str, int, int], Chunk]:
    ref = fixtures()["ref"]
    root = ROOT / "data" / "paperless-src" / ref
    if not (root / SOURCE_META_FILE).exists():
        message = f"нет исходников Paperless {ref}: выполните make sources"
        if os.environ.get("REQUIRE_PAPERLESS_SOURCES") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return chunks_by_lines(SourceTree(root=root, ref=ref, commit=None), REPO_URL)


def test_saved_answers_get_the_same_verdicts(
    chunks: dict[tuple[str, int, int], Chunk],
) -> None:
    """Правка проверки ссылок не должна молча начать принимать выдумки или отвергать цитаты."""
    changed = []
    for answer in fixtures()["answers"]:
        valid = [[c.valid for c in s.citations] for s in recheck(answer, chunks)]
        if valid != answer["valid"]:
            changed.append(f"{answer['id']}: было {answer['valid']}, стало {valid}")

    assert not changed, "вердикты по цитатам изменились:\n" + "\n".join(changed)


def test_citation_accuracy_of_saved_answers_meets_threshold(
    chunks: dict[tuple[str, int, int], Chunk],
) -> None:
    marks = [
        c.valid
        for answer in fixtures()["answers"]
        for s in recheck(answer, chunks)
        for c in s.citations
    ]
    accuracy = Rate(sum(marks), len(marks))

    assert accuracy.value >= ANSWER_THRESHOLDS["citation_accuracy"], str(accuracy)


def test_export_keeps_coordinates_not_texts() -> None:
    record = {
        "id": "howto-001",
        "answer": {"model": "GigaChat-2-Max:1"},
        "content": '{"status": "answer", "statements": []}',
        "sources": [
            {
                "number": 1,
                "chunk": {"path": "docs/usage.md", "start_line": 3, "end_line": 9},
                "text": "текст документации",
            }
        ],
        "checked": [
            {"citations": [{"error": None}, {"error": "фрагмента нет в источнике"}]}
        ],
    }
    refusal = {**record, "id": "out_of_scope-001", "checked": []}
    raw = {
        "meta": {"manifest": {"ref": "v3.4.2"}},
        "records": [record, record, refusal],
    }

    data = export(raw, "evals/reports/x.md")

    assert data["ref"] == "v3.4.2"
    assert [a["id"] for a in data["answers"]] == ["howto-001"]
    assert data["answers"][0]["valid"] == [[True, False]]
    assert "текст документации" not in json.dumps(data, ensure_ascii=False)
