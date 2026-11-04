"""Сохранённые ответы модели для гейта CI: `python -m evals.fixtures RAW_JSON`.

Точность ссылок должна блокировать PR (CLAUDE.md, «Правила прогона»), но
ключа GigaChat в CI нет. Поэтому CI перепроверяет ответы, которые модель
уже дала на eval. Если правка `citations.py` начнёт отвергать настоящие
цитаты или принимать выдуманные, тест `tests/test_saved_answers.py` упадёт.

Куски документации и кода Paperless в фикстуре не хранятся: репозиторий
их не копирует (README, «Лицензия»). Хранятся координаты источников
(путь и строки) и сырой ответ модели. В ответе модели есть её цитаты —
короткие фрагменты, до ~40 слов, как в любом ответе ассистента. Тексты
кусков тест восстанавливает нашим же нарезчиком из исходников тега
(`make sources`, в CI — шаг перед тестами).

Выгружается первый прогон каждого кейса, где модель ответила по существу,
без отбора: фикстура показывает модель такой, какая она есть.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from assistant.generation.answer import prepare_sources
from assistant.generation.citations import (
    CheckedStatement,
    ModelAnswer,
    SourceText,
    check_answer,
)
from assistant.indexing.build import collect_chunks
from assistant.indexing.chunks import Chunk
from assistant.indexing.sources import SourceTree
from assistant.retrieval.store import Hit

FIXTURES_FILE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "answers.json"
)


def export(raw: dict[str, Any], report: str) -> dict[str, Any]:
    answers = []
    seen: set[str] = set()
    for record in raw["records"]:
        # Только ответы по существу: у них есть утверждения с цитатами.
        if record["id"] in seen or not record["checked"]:
            continue
        seen.add(record["id"])
        answers.append(
            {
                "id": record["id"],
                "model": record["answer"]["model"],
                "sources": [
                    {
                        "number": s["number"],
                        "path": s["chunk"]["path"],
                        "start_line": s["chunk"]["start_line"],
                        "end_line": s["chunk"]["end_line"],
                    }
                    for s in record["sources"]
                ],
                "content": record["content"],
                "valid": [
                    [c["error"] is None for c in s["citations"]]
                    for s in record["checked"]
                ],
            }
        )
    return {
        "ref": raw["meta"]["manifest"]["ref"],
        "report": report,
        "answers": answers,
    }


def chunks_by_lines(
    tree: SourceTree, repo_url: str
) -> dict[tuple[str, int, int], Chunk]:
    return {
        (c.path, c.start_line, c.end_line): c for c in collect_chunks(tree, repo_url)
    }


def rebuild_sources(
    answer: dict[str, Any], chunks: dict[tuple[str, int, int], Chunk]
) -> list[SourceText]:
    """Тексты источников ровно такие, какие видела модель: тот же нарезчик и то же маскирование."""
    ordered = sorted(answer["sources"], key=lambda s: s["number"])
    hits = [
        Hit(chunks[(s["path"], s["start_line"], s["end_line"])], 0.0) for s in ordered
    ]
    return prepare_sources(hits)


def recheck(
    answer: dict[str, Any], chunks: dict[tuple[str, int, int], Chunk]
) -> list[CheckedStatement]:
    parsed = ModelAnswer.model_validate_json(answer["content"])
    return check_answer(parsed, rebuild_sources(answer, chunks))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.fixtures", description=__doc__
    )
    parser.add_argument(
        "raw", type=Path, help="сырые данные прогона из evals/reports/raw/"
    )
    parser.add_argument(
        "--report", required=True, help="отчёт этого прогона, для ссылки"
    )
    args = parser.parse_args(argv)

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    fixtures = export(raw, args.report)
    FIXTURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES_FILE.write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(fixtures['answers'])} ответов в {FIXTURES_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
