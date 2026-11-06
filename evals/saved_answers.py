"""Фикстура CI-гейта из сырых данных прогона: `python -m evals.saved_answers RAW_JSON`.

Ключей API в CI нет, поэтому точность цитат на каждом PR проверяется на
сохранённых ответах модели: `tests/test_saved_answers.py` прогоняет их через
`citations.py` и сверяет, какие утверждения прошли, с тем, что было в прогоне.

В фикстуре — только сырой ответ модели и id рецептов, которые она видела, в
том же порядке. Тексты рецептов тест собирает из датасета тем же кодом, что и
ассистент: датасет без лицензии в репозиторий не кладём (раздел «Данные» в
README), а CI скачивает его с проверкой sha256.

Выборка: все ответы с отброшенными утверждениями (проверка «нет» у кода),
по паре болтовни и «не нашёл» (разбор формата) и случайные ответы — сид фиксирован.
"""

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "saved_answers.json"
SEED = 20261106
ANSWERS = 24


def pick(
    records: Sequence[dict[str, Any]], size: int = ANSWERS, seed: int = SEED
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ran = [r for r in records if r["finish_reason"] == "stop" and r["content"]]
    dropped = [
        r
        for r in ran
        if any(
            s["unknown_numbers"] or all(c["error"] for c in s["citations"])
            for s in r["checked"]
        )
    ]
    chat = [r for r in ran if r["answer"]["status"] == "chat"]
    refused = [r for r in ran if r["answer"]["reason"] == "model_not_found"]
    rest = [r for r in ran if r["answer"]["status"] == "answer" and r not in dropped]
    picked = (
        dropped
        + rng.sample(chat, min(2, len(chat)))
        + rng.sample(refused, min(2, len(refused)))
    )
    picked += rng.sample(rest, max(0, min(size - len(picked), len(rest))))
    return sorted(picked, key=lambda r: (r["id"], r["run"]))


def fixture_record(record: dict[str, Any]) -> dict[str, Any]:
    sources = sorted(record["sources"], key=lambda s: s["number"])
    return {
        "id": record["id"],
        "run": record["run"],
        "recipe_ids": [s["chunk"]["recipe_id"] for s in sources],
        "content": record["content"],
        "finish_reason": record["finish_reason"],
        "expected": {
            "status": record["answer"]["status"],
            "kept": [
                not s["unknown_numbers"]
                and any(c["error"] is None for c in s["citations"])
                for s in record["checked"]
            ],
        },
    }


def build(raw: dict[str, Any], source: str) -> dict[str, Any]:
    first = raw["records"][0]["answer"]
    return {
        "source": source,
        "dataset_revision": first["dataset_revision"],
        "prompt_version": first["prompt_version"],
        "answers": [fixture_record(r) for r in pick(raw["records"])],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.saved_answers", description=__doc__
    )
    parser.add_argument("raw", type=Path)
    args = parser.parse_args(argv)
    data = build(json.loads(args.raw.read_text(encoding="utf-8")), args.raw.name)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    dropped = sum(not all(a["expected"]["kept"]) for a in data["answers"])
    print(
        f"{FIXTURE}: {len(data['answers'])} ответов, с отброшенными утверждениями {dropped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
