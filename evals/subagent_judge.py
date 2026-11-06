"""Судья-субагент вместо API: `python -m evals.subagent_judge export|score`.

API Anthropic закрыт (организация отключена), а Sonnet доступен автору как
субагент Claude Code. Судейство вынесено из раннера в файлы:

1. `export RAW` — задание для судьи из сырых данных `make eval`: по каждому
   кейсу (первый прогон) вопрос, утверждения, которые увидел пользователь, с
   процитированными рецептами целиком (как у судьи раннера, `claims_of`), и
   рецепты ответа — для релевантности. Вслепую подмешаны канарейки опоры и
   релевантности: по ним видно, можно ли верить самому судье. Пункты
   перемешаны (сид фиксирован) и поделены на части — по одной на субагента.
   Ключ «пункт → кейс или канарейка» лежит отдельно, судья его не видит.
2. Субагенты (Sonnet, effort high) судят по рубрикам eval — опора на рецепт
   `judge_v10` (без раздела о фактах), релевантность `relevance_v3` — и пишут
   вердикты в JSON.
3. `score RAW` — вердикты против ключа: доля утверждений с нарушением,
   рецепты ответа по вопросу, точность судьи на канарейках; отчёт в
   `evals/reports/<имя>.md`.
"""

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.answer_metrics import Rate
from evals.dataset import Case, load_canaries, load_cases, load_relevance_canaries
from evals.judge import claims_of
from evals.run import REPORTS_DIR, load_answer_run

SEED = 20261106
PARTS = 3


def _case_item(record: dict[str, Any], case: Case) -> dict[str, Any]:
    run = load_answer_run(record)
    claims = claims_of(run)
    chunks = {s["chunk"]["recipe_id"]: s["chunk"] for s in record["sources"]}
    recipes = (
        [
            {
                "n": n,
                "name": chunks[link["recipe_id"]]["name"],
                "ingredients": chunks[link["recipe_id"]]["ingredients"],
            }
            for n, link in enumerate(record["answer"]["sources"], start=1)
            if link["recipe_id"] in chunks
        ]
        if record["answer"]["status"] == "answer"
        else []
    )
    return {
        "question": case.question,
        "dialog": [t.text for t in case.turns if t.role == "user"],
        "statements": [
            {
                "n": n,
                "text": c.text,
                "intro": c.intro,
                "recipes": c.fragments,
            }
            for n, c in enumerate(claims, start=1)
        ],
        "recipes": recipes,
    }


def export(raw: Path, out: Path, parts: int = PARTS) -> list[Path]:
    data = json.loads(raw.read_text(encoding="utf-8"))
    cases = {c.id: c for c in load_cases()}
    items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in data["records"]:
        case = cases.get(record["id"])
        if record["run"] != 0 or case is None:
            continue
        item = _case_item(record, case)
        if item["statements"] or item["recipes"]:
            items.append(
                (item, {"kind": "case", "id": case.id, "category": case.category})
            )
    for c in load_canaries():
        item = {
            "question": "",
            "dialog": [],
            "statements": [
                {"n": 1, "text": c.statement, "intro": False, "recipes": [c.quote]}
            ],
            "recipes": [],
        }
        items.append((item, {"kind": "grounding", "id": c.id, "expected": c.supported}))
    for r in load_relevance_canaries():
        item = {
            "question": r.question,
            "dialog": [],
            "statements": [],
            "recipes": [{"n": 1, "name": r.name, "ingredients": list(r.ingredients)}],
        }
        items.append((item, {"kind": "relevance", "id": r.id, "expected": r.relevant}))
    random.Random(SEED).shuffle(items)
    key: dict[str, dict[str, Any]] = {}
    tasks: list[list[dict[str, Any]]] = [[] for _ in range(parts)]
    for number, (item, meaning) in enumerate(items, start=1):
        item_id = f"P{number:03d}"
        key[item_id] = meaning
        tasks[number % parts].append({"item": item_id, **item})
    out.mkdir(parents=True, exist_ok=True)
    (out / "key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    paths = []
    for part, task in enumerate(tasks, start=1):
        path = out / f"part{part}.json"
        path.write_text(
            json.dumps(task, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        paths.append(path)
    return paths


def score(out: Path, name: str) -> str:
    key: dict[str, dict[str, Any]] = json.loads(
        (out / "key.json").read_text(encoding="utf-8")
    )
    verdicts: dict[str, dict[str, Any]] = {}
    for path in sorted(out.glob("verdicts*.json")):
        for v in json.loads(path.read_text(encoding="utf-8")):
            verdicts[v["item"]] = v
    missing = sorted(set(key) - set(verdicts))
    supported = Counter[str]()
    total = Counter[str]()
    relevant = Counter[str]()
    recipes = Counter[str]()
    grounding_right = grounding_n = relevance_right = relevance_n = 0
    violations: list[str] = []
    for item_id, meaning in key.items():
        v = verdicts.get(item_id)
        if v is None:
            continue
        marks = [s.get("supported") for s in v.get("statements", [])]
        fits = [r.get("relevant") for r in v.get("recipes", [])]
        if meaning["kind"] == "grounding":
            grounding_n += 1
            grounding_right += bool(marks) and marks[0] is meaning["expected"]
        elif meaning["kind"] == "relevance":
            relevance_n += 1
            relevance_right += bool(fits) and fits[0] is meaning["expected"]
        else:
            category = meaning["category"]
            for s in v.get("statements", []):
                total[category] += 1
                if s.get("supported") is True:
                    supported[category] += 1
                else:
                    violations.append(f"- `{meaning['id']}`: {s.get('comment', '')}")
            recipes[category] += len(fits)
            relevant[category] += sum(f is True for f in fits)
    all_total, all_supported = sum(total.values()), sum(supported.values())
    lines = [
        f"# {name}: ответы, судья — субагент Sonnet",
        "",
        "Судья — субагент Claude Code на Sonnet (effort high), рубрики `judge_v10` "
        "(опора, без раздела о фактах) и `relevance_v3`; канарейки подмешаны вслепую.",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| утверждений без нарушений | {Rate(all_supported, all_total)} |",
        f"| рецепты ответа по вопросу | {Rate(sum(relevant.values()), sum(recipes.values()))} |",
        f"| судья на канарейках опоры | {Rate(grounding_right, grounding_n)} |",
        f"| судья на канарейках релевантности | {Rate(relevance_right, relevance_n)} |",
        f"| пунктов без вердикта | {len(missing)} |",
        "",
        "| Категория | утверждения без нарушений | рецепты по вопросу |",
        "|---|---|---|",
    ]
    for category in sorted(total | recipes):
        lines.append(
            f"| {category} | {Rate(supported[category], total[category])} | "
            f"{Rate(relevant[category], recipes[category])} |"
        )
    lines += ["", "## Нарушения", "", *violations]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.subagent_judge")
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export", help="задание для судьи-субагента")
    exp.add_argument("raw", type=Path)
    exp.add_argument("out", type=Path)
    exp.add_argument("--parts", type=int, default=PARTS)
    sco = commands.add_parser("score", help="вердикты против ключа → отчёт")
    sco.add_argument("out", type=Path)
    sco.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        for path in export(args.raw, args.out, args.parts):
            print(path)
        return 0
    text = score(args.out, args.name)
    path = REPORTS_DIR / f"{args.name}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
