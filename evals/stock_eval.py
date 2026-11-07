"""Подбор «из того, что есть» на кейсах stock — `python -m evals.stock_eval run|score`.

`run` — по каждому кейсу `evals/dataset/stock/` ответ ассистента в режиме
склада: демо-склад из сида (тот же, что на сервере), разбор и ответ модели
ассистента, строгий и расширенный поиск. Метрики считает код:
- нашлось ли что-то целиком из склада;
- у строгих рецептов все ингредиенты — со склада или из базовых (должно быть
  1.00: это фильтр Qdrant);
- условия вопроса («с рыбой», «без лука») у строгих рецептов;
- предложен ли расширенный поиск и сколько у его рецептов не хватает (1–4);
- ответ: статус и утверждения после проверки цитат кодом.
Сырые данные — `raw/<дата>-stock.json`, задание судье — `task.json` в
каталоге `--task`: вопрос, рецепты (строгие и расширенные вперемешку, без
пометок) и утверждения ответа с рецептами целиком.

`score` — вердикты судьи-субагента (Sonnet, рубрики `judge_v10` — опора на
рецепт и `relevance_v3` — рецепт по вопросу) → отчёт `<дата>-stock.md`.
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.generation.answer import AnswerResult, StockRequest, make_answerer
from assistant.generation.cloudru import cost_rub
from assistant.generation.llm import llm_model, make_http_client
from assistant.grocy.seed import DEMO_STOCK
from assistant.retrieval.stock import MAX_MISSING, STAPLES
from evals.answer_metrics import Rate
from evals.dataset import Case, load_stock_cases
from evals.judge import Claim, claim_of
from evals.retrieval_metrics import fits, history
from evals.run import REPORTS_DIR, _clock_text

RAW = REPORTS_DIR / "raw"
STOCK_NAMES = tuple(item.name for item in DEMO_STOCK)


def _recipe(chunk: Any, missing: Sequence[str] = ()) -> dict[str, Any]:
    row = {
        "recipe_id": chunk.recipe_id,
        "name": chunk.name,
        "ingredients": chunk.ingredients,
    }
    return {**row, "missing": list(missing)} if missing else row


def record(case: Case, result: AnswerResult, seconds: float) -> dict[str, Any]:
    """Всё о кейсе: строгие и расширенные рецепты, проверки кодом, утверждения."""
    trace, answer = result.trace, result.answer
    allowed = {*STOCK_NAMES, *STAPLES}
    strict = [hit.chunk for hit in trace.hits]
    claims: list[Claim] = [claim_of(s, trace.sources) for s in trace.checked if s.kept]
    if answer.status == "answer" and answer.message:
        fragments = list(dict.fromkeys(f for c in claims for f in c.fragments))
        claims.append(Claim(answer.message, fragments, intro=True))
    return {
        "id": case.id,
        "question": case.question,
        "dialog": [turn.text for turn in case.turns if turn.role == "user"],
        "status": answer.status,
        "reason": answer.reason,
        "strict": [_recipe(chunk) for chunk in strict],
        "strict_pure": all(set(chunk.ingredients) <= allowed for chunk in strict),
        "strict_fits": (
            [fits(case, chunk.name, chunk.ingredients) for chunk in strict]
            # has_conditions смотрит категории основного набора; здесь — сами условия.
            if case.expected_include or case.expected_exclude or case.expected_name
            else None
        ),
        "near_available": answer.near_available,
        "near": [_recipe(chunk, missing) for chunk, missing in trace.stock_near],
        "statements": [
            {"n": n, "text": c.text, "intro": c.intro, "recipes": c.fragments}
            for n, c in enumerate(claims, start=1)
        ],
        "seconds": seconds,
        "error": "",
    }


def run(cases: Sequence[Case], workers: int) -> tuple[list[dict[str, Any]], str, str]:
    settings = Settings()
    model = llm_model(settings)
    with httpx2.Client(timeout=30) as http, make_http_client(settings) as llm_http:
        answerer = make_answerer(
            settings,
            embeddings_http=http,
            llm_http=llm_http,
            qdrant=QdrantClient(url=settings.qdrant_url, timeout=30),
        )
        started = time.monotonic()
        done: list[str] = []

        def one(case: Case) -> dict[str, Any]:
            begun = time.monotonic()
            try:
                result = answerer.answer(
                    case.question, history(case), stock=StockRequest(names=STOCK_NAMES)
                )
                row = record(case, result, time.monotonic() - begun)
            except Exception as error:  # один кейс не роняет прогон
                row = {
                    "id": case.id,
                    "error": f"{type(error).__name__}: {str(error)[:200]}",
                }
            done.append(case.id)
            elapsed = time.monotonic() - started
            left = elapsed / len(done) * (len(cases) - len(done))
            verdict = row["error"] or (
                f"строго {len(row['strict'])}, расширенный {len(row['near'])}, {row['status']}"
            )
            print(
                f"[{len(done)}/{len(cases)}] {case.id}: {verdict} · прошло "
                f"{_clock_text(elapsed)} · осталось ≈ {_clock_text(left)}",
                file=sys.stderr,
                flush=True,
            )
            return row

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            rows = list(pool.map(one, cases))
        rub = cost_rub(model, answerer._llm.stats)
    return rows, model, "—" if rub is None else f"≈ {rub:.0f} ₽"


def task(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Задание судье: без пометок «строгий / расширенный» и без «не хватает»."""
    items = []
    for row in rows:
        if row.get("error"):
            continue
        recipes = [*row["strict"], *row["near"]]
        items.append(
            {
                "item": row["id"],
                "question": row["question"],
                "dialog": row["dialog"],
                "recipes": [
                    {"n": n, "name": r["name"], "ingredients": r["ingredients"]}
                    for n, r in enumerate(recipes, start=1)
                ],
                "statements": row["statements"],
            }
        )
    return items


def render(
    rows: Sequence[dict[str, Any]], meta: dict[str, Any], verdicts: dict[str, Any]
) -> str:
    ok = [r for r in rows if not r.get("error")]
    found = [r for r in ok if r["strict"]]
    fits_all = [f for r in ok if r["strict_fits"] is not None for f in r["strict_fits"]]
    missing = [len(n["missing"]) for r in ok for n in r["near"]]
    seconds = [r["seconds"] for r in ok]
    lines = [
        f"# Подбор «из того, что есть» ({meta['date']})",
        "",
        f"Кейсы `evals/dataset/stock/` ({len(rows)}, от независимого субагента), демо-склад "
        f"из сида ({len(STOCK_NAMES)} продуктов + базовые: {', '.join(STAPLES)}). Модель — "
        f"{meta['model']}; один прогон. Судья — субагент Sonnet (judge_v10, relevance_v3).",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| нашлось целиком из склада | {Rate(len(found), len(ok))} |",
        f"| у строгих рецептов всё со склада или базовое | "
        f"{Rate(sum(r['strict_pure'] for r in ok), len(ok))} |",
        f"| условия вопроса у строгих рецептов | {Rate(sum(fits_all), len(fits_all))} |",
        f"| предложен расширенный поиск | {Rate(sum(r['near_available'] for r in ok), len(ok))} |",
        f"| у расширенных не хватает 1–{MAX_MISSING} | "
        f"{Rate(sum(1 <= m <= MAX_MISSING for m in missing), len(missing))} |",
        f"| ответ со ссылками (status answer) | {Rate(sum(r['status'] == 'answer' for r in ok), len(ok))} |",
        f"| сбои | {len(rows) - len(ok)} |",
        f"| время на вопрос, медиана | {statistics.median(seconds) if seconds else 0:.0f} с |",
        f"| цена прогона (кэш промпта — по полной цене) | {meta['cost']} |",
    ]
    if verdicts:
        strict_ids = {r["id"]: len(r["strict"]) for r in ok}
        relevant_strict = relevant_near = total_strict = total_near = 0
        supported = statements = 0
        for item_id, v in verdicts.items():
            border = strict_ids.get(item_id, 0)
            for mark in v.get("recipes", []):
                if mark["n"] <= border:
                    total_strict += 1
                    relevant_strict += mark.get("relevant") is True
                else:
                    total_near += 1
                    relevant_near += mark.get("relevant") is True
            for mark in v.get("statements", []):
                statements += 1
                supported += mark.get("supported") is True
        lines += [
            f"| строгие рецепты по вопросу (судья) | {Rate(relevant_strict, total_strict)} |",
            f"| расширенные рецепты по вопросу (судья) | {Rate(relevant_near, total_near)} |",
            f"| утверждения ответа без нарушений (судья) | {Rate(supported, statements)} |",
        ]
    lines += [
        "",
        "## Кейсы",
        "",
        "| кейс | строго | расширенный | статус | что не так |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| `{r['id']}` | — | — | сбой | {r['error'][:80]} |")
            continue
        v = verdicts.get(r["id"], {})
        notes = [
            m.get("comment", "")
            for m in v.get("statements", [])
            if m.get("supported") is not True
        ]
        notes += [
            m.get("comment", "")
            for m in v.get("recipes", [])
            if m.get("relevant") is not True
        ]
        lines.append(
            f"| `{r['id']}` | {len(r['strict'])} | {len(r['near'])} | {r['status']} | "
            + "; ".join(n.replace("|", "/") for n in notes)[:200]
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.stock_eval", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    go = commands.add_parser("run", help="прогон по кейсам stock")
    go.add_argument(
        "--task", type=Path, required=True, help="куда положить задание судье"
    )
    go.add_argument("--workers", type=int, default=4)
    sc = commands.add_parser("score", help="вердикты судьи → отчёт")
    sc.add_argument("raw", type=Path)
    sc.add_argument("task", type=Path, help="каталог с verdicts*.json")
    args = parser.parse_args(argv)
    today = date.today()
    report = REPORTS_DIR / f"{today}-stock.md"
    if args.command == "run":
        cases = load_stock_cases()
        rows, model, cost = run(cases, args.workers)
        meta = {"date": str(today), "model": model, "cost": cost}
        raw = RAW / f"{today}-stock.json"
        raw.write_text(
            json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        args.task.mkdir(parents=True, exist_ok=True)
        (args.task / "task.json").write_text(
            json.dumps(task(rows), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        report.write_text(render(rows, meta, {}), encoding="utf-8")
        print(report.read_text(encoding="utf-8"))
        print(
            f"сырые данные: {raw}; задание судье: {args.task / 'task.json'}",
            file=sys.stderr,
        )
        return 0
    data = json.loads(args.raw.read_text(encoding="utf-8"))
    verdicts: dict[str, Any] = {}
    for path in sorted(args.task.glob("verdicts*.json")):
        for v in json.loads(path.read_text(encoding="utf-8")):
            verdicts[v["item"]] = v
    report.write_text(render(data["rows"], data["meta"], verdicts), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
