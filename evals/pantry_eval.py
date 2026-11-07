"""Перечень «что есть дома» в вопросе — `python -m evals.pantry_eval run|score`.

Кейсы `evals/dataset/pantry/`: пользователь перечисляет, что есть в
холодильнике, и просит подобрать блюда. Перечень — то, из чего выбирать, а не
обязательный состав (recipes_v6). Проверяется сама эта разница: разбор вопроса
и поиск — один раз на кейс (`CachingPlanner`), ответ пишут две версии промпта
на одних и тех же найденных рецептах.

Метрики кодом, по версиям: доля ответов (не «не нашёл»), показанные рецепты
по условиям кейса (`fits`: хотя бы один главный продукт перечня, ни одного
«без»), утверждения после проверки цитат. По смыслу — судья-субагент Sonnet
локально (рубрики `judge_v10` и `relevance_v3`, как в stock_eval): задание —
`task.json` в каталоге `--task`, вердикты — `verdicts*.json` там же.
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
from assistant.generation.answer import Answerer, AnswerResult
from assistant.generation.cloudru import cost_rub
from assistant.generation.gigachat import Message
from assistant.generation.llm import llm_model, make_chat_model, make_http_client
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from assistant.tools.search_recipes import SearchArgs, SearchPlanner
from evals.answer_metrics import Rate
from evals.dataset import Case, load_pantry_cases
from evals.judge import Claim, claim_of
from evals.retrieval_metrics import fits, history
from evals.run import REPORTS_DIR, _clock_text

RAW = REPORTS_DIR / "raw"
PROMPTS = ("recipes_v5", "recipes_v6")


class CachingPlanner:
    """Один разбор вопроса на кейс: обе версии ответа получают одинаковый поиск."""

    def __init__(self, planner: SearchPlanner) -> None:
        self._planner = planner
        self._plans: dict[str, SearchArgs] = {}

    def plan(
        self, question: str, history: Sequence[Message] = (), note: str = ""
    ) -> SearchArgs:
        key = json.dumps([question, list(history), note], ensure_ascii=False)
        if key not in self._plans:
            self._plans[key] = self._planner.plan(question, history, note)
        return self._plans[key]


def record(case: Case, result: AnswerResult, seconds: float) -> dict[str, Any]:
    """Одна версия ответа: статус, показанные рецепты и их условия, утверждения."""
    trace, answer = result.trace, result.answer
    by_id = {s.chunk.recipe_id: s.chunk for s in trace.sources}
    shown = [
        by_id[link.recipe_id] for link in answer.sources if link.recipe_id in by_id
    ]
    claims: list[Claim] = [claim_of(s, trace.sources) for s in trace.checked if s.kept]
    if answer.status == "answer" and answer.message:
        fragments = list(dict.fromkeys(f for c in claims for f in c.fragments))
        claims.append(Claim(answer.message, fragments, intro=True))
    return {
        "status": answer.status,
        "reason": answer.reason,
        "message": answer.message,
        "shown": [
            {"recipe_id": c.recipe_id, "name": c.name, "ingredients": c.ingredients}
            for c in shown
        ],
        "fits": [fits(case, c.name, c.ingredients) for c in shown],
        "statements": [
            {"n": n, "text": c.text, "intro": c.intro, "recipes": c.fragments}
            for n, c in enumerate(claims, start=1)
        ],
        "kept": sum(s.kept for s in trace.checked),
        "dropped": sum(not s.kept for s in trace.checked),
        "seconds": seconds,
    }


def run(cases: Sequence[Case], workers: int) -> tuple[list[dict[str, Any]], str, str]:
    settings = Settings()
    model = llm_model(settings)
    with httpx2.Client(timeout=30) as http, make_http_client(settings) as llm_http:
        retriever = Retriever(
            VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30)),
            make_embedder(settings, http),
            ref=settings.dataset_revision,
        )
        llm = make_chat_model(settings, llm_http)
        planner = CachingPlanner(
            SearchPlanner(llm, model=model, vocabulary=retriever.vocabulary())
        )
        answerers = {
            prompt: Answerer(
                retriever,
                llm,
                model=model,
                ref=settings.dataset_revision,
                prompt_version=prompt,
                planner=planner,
            )
            for prompt in PROMPTS
        }
        started = time.monotonic()
        done: list[str] = []

        def one(case: Case) -> dict[str, Any]:
            row: dict[str, Any] = {
                "id": case.id,
                "question": case.question,
                "dialog": [t.text for t in case.turns if t.role == "user"],
                "error": "",
            }
            try:
                for prompt, answerer in answerers.items():
                    begun = time.monotonic()
                    result = answerer.answer(case.question, history(case))
                    row[prompt] = record(case, result, time.monotonic() - begun)
            except Exception as error:  # один кейс не роняет прогон
                row["error"] = f"{type(error).__name__}: {str(error)[:200]}"
            done.append(case.id)
            elapsed = time.monotonic() - started
            left = elapsed / len(done) * (len(cases) - len(done))
            verdict = row["error"] or ", ".join(
                f"{p[-2:]}: {row[p]['status']}" for p in PROMPTS
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
        rub = cost_rub(model, llm.stats)
    return rows, model, "—" if rub is None else f"≈ {rub:.0f} ₽"


def task(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Задание судье: обе версии вперемешку, без пометки версии (item — шифр)."""
    items = []
    for row in rows:
        if row.get("error"):
            continue
        for n, prompt in enumerate(PROMPTS):
            answer = row[prompt]
            if answer["status"] != "answer":
                continue
            items.append(
                {
                    "item": f"{row['id']}:{n}",
                    "question": row["question"],
                    "dialog": row["dialog"],
                    "recipes": [
                        {"n": i, "name": r["name"], "ingredients": r["ingredients"]}
                        for i, r in enumerate(answer["shown"], start=1)
                    ],
                    "statements": answer["statements"],
                }
            )
    return items


def render(
    rows: Sequence[dict[str, Any]], meta: dict[str, Any], verdicts: dict[str, Any]
) -> str:
    ok = [r for r in rows if not r.get("error")]
    lines = [
        f"# Перечень «что есть дома»: recipes_v5 против v6 ({meta['date']})",
        "",
        f"Кейсы `evals/dataset/pantry/` ({len(rows)}, от независимого субагента). Модель — "
        f"{meta['model']}; разбор вопроса и поиск — один раз на кейс, ответ — обеими "
        f"версиями на тех же рецептах; один прогон. Судья — субагент Sonnet "
        f"(judge_v10, relevance_v3), версия ему не видна.",
        "",
        "| Метрика | " + " | ".join(PROMPTS) + " |",
        "|---|" + "---|" * len(PROMPTS),
    ]

    def per(name: str, value: Any) -> None:
        lines.append(f"| {name} | " + " | ".join(value(p) for p in PROMPTS) + " |")

    per(
        "ответ, а не «не нашёл»",
        lambda p: str(Rate(sum(r[p]["status"] == "answer" for r in ok), len(ok))),
    )
    per(
        "показанные рецепты по условиям кейса",
        lambda p: str(
            Rate(
                sum(f for r in ok for f in r[p]["fits"]),
                sum(len(r[p]["fits"]) for r in ok),
            )
        ),
    )
    per(
        "утверждения прошли проверку кодом",
        lambda p: str(
            Rate(
                sum(r[p]["kept"] for r in ok),
                sum(r[p]["kept"] + r[p]["dropped"] for r in ok),
            )
        ),
    )
    per(
        "время ответа, медиана",
        lambda p: (
            f"{statistics.median([r[p]['seconds'] for r in ok]) if ok else 0:.0f} с"
        ),
    )
    if verdicts:

        def judged(p: str, key: str, field: str) -> Rate:
            n = PROMPTS.index(p)
            marks = [
                m for r in ok for m in verdicts.get(f"{r['id']}:{n}", {}).get(key, [])
            ]
            return Rate(sum(m.get(field) is True for m in marks), len(marks))

        per(
            "рецепты по вопросу (судья)",
            lambda p: str(judged(p, "recipes", "relevant")),
        )
        per(
            "утверждения без нарушений (судья)",
            lambda p: str(judged(p, "statements", "supported")),
        )
    lines += [
        f"| сбои | {len(rows) - len(ok)} | |",
        f"| цена прогона (кэш промпта — по полной цене) | {meta['cost']} | |",
        "",
        "## Кейсы",
        "",
        "| кейс | " + " | ".join(PROMPTS) + " |",
        "|---|" + "---|" * len(PROMPTS),
    ]
    for r in rows:
        if r.get("error"):
            lines.append(f"| `{r['id']}` | сбой: {r['error'][:80]} | |")
            continue
        cells = []
        for p in PROMPTS:
            a = r[p]
            cell = (
                a["status"]
                if a["status"] != "answer"
                else (
                    f"{len(a['shown'])} рец., условия {sum(a['fits'])}/{len(a['fits'])}"
                )
            )
            if a["status"] != "answer":
                cell += f" — {a['message'][:90]}".replace("|", "/")
            cells.append(cell)
        lines.append(f"| `{r['id']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.pantry_eval", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    go = commands.add_parser("run", help="прогон по кейсам pantry, обе версии")
    go.add_argument(
        "--task", type=Path, required=True, help="куда положить задание судье"
    )
    go.add_argument("--workers", type=int, default=4)
    sc = commands.add_parser("score", help="вердикты судьи → отчёт")
    sc.add_argument("raw", type=Path)
    sc.add_argument("task", type=Path, help="каталог с verdicts*.json")
    args = parser.parse_args(argv)
    today = date.today()
    report = REPORTS_DIR / f"{today}-pantry.md"
    if args.command == "run":
        rows, model, cost = run(load_pantry_cases(), args.workers)
        meta = {"date": str(today), "model": model, "cost": cost}
        raw = RAW / f"{today}-pantry.json"
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
