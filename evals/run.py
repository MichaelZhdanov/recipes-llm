"""Прогон эталонного набора: `python -m evals.run` (через `make eval`).

Здесь только порядок действий: загрузить кейсы, прогнать, записать отчёт
и сырые данные, вернуть код выхода. Метрики поиска считает
`evals/retrieval_metrics.py`, отчёт оформляет `evals/report.py`.

Каждый кейс прогоняется RUNS раз (правило из CLAUDE.md), в отчёте среднее
и разброс. Поиск детерминирован, поэтому разброс ожидается нулевой; если
он не нулевой, это повод разобраться.

Отчёт: evals/reports/<дата>[-метка].md, сырые данные — в evals/reports/raw/
(эта папка в .gitignore), одна запись на каждую пару «кейс, прогон».
Код выхода 1, если метрика ниже порога.
"""

import argparse
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx2
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from evals.dataset import load_cases
from evals.report import THRESHOLDS, git_revision, render_report, report_path
from evals.retrieval_metrics import (
    CaseResult,
    K,
    missing_sources,
    recall_per_run,
    run_cases,
)

RUNS = 3
# Категории, где проверяется поиск.
EVAL_CATEGORIES = frozenset({"howto", "config"})
REPORTS_DIR = Path(__file__).parent / "reports"


def raw_records(results: Sequence[CaseResult]) -> list[dict[str, object]]:
    """Сырые данные: одна запись на пару «кейс, прогон», чтобы разбирать любой прогон."""
    return [
        {
            "id": r.case.id,
            "run": number,
            "rank": run.rank,
            "top": run.top,
            "latency": run.latency,
        }
        for r in results
        for number, run in enumerate(r.runs)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--label", help="суффикс имени отчёта, например имя модели")
    args = parser.parse_args(argv)

    settings = Settings()
    cases = [c for c in load_cases() if c.category in EVAL_CATEGORIES]
    with httpx2.Client(timeout=30) as http:
        embedder = make_embedder(settings, http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        retriever = Retriever(store, embedder, ref=settings.paperless_ref)
        missing = missing_sources(cases, store.has_source)
        if missing:
            print(
                "кейсы ссылаются на источники, которых нет в индексе:",
                *missing,
                sep="\n  ",
            )
            return 2
        results = run_cases(cases, retriever, runs=args.runs)

    today = date.today()
    report = render_report(
        results,
        manifest=retriever.manifest,
        revision=git_revision(),
        provider=embedder.provider.name,
        query_template=embedder.query_template,
        stats=embedder.stats,
        today=today,
    )
    path = report_path(REPORTS_DIR, today, args.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    raw = REPORTS_DIR / "raw" / f"{path.stem}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        json.dumps(raw_records(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    recall = statistics.mean(recall_per_run(results))
    print(
        f"recall@{K} = {recall:.2f} (порог {THRESHOLDS['recall@5']:.2f}), отчёт: {path}"
    )
    return 0 if recall >= THRESHOLDS["recall@5"] else 1


if __name__ == "__main__":
    sys.exit(main())
