"""Прогон эталонного набора: `python -m evals.run` (через `make eval`).

Здесь только порядок действий: загрузить кейсы, прогнать, записать отчёт
и сырые данные, вернуть код выхода. Метрики считают
`evals/retrieval_metrics.py` и `evals/answer_metrics.py`, отчёт
оформляет `evals/report.py`.

Режимы (`--stage`):
- `answer` (по умолчанию) — ассистент целиком: поиск, GigaChat, проверка
  ссылок. Кейсы howto, config и out_of_scope;
- `retrieval` — только поиск, как на этапе 1: ключ GigaChat не нужен.

Каждый кейс прогоняется RUNS раз (правило из CLAUDE.md), в отчёте среднее
и разброс.

Отчёт: evals/reports/<дата>[-метка].md, сырые данные — в evals/reports/raw/
(эта папка в .gitignore), одна запись на каждую пару «кейс, прогон».
Код выхода 1, если метрика ниже порога.
"""

import argparse
import dataclasses
import json
import statistics
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx2
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.generation.answer import Answerer
from assistant.generation.gigachat import (
    DETERMINISTIC_TEMPERATURE,
    llm_model,
    make_http_client,
    make_llm,
)
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from evals.answer_metrics import AnswerCase, AnswerRun
from evals.dataset import Case, load_cases
from evals.report import (
    THRESHOLDS,
    answer_gates,
    git_revision,
    render_answer_report,
    render_report,
    report_path,
)
from evals.retrieval_metrics import (
    CaseResult,
    K,
    missing_sources,
    recall_per_run,
    run_cases,
)

RUNS = 3
# Категории каждого режима.
RETRIEVAL_CATEGORIES = frozenset({"howto", "config"})
ANSWER_CATEGORIES = frozenset({"howto", "config", "out_of_scope"})
REPORTS_DIR = Path(__file__).parent / "reports"


def raw_records(results: Sequence[CaseResult]) -> list[dict[str, object]]:
    """Сырые данные поиска: одна запись на пару «кейс, прогон»."""
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


def answer_records(results: Sequence[AnswerCase]) -> list[dict[str, object]]:
    """Сырые данные ответов: всё, чтобы разобрать прогон и пересчитать метрики без LLM."""
    return [
        {
            "id": r.case.id,
            "run": number,
            "answer": run.answer.model_dump(),
            "content": run.content,
            "finish_reason": run.finish_reason,
            "usage": dataclasses.asdict(run.usage) if run.usage else None,
            "retrieval_seconds": run.retrieval_seconds,
            "generation_seconds": run.generation_seconds,
            "top": run.top,
            "sources": [
                {"number": s.number, "chunk": s.chunk.model_dump(), "text": s.text}
                for s in run.sources
            ],
            "checked": [dataclasses.asdict(s) for s in run.checked],
        }
        for r in results
        for number, run in enumerate(r.runs)
    ]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_answers(
    cases: Sequence[Case], answerer: Answerer, *, runs: int
) -> list[AnswerCase]:
    results = []
    for index, case in enumerate(cases, start=1):
        case_runs = [
            AnswerRun.from_result(answerer.answer(case.question)) for _ in range(runs)
        ]
        outcome = ", ".join(run.answer.reason or "ответ" for run in case_runs)
        _log(f"[{index}/{len(cases)}] {case.id}: {outcome}")
        results.append(AnswerCase(case=case, runs=case_runs))
    return results


def _write(path: Path, report: str, raw: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    raw_path = REPORTS_DIR / "raw" / f"{path.stem}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_sources(cases: Sequence[Case], store: VectorStore) -> bool:
    missing = missing_sources(cases, store.has_source)
    if missing:
        print(
            "кейсы ссылаются на источники, которых нет в индексе:",
            *missing,
            sep="\n  ",
        )
    return not missing


def _retrieval_stage(settings: Settings, runs: int, label: str | None) -> int:
    cases = [c for c in load_cases() if c.category in RETRIEVAL_CATEGORIES]
    with httpx2.Client(timeout=30) as http:
        embedder = make_embedder(settings, http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        retriever = Retriever(store, embedder, ref=settings.paperless_ref)
        if not _check_sources(cases, store):
            return 2
        results = run_cases(cases, retriever, runs=runs)

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
    path = report_path(REPORTS_DIR, today, label)
    _write(path, report, raw_records(results))
    recall = statistics.mean(recall_per_run(results))
    print(
        f"recall@{K} = {recall:.2f} (порог {THRESHOLDS['recall@5']:.2f}), отчёт: {path}"
    )
    return 0 if recall >= THRESHOLDS["recall@5"] else 1


def _answer_stage(settings: Settings, runs: int, label: str | None) -> int:
    cases = [c for c in load_cases() if c.category in ANSWER_CATEGORIES]
    model = llm_model(settings)
    with httpx2.Client(timeout=30) as embeddings_http, make_http_client() as llm_http:
        embedder = make_embedder(settings, embeddings_http)
        llm = make_llm(settings, llm_http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        retriever = Retriever(store, embedder, ref=settings.paperless_ref)
        if not _check_sources(cases, store):
            return 2
        answerer = Answerer(retriever, llm, model=model, ref=settings.paperless_ref)
        _log(f"модель {model}, {len(cases)} кейсов × {runs} прогона")
        results = run_answers(cases, answerer, runs=runs)

    today = date.today()
    report = render_answer_report(
        results,
        manifest=retriever.manifest,
        revision=git_revision(),
        provider=embedder.provider.name,
        query_template=embedder.query_template,
        embedding_stats=embedder.stats,
        llm_stats=llm.stats,
        llm_model=model,
        temperature=DETERMINISTIC_TEMPERATURE,
        prompt_version=answerer.prompt_version,
        today=today,
    )
    path = report_path(REPORTS_DIR, today, label)
    _write(path, report, answer_records(results))
    gates = answer_gates(results)
    failed = [name for name, passed in gates.items() if not passed]
    print(f"не прошли порог: {', '.join(failed)}" if failed else "все пороги пройдены")
    print(f"отчёт: {path}")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--label", help="суффикс имени отчёта, например имя модели")
    parser.add_argument(
        "--stage",
        choices=("answer", "retrieval"),
        default="answer",
        help="answer — ассистент целиком; retrieval — только поиск, без GigaChat",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    if args.stage == "retrieval":
        return _retrieval_stage(settings, args.runs, args.label)
    return _answer_stage(settings, args.runs, args.label)


if __name__ == "__main__":
    sys.exit(main())
