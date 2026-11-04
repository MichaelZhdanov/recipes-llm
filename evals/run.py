"""Прогон эталонного набора: `python -m evals.run` (через `make eval`).

Здесь только порядок действий: загрузить кейсы, прогнать, записать отчёт
и сырые данные, вернуть код выхода. Метрики считают
`evals/retrieval_metrics.py` и `evals/answer_metrics.py`, отчёт
оформляет `evals/report.py`.

Режимы (`--stage`):
- `answer` (по умолчанию) — ассистент целиком: поиск, GigaChat, проверка
  ссылок. Кейсы howto, config и out_of_scope;
- `retrieval` — только поиск, как на этапе 1: ключ GigaChat не нужен.

После генерации модель-судья (`evals/judge.py`) оценивает groundedness
и проверяется на канарейках; `--no-judge` её выключает. `--rescore RAW`
пересчитывает отчёт по сохранённым ответам модели без новой генерации:
ссылки проверяются текущим кодом, судья запускается заново.

Каждый кейс прогоняется RUNS раз (правило из CLAUDE.md), в отчёте среднее
и разброс.

Быстрый прогон при работе над промптом: `--sample N --runs 1` — N кейсов,
поровну из каждой категории, с фиксированным сидом. Одинаковый сид даёт ту же
выборку, так что быстрые прогоны разных версий сравнимы между собой. Цифры
такого прогона ориентировочные. В отчёт этапа идёт только полный прогон.

Отчёт: evals/reports/<дата>[-метка].md, сырые данные — в evals/reports/raw/
(эта папка в .gitignore), одна запись на каждую пару «кейс, прогон».
Код выхода 1, если метрика ниже порога.
"""

import argparse
import dataclasses
import json
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import httpx2
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.generation.answer import Answerer, assemble
from assistant.generation.citations import SourceText
from assistant.generation.gigachat import (
    DETERMINISTIC_TEMPERATURE,
    Completion,
    LlmStats,
    Usage,
    llm_model,
    make_http_client,
    make_llm,
)
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import EmbeddingStats, make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import Hit, IndexManifest, VectorStore
from evals.answer_metrics import AnswerCase, AnswerRun
from evals.dataset import (
    CLARITY_CANARIES_FILE,
    DATASET_DIR,
    HOLDOUT_DIR,
    Case,
    load_canaries,
    load_cases,
    load_clarity_canaries,
)
from evals.judge import Judge, Judging, run_judge
from evals.report import (
    THRESHOLDS,
    RunMeta,
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
SAMPLE_SEED = 20261104
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


def answer_records(
    results: Sequence[AnswerCase], judging: Judging | None = None
) -> list[dict[str, object]]:
    """Сырые данные ответов: всё, чтобы разобрать прогон и пересчитать метрики без LLM."""
    verdicts = judging.verdicts if judging else {}
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
            "judge": (
                dataclasses.asdict(verdicts[(r.case.id, number)])
                if (r.case.id, number) in verdicts
                else None
            ),
        }
        for r in results
        for number, run in enumerate(r.runs)
    ]


def meta_record(meta: RunMeta) -> dict[str, object]:
    return {
        "manifest": meta.manifest.model_dump(),
        "revision": meta.revision,
        "provider": meta.provider,
        "query_template": meta.query_template,
        "embedding_stats": dataclasses.asdict(meta.embedding_stats),
        "llm_stats": dataclasses.asdict(meta.llm_stats),
        "llm_model": meta.llm_model,
        "temperature": meta.temperature,
        "prompt_version": meta.prompt_version,
        "split": meta.split,
        "sample": meta.sample,
    }


def load_meta(data: dict[str, Any], *, revision: str, rescored_from: str) -> RunMeta:
    embedding = data["embedding_stats"]
    llm = data["llm_stats"]
    return RunMeta(
        manifest=IndexManifest.model_validate(data["manifest"]),
        revision=revision,
        provider=data["provider"],
        query_template=data["query_template"],
        embedding_stats=EmbeddingStats(
            requests=embedding["requests"],
            tokens=embedding["tokens"],
            masked=Counter(embedding["masked"]),
        ),
        llm_stats=LlmStats(**{**llm, "masked": Counter(llm["masked"])}),
        llm_model=data["llm_model"],
        temperature=data["temperature"],
        prompt_version=data["prompt_version"],
        split=data.get("split", "dev"),
        sample=data.get("sample"),
        rescored_from=f"{rescored_from}, прогон на {data['revision']}",
    )


def load_answer_run(record: dict[str, Any]) -> AnswerRun:
    """Прогон из сырых данных. Ссылки проверяются заново текущим кодом (assemble)."""
    sources = [
        SourceText(s["number"], Chunk.model_validate(s["chunk"]), s["text"])
        for s in record["sources"]
    ]
    hits = [
        Hit(chunk=source.chunk, score=score)
        for source, (_, score) in zip(sources, record["top"], strict=True)
    ]
    completion = None
    if record["finish_reason"] is not None:
        completion = Completion(
            content=record["content"],
            finish_reason=record["finish_reason"],
            model=record["answer"]["model"],
            usage=Usage(**record["usage"]),
        )
    result = assemble(
        hits,
        sources,
        completion,
        prompt_version=record["answer"]["prompt_version"],
        ref=record["answer"]["paperless_ref"],
        retrieval_seconds=record["retrieval_seconds"],
        generation_seconds=record["generation_seconds"],
    )
    return AnswerRun.from_result(result)


def load_answer_cases(
    records: Sequence[dict[str, Any]], cases: dict[str, Case]
) -> list[AnswerCase]:
    runs: dict[str, list[AnswerRun]] = {}
    for record in records:
        runs.setdefault(record["id"], []).append(load_answer_run(record))
    return [
        AnswerCase(cases[case_id], case_runs) for case_id, case_runs in runs.items()
    ]


def pick_cases(cases: Sequence[Case], size: int, seed: int = SAMPLE_SEED) -> list[Case]:
    """size кейсов: по кругу из каждой категории, порядок набора сохраняется.

    По кругу, а не просто случайно: в маленькой выборке иначе может не
    оказаться out_of_scope, и abstention не посчитать.
    """
    if size >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    groups: dict[str, list[Case]] = {}
    for case in cases:
        groups.setdefault(case.category, []).append(case)
    queues = [rng.sample(group, len(group)) for _, group in sorted(groups.items())]
    picked: list[Case] = []
    while len(picked) < size:
        for queue in queues:
            if queue and len(picked) < size:
                picked.append(queue.pop())
    order = {case.id: index for index, case in enumerate(cases)}
    return sorted(picked, key=lambda case: order[case.id])


def _choose(
    cases: list[Case], sample: int | None, seed: int
) -> tuple[list[Case], str | None]:
    if sample is None or sample >= len(cases):
        return cases, None
    return pick_cases(cases, sample, seed), f"{sample} из {len(cases)}, сид {seed}"


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


def _retrieval_stage(
    settings: Settings, runs: int, label: str | None, sample: int | None, seed: int
) -> int:
    # Коммит запоминается до прогона: правки во время долгого прогона
    # не должны попасть в его отчёт.
    revision = git_revision()
    cases, sampled = _choose(
        [c for c in load_cases() if c.category in RETRIEVAL_CATEGORIES], sample, seed
    )
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
        revision=revision,
        provider=embedder.provider.name,
        query_template=embedder.query_template,
        stats=embedder.stats,
        today=today,
        sample=sampled,
    )
    path = report_path(REPORTS_DIR, today, label)
    _write(path, report, raw_records(results))
    recall = statistics.mean(recall_per_run(results))
    print(
        f"recall@{K} = {recall:.2f} (порог {THRESHOLDS['recall@5']:.2f}), отчёт: {path}"
    )
    return 0 if recall >= THRESHOLDS["recall@5"] else 1


def _judge(
    settings: Settings, http: httpx2.Client, results: Sequence[AnswerCase]
) -> Judging:
    # Свой экземпляр клиента: токены судьи считаются отдельно от токенов ассистента.
    judge = Judge(make_llm(settings, http))
    _log(
        f"судья {judge.model}: {sum(len(r.runs) for r in results)} ответов и канарейки"
    )
    clarity = load_clarity_canaries() if CLARITY_CANARIES_FILE.exists() else []
    return run_judge(results, judge, load_canaries(), clarity)


def _finish(
    results: Sequence[AnswerCase],
    meta: RunMeta,
    judging: Judging | None,
    label: str | None,
) -> int:
    today = date.today()
    report = render_answer_report(results, meta=meta, judging=judging, today=today)
    path = report_path(REPORTS_DIR, today, label)
    raw = {"meta": meta_record(meta), "records": answer_records(results, judging)}
    _write(path, report, raw)
    gates = answer_gates(results, judging)
    failed = [name for name, passed in gates.items() if not passed]
    print(f"не прошли порог: {', '.join(failed)}" if failed else "все пороги пройдены")
    print(f"отчёт: {path}")
    return 1 if failed else 0


def _answer_stage(
    settings: Settings,
    runs: int,
    label: str | None,
    judge: bool,
    split: str,
    sample: int | None = None,
    seed: int = SAMPLE_SEED,
) -> int:
    revision = git_revision()
    directory = HOLDOUT_DIR if split == "holdout" else DATASET_DIR
    cases, sampled = _choose(
        [c for c in load_cases(directory) if c.category in ANSWER_CATEGORIES],
        sample,
        seed,
    )
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
        judging = _judge(settings, llm_http, results) if judge else None

    meta = RunMeta(
        manifest=retriever.manifest,
        revision=revision,
        provider=embedder.provider.name,
        query_template=embedder.query_template,
        embedding_stats=embedder.stats,
        llm_stats=llm.stats,
        llm_model=model,
        temperature=DETERMINISTIC_TEMPERATURE,
        prompt_version=answerer.prompt_version,
        split=split,
        sample=sampled,
    )
    return _finish(results, meta, judging, label)


def rescore_label(raw_path: Path) -> str:
    """2026-11-04-gigachat-2-pro.json → gigachat-2-pro-rescore: дату добавит report_path."""
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}-?", "", raw_path.stem)
    return f"{stem}-rescore" if stem else "rescore"


def _rescore(settings: Settings, raw_path: Path, label: str | None, judge: bool) -> int:
    """Метрики заново по сохранённым ответам модели: без генерации, ссылки — текущим кодом."""
    revision = git_revision()
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    cases = {c.id: c for c in [*load_cases(), *load_cases(HOLDOUT_DIR)]}
    results = load_answer_cases(data["records"], cases)
    meta = load_meta(data["meta"], revision=revision, rescored_from=str(raw_path))
    judging = None
    if judge:
        with make_http_client() as llm_http:
            judging = _judge(settings, llm_http, results)
    return _finish(results, meta, judging, label or rescore_label(raw_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="быстрый прогон: N кейсов поровну из категорий (с --runs 1 — за минуты)",
    )
    parser.add_argument(
        "--seed", type=int, default=SAMPLE_SEED, help="сид для --sample"
    )
    parser.add_argument(
        "--split",
        choices=("dev", "holdout"),
        default="dev",
        help="holdout — отложенная выборка, её прогоняют один раз в конце",
    )
    parser.add_argument("--label", help="суффикс имени отчёта, например имя модели")
    parser.add_argument(
        "--stage",
        choices=("answer", "retrieval"),
        default="answer",
        help="answer — ассистент целиком; retrieval — только поиск, без GigaChat",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="без модели-судьи (groundedness не считается)",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        metavar="RAW_JSON",
        help="пересчитать метрики по сохранённым ответам модели, без генерации",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    if args.rescore:
        return _rescore(settings, args.rescore, args.label, not args.no_judge)
    if args.stage == "retrieval":
        return _retrieval_stage(settings, args.runs, args.label, args.sample, args.seed)
    label = args.label or ("holdout" if args.split == "holdout" else None)
    return _answer_stage(
        settings,
        args.runs,
        label,
        not args.no_judge,
        args.split,
        args.sample,
        args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
