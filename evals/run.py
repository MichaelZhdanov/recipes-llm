"""Прогон эталонного набора: `python -m evals.run` (через `make eval`).

Этап 1 проверяет только поиск, LLM ещё нет. Метрики:
- recall@5 — доля кейсов, где хотя бы один ожидаемый источник попал в топ-5;
- MRR — среднее 1/ранг первого ожидаемого источника (0, если его нет в топ-5);
- задержка поиска (эмбеддинг вопроса + Qdrant), p50 и p95.

Каждый кейс прогоняется RUNS раз (правило из CLAUDE.md), в отчёте среднее
и разброс. Поиск детерминирован, поэтому разброс ожидается нулевой; если
он не нулевой, это повод разобраться.

Отчёт: evals/reports/<дата>[-метка].md, сырые данные — в evals/reports/raw/
(эта папка в .gitignore). Код выхода 1, если метрика ниже порога.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx2
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.indexing.build import PRICE_PER_MTOK
from assistant.retrieval.embeddings import EmbeddingStats, make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import IndexManifest, VectorStore
from evals.dataset import Case, load_cases

K = 5
RUNS = 3
# Пороги из CLAUDE.md («Метрики и пороги»). Менять только с записью в отчёте.
THRESHOLDS = {"recall@5": 0.80}
# Категории, где на этапе 1 проверяется поиск.
EVAL_CATEGORIES = frozenset({"howto", "config"})
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class CaseResult:
    case: Case
    # Ранг первого ожидаемого источника в каждом прогоне (1-based), None — нет в топ-K.
    ranks: list[int | None]
    # Источники топ-K и их score из последнего прогона: для разбора промахов.
    top: list[tuple[str, float]]
    latencies: list[float]


def first_hit_rank(expected: Collection[str], retrieved: Sequence[str]) -> int | None:
    for rank, source_id in enumerate(retrieved, start=1):
        if source_id in expected:
            return rank
    return None


def recall_per_run(results: Sequence[CaseResult], k: int = K) -> list[float]:
    runs = len(results[0].ranks)
    return [
        sum(1 for r in results if (rank := r.ranks[run]) is not None and rank <= k)
        / len(results)
        for run in range(runs)
    ]


def mrr_per_run(results: Sequence[CaseResult]) -> list[float]:
    runs = len(results[0].ranks)
    return [
        sum(1 / rank if (rank := r.ranks[run]) else 0.0 for r in results) / len(results)
        for run in range(runs)
    ]


def percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def missing_sources(
    cases: Sequence[Case], has_source: Callable[[str], bool]
) -> list[str]:
    """Кейсы, ссылающиеся на источник, которого нет в индексе: такой кейс не засчитается никогда."""
    return [
        f"{c.id}: {s}" for c in cases for s in c.expected_sources if not has_source(s)
    ]


def run_cases(
    cases: Sequence[Case],
    retriever: Retriever,
    *,
    runs: int = RUNS,
    k: int = K,
    clock: Callable[[], float] = time.perf_counter,
) -> list[CaseResult]:
    results = []
    for case in cases:
        ranks: list[int | None] = []
        latencies: list[float] = []
        top: list[tuple[str, float]] = []
        for _ in range(runs):
            started = clock()
            hits = retriever.search(case.question, k=k)
            latencies.append(clock() - started)
            top = [(h.chunk.source_id, h.score) for h in hits]
            ranks.append(first_hit_rank(case.expected_sources, [s for s, _ in top]))
        results.append(CaseResult(case=case, ranks=ranks, top=top, latencies=latencies))
    return results


def _mean_std(values: Sequence[float]) -> str:
    return f"{statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}"


def _git_revision() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "неизвестен"
    return f"{sha} (есть незакоммиченные изменения)" if dirty else sha


def render_report(
    results: Sequence[CaseResult],
    *,
    manifest: IndexManifest,
    revision: str,
    stats: EmbeddingStats,
    today: date,
) -> str:
    recall = recall_per_run(results)
    mrr = mrr_per_run(results)
    latencies = [lat for r in results for lat in r.latencies]
    passed = statistics.mean(recall) >= THRESHOLDS["recall@5"]
    price = PRICE_PER_MTOK.get(manifest.embedding_model)
    cost = f" (≈${stats.tokens / 1_000_000 * price:.5f})" if price is not None else ""
    categories = Counter(r.case.category for r in results)
    chunks = ", ".join(f"{k} {v}" for k, v in sorted(manifest.chunks.items()))

    lines = [
        f"# Отчёт eval — {today.isoformat()}",
        "",
        "Этап 1: проверяется только поиск (LLM нет). Набор: `evals/dataset/howto_config.yaml`,",
        "его писал независимый субагент до появления кода поиска (`evals/dataset/README.md`).",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Коммит ассистента | {revision} |",
        f"| Paperless | {manifest.ref} ({manifest.commit or 'коммит неизвестен'}) |",
        f"| Модель эмбеддингов | {manifest.embedding_model}, размерность {manifest.dim} |",
        f"| Индекс | {sum(manifest.chunks.values())} кусков: {chunks}; построен {manifest.built_at} |",
        f"| Кейсов | {len(results)} ({', '.join(f'{k} {v}' for k, v in sorted(categories.items()))}), "
        f"прогонов {len(results[0].ranks)} |",
        "| Промпт | не используется |",
        "",
        "## Метрики",
        "",
        "| Метрика | Значение | Порог | Статус |",
        "|---|---|---|---|",
        f"| recall@{K} | {_mean_std(recall)} | ≥ {THRESHOLDS['recall@5']:.2f} | "
        f"{'✅' if passed else '❌'} |",
        f"| MRR | {_mean_std(mrr)} | — | |",
        f"| задержка поиска p50 / p95 | {percentile(latencies, 0.5):.2f} / "
        f"{percentile(latencies, 0.95):.2f} с | показать | |",
        f"| токены запросов Voyage | {stats.tokens}{cost} | показать | |",
        f"| срабатывания маскера ПДн | {sum(stats.masked.values())} | — | |",
        "",
        "## По категориям и ролям",
        "",
        "| Группа | Кейсов | recall@5 |",
        "|---|---|---|",
    ]
    for label, key in (("категория", "category"), ("роль", "role")):
        for value in sorted({getattr(r.case, key) for r in results}):
            group = [r for r in results if getattr(r.case, key) == value]
            lines.append(
                f"| {label} {value} | {len(group)} | {statistics.mean(recall_per_run(group)):.2f} |"
            )

    lines += [
        "",
        "## По кейсам",
        "",
        "| id | роль | ранг по прогонам | топ-1 |",
        "|---|---|---|---|",
    ]
    for r in results:
        ranks = ", ".join(str(rank) if rank else "—" for rank in r.ranks)
        top1 = r.top[0][0] if r.top else "—"
        lines.append(f"| {r.case.id} | {r.case.role} | {ranks} | `{top1}` |")

    misses = [r for r in results if any(rank is None for rank in r.ranks)]
    lines += ["", f"## Промахи ({len(misses)})", ""]
    if not misses:
        lines.append("Нет.")
    for r in misses:
        lines += [
            f"### {r.case.id} — «{r.case.question}»",
            "",
            "Ожидалось: " + ", ".join(f"`{s}`" for s in r.case.expected_sources),
            "",
            "Найдено:",
            *[
                f"{n}. `{source}` ({score:.3f})"
                for n, (source, score) in enumerate(r.top, start=1)
            ],
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def report_path(directory: Path, today: date, label: str | None) -> Path:
    return directory / f"{today.isoformat()}{f'-{label}' if label else ''}.md"


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
        revision=_git_revision(),
        stats=embedder.stats,
        today=today,
    )
    path = report_path(REPORTS_DIR, today, args.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    raw = REPORTS_DIR / "raw" / f"{path.stem}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(
        json.dumps(
            [
                {
                    "id": r.case.id,
                    "ranks": r.ranks,
                    "top": r.top,
                    "latencies": r.latencies,
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    recall = statistics.mean(recall_per_run(results))
    print(
        f"recall@{K} = {recall:.2f} (порог {THRESHOLDS['recall@5']:.2f}), отчёт: {path}"
    )
    return 0 if recall >= THRESHOLDS["recall@5"] else 1


if __name__ == "__main__":
    sys.exit(main())
