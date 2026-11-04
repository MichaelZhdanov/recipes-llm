"""Отчёт eval в markdown: параметры прогона, метрики с порогами, разбор кейсов.

Здесь только оформление: метрики считают `retrieval_metrics`, а запись файлов
и код выхода — `run`.
"""

import statistics
import subprocess
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from assistant.indexing.build import estimate_cost
from assistant.retrieval.embeddings import RETRIEVAL_TASK, EmbeddingStats
from assistant.retrieval.store import IndexManifest
from evals.retrieval_metrics import (
    CaseResult,
    K,
    mrr_per_run,
    percentile,
    recall_per_run,
)

# Пороги из CLAUDE.md («Метрики и пороги»). Менять только с записью в отчёте.
THRESHOLDS = {"recall@5": 0.80}


def _mean_std(values: Sequence[float]) -> str:
    return f"{statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}"


def git_revision(cwd: Path | None = None) -> str:
    """Короткий SHA и пометка, если отслеживаемые файлы изменены.

    Неотслеживаемые файлы (черновики, заметки) на код не влияют, поэтому
    отчёт ими не помечается.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "неизвестен"
    return f"{sha} (есть незакоммиченные изменения)" if dirty else sha


def _instruction_cell(template: str | None) -> str:
    if not template:
        return "нет, вопрос уходит как есть"
    shown = template.format(task=RETRIEVAL_TASK, query="<вопрос>")
    # Перевод строки сломал бы таблицу markdown.
    return "`" + shown.replace("\n", "\\n") + "`"


def render_report(
    results: Sequence[CaseResult],
    *,
    manifest: IndexManifest,
    revision: str,
    provider: str,
    query_template: str | None,
    stats: EmbeddingStats,
    today: date,
) -> str:
    recall = recall_per_run(results)
    mrr = mrr_per_run(results)
    latencies = [lat for r in results for lat in r.latencies]
    passed = statistics.mean(recall) >= THRESHOLDS["recall@5"]
    cost = estimate_cost(manifest.embedding_model, stats.tokens)
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
        f"| Модель эмбеддингов | {provider}: {manifest.embedding_model}, размерность {manifest.dim} |",
        f"| Инструкция к вопросу | {_instruction_cell(query_template)} |",
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
        f"| токены запросов к API эмбеддингов | {stats.tokens}"
        f"{f' ({cost})' if cost else ''} | показать | |",
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
