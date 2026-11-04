"""Журнал долгого прогона: каждый готовый кейс сразу пишется на диск.

Полный прогон идёт до часа, а GigaChat на Freemium бывает недоступен (429)
дольше, чем ждут повторы. Раньше сырые данные писались только в конце,
и сбой на кейсе 41 из 42 терял 120 готовых ответов. Теперь после каждого
кейса в журнал дописывается строка, а `--resume ЖУРНАЛ` продолжает прогон:
готовые кейсы не прогоняются заново.

Формат — JSON Lines:
- первая строка — шапка: коммит, с которого начат прогон, список кейсов
  и число прогонов. Продолжить можно только тот же прогон, иначе в одном
  отчёте смешались бы разные наборы;
- дальше по строке на кейс: его сырые записи (тот же формат, что в raw JSON)
  и статистика токенов, накопленная с начала прогона.

Когда отчёт собран и сырые данные записаны, журнал больше не нужен:
всё, что в нём было, есть в raw JSON.
"""

import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assistant.generation.gigachat import LlmStats
from assistant.retrieval.embeddings import EmbeddingStats


def journal_path(raw_dir: Path, report: Path) -> Path:
    """evals/reports/2026-11-04-errors.md → raw/2026-11-04-errors.partial.jsonl."""
    return raw_dir / f"{report.stem}.partial.jsonl"


def add_llm_stats(a: LlmStats, b: LlmStats) -> LlmStats:
    return LlmStats(
        requests=a.requests + b.requests,
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        precached_prompt_tokens=a.precached_prompt_tokens + b.precached_prompt_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        masked=a.masked + b.masked,
    )


def add_embedding_stats(a: EmbeddingStats, b: EmbeddingStats) -> EmbeddingStats:
    return EmbeddingStats(
        requests=a.requests + b.requests,
        tokens=a.tokens + b.tokens,
        masked=a.masked + b.masked,
    )


@dataclass
class Journal:
    """Что уже сделано в прогоне."""

    revision: str
    case_ids: list[str]
    runs: int
    # id кейса → его сырые записи, по одной на прогон.
    done: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Статистика с начала прогона, до текущего запуска.
    llm_stats: LlmStats = field(default_factory=LlmStats)
    embedding_stats: EmbeddingStats = field(default_factory=EmbeddingStats)


def _write_line(file: Any, data: object) -> None:
    file.write(json.dumps(data, ensure_ascii=False) + "\n")
    # Строка должна лечь на диск сразу: следующий кейс может не завершиться.
    file.flush()
    os.fsync(file.fileno())


def start(path: Path, revision: str, case_ids: Sequence[str], runs: int) -> Journal:
    """Новый журнал. Если он уже есть — FileExistsError: незаконченный прогон не затирается."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        _write_line(file, {"revision": revision, "cases": list(case_ids), "runs": runs})
    return Journal(revision, list(case_ids), runs)


def append(
    path: Path,
    records: Sequence[dict[str, object]],
    llm_stats: LlmStats,
    embedding_stats: EmbeddingStats,
) -> None:
    """Строка готового кейса: записи и статистика с начала прогона."""
    with path.open("a", encoding="utf-8") as file:
        _write_line(
            file,
            {
                "records": list(records),
                "llm_stats": llm_stats.as_dict(),
                "embedding_stats": embedding_stats.as_dict(),
            },
        )


def load(path: Path) -> Journal:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    journal = Journal(header["revision"], header["cases"], header["runs"])
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # Оборванная последняя строка: процесс упал во время записи, кейс не готов.
            if number == len(lines):
                break
            raise
        for record in entry["records"]:
            journal.done.setdefault(record["id"], []).append(record)
        llm = entry["llm_stats"]
        journal.llm_stats = LlmStats(**{**llm, "masked": Counter(llm["masked"])})
        embedding = entry["embedding_stats"]
        journal.embedding_stats = EmbeddingStats(
            requests=embedding["requests"],
            tokens=embedding["tokens"],
            masked=Counter(embedding["masked"]),
        )
    return journal
