from datetime import date
from itertools import count
from pathlib import Path
from typing import Literal

import pytest

from assistant.indexing.docs import chunk_markdown
from assistant.retrieval.embeddings import Embedder, EmbeddingStats
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import IndexManifest, VectorStore
from evals.dataset import Case
from evals.run import (
    CaseResult,
    first_hit_rank,
    missing_sources,
    mrr_per_run,
    recall_per_run,
    render_report,
    report_path,
    run_cases,
)

MANIFEST = IndexManifest(
    ref="v3.4.2",
    commit="d939d87",
    embedding_model="fake-model",
    dim=64,
    built_at="2026-11-04T00:00:00+00:00",
    chunks={"doc": 3},
    assistant_version="0.1.0",
)


def case(
    case_id: str,
    sources: list[str],
    question: str = "?",
    role: Literal["user", "engineer"] = "user",
) -> Case:
    return Case(
        id=case_id,
        category="howto",
        role=role,
        question=question,
        expected_behavior="answer",
        expected_sources=sources,
    )


def result(
    case_id: str, ranks: list[int | None], role: Literal["user", "engineer"] = "user"
) -> CaseResult:
    return CaseResult(
        case=case(case_id, ["docs/a.md#x"], role=role),
        ranks=ranks,
        top=[("docs/b.md#y", 0.5)],
        latencies=[0.1] * len(ranks),
    )


def test_first_hit_rank_takes_any_expected_source() -> None:
    retrieved = ["docs/a.md#1", "docs/b.md#2", "docs/c.md#3"]

    assert first_hit_rank({"docs/c.md#3", "docs/b.md#2"}, retrieved) == 2
    assert first_hit_rank({"docs/z.md#9"}, retrieved) is None


def test_recall_and_mrr_per_run() -> None:
    results = [
        result("howto-001", [1, 1]),
        result("howto-002", [2, None]),
        result("howto-003", [None, None]),
    ]

    assert recall_per_run(results) == [2 / 3, 1 / 3]
    assert mrr_per_run(results) == [(1 + 0.5) / 3, 1 / 3]


def test_missing_sources_are_reported_per_case() -> None:
    cases = [case("howto-001", ["docs/a.md#ok", "docs/a.md#gone"])]

    assert missing_sources(cases, lambda s: s.endswith("#ok")) == [
        "howto-001: docs/a.md#gone"
    ]


DOCS = """\
# Usage

## Incoming mail

Paperless can fetch attachments from IMAP mail accounts.

## Trash

Deleted documents go to the trash and can be restored.

## Time zone

Set PAPERLESS_TIME_ZONE.
"""


def test_run_cases_measures_rank_and_latency(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    chunks = chunk_markdown(
        DOCS, path="docs/usage.md", ref="v3.4.2", repo_url="https://x"
    )
    store.recreate(MANIFEST)
    store.upsert(
        chunks, fake_embedder.embed_documents([c.embedding_text() for c in chunks])
    )
    retriever = Retriever(store, fake_embedder, ref="v3.4.2")
    ticks = count()

    results = run_cases(
        [
            case(
                "howto-001",
                ["docs/usage.md#trash"],
                question="deleted documents restored trash",
            )
        ],
        retriever,
        runs=3,
        clock=lambda: float(next(ticks)),
    )

    assert results[0].ranks == [1, 1, 1]
    assert results[0].latencies == [1.0, 1.0, 1.0]
    assert results[0].top[0][0] == "docs/usage.md#trash"


def test_report_shows_metrics_threshold_and_misses() -> None:
    results = [
        result("howto-001", [1, 1, 1], role="engineer"),
        result("howto-002", [None, None, None]),
    ]

    report = render_report(
        results,
        manifest=MANIFEST,
        revision="abc1234",
        provider="cloud.ru",
        query_template="Instruct: {task}\nQuery: {query}",
        stats=EmbeddingStats(tokens=42),
        today=date(2026, 11, 4),
    )

    assert report.startswith("# Отчёт eval — 2026-11-04")
    assert "| recall@5 | 0.50 ± 0.00 | ≥ 0.80 | ❌ |" in report
    assert "| Paperless | v3.4.2 (d939d87) |" in report
    assert "| Модель эмбеддингов | cloud.ru: fake-model, размерность 64 |" in report
    # Перевод строки в инструкции не должен ломать таблицу markdown.
    assert "\\nQuery: <вопрос>` |" in report
    assert "| роль engineer | 1 | 1.00 |" in report
    assert "## Промахи (1)" in report
    assert "### howto-002" in report


@pytest.mark.parametrize(
    ("label", "name"),
    [(None, "2026-11-04.md"), ("voyage-4-lite", "2026-11-04-voyage-4-lite.md")],
)
def test_report_path(label: str | None, name: str) -> None:
    assert report_path(Path("r"), date(2026, 11, 4), label) == Path("r") / name
