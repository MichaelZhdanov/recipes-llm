from itertools import count

from eval_helpers import MANIFEST, case, result

from assistant.indexing.docs import chunk_markdown
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from evals.retrieval_metrics import (
    first_hit_rank,
    missing_sources,
    mrr_per_run,
    recall_per_run,
    run_cases,
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
