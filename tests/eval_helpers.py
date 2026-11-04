"""Общие заготовки для тестов evals: кейс, результат поиска, паспорт индекса."""

from typing import Literal

from assistant.retrieval.store import IndexManifest
from evals.dataset import Case
from evals.retrieval_metrics import CaseResult, RetrievalRun

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
        runs=[
            RetrievalRun(top=[("docs/b.md#y", 0.5)], rank=rank, latency=0.1)
            for rank in ranks
        ],
    )
