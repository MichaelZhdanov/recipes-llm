"""Общие заготовки для тестов evals: кейс, результат поиска, паспорт индекса, ответ."""

import json
from typing import Literal

from chunk_helpers import text_chunk

from assistant.generation.answer import assemble
from assistant.generation.citations import SourceText
from assistant.generation.gigachat import Completion, Usage
from assistant.retrieval.store import Hit, IndexManifest
from evals.answer_metrics import AnswerRun
from evals.dataset import Case
from evals.retrieval_metrics import CaseResult, RetrievalRun

MANIFEST = IndexManifest(
    dataset="rogozinushka/povarenok-recipes",
    ref="v3.4.2",
    file_sha256="0" * 64,
    recipes=3,
    skipped={},
    sample=None,
    seed=None,
    embedding_model="fake-model",
    dim=64,
    quantization=None,
    built_at="2026-11-04T00:00:00+00:00",
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


# --- Ответ ассистента: один источник, три утверждения, второе с выдуманной цитатой ---

TEXT = """\
Mail

Paperless can fetch attachments from IMAP mail accounts.
It checks the mail every ten minutes by default."""

CHUNK = text_chunk(TEXT, recipe_id=10)
SOURCES = [SourceText(1, CHUNK, TEXT)]


def answered(
    statements: list[tuple[str, str]], finish_reason: str = "stop"
) -> AnswerRun:
    content = json.dumps(
        {
            "status": "answer",
            "statements": [
                {"text": text, "citations": [{"source": 1, "quote": quote}]}
                for text, quote in statements
            ],
        }
    )
    completion = Completion(
        content, finish_reason, "GigaChat-2-Max:1", Usage(100, 20, 0, 120)
    )
    result = assemble(
        [Hit(CHUNK, 0.6)], SOURCES, completion, prompt_version="answer_v1", ref="v3.4.2"
    )
    return AnswerRun.from_result(result)


RUN = answered(
    [
        ("Почта забирается из IMAP.", "fetch attachments from IMAP mail accounts"),
        ("Выдумка.", "this quote is not in the source"),
        ("Проверка раз в десять минут.", "checks the mail every ten minutes"),
    ]
)
