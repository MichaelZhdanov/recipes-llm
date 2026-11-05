from collections import Counter
from pathlib import Path
from typing import cast

import pytest
from eval_helpers import CHUNK, SOURCES, case

from assistant.generation.answer import Answerer, AnswerResult, assemble
from assistant.generation.gigachat import Completion, LlmError, LlmStats, Usage
from assistant.retrieval.embeddings import EmbeddingStats
from assistant.retrieval.store import Hit
from evals import progress
from evals.answer_metrics import AnswerCase
from evals.run import (
    _open_journal,
    _resumed_cases,
    _without_resume,
    answer_records,
    run_answers,
)

# Вопрос — id кейса: так заглушка знает, на каком кейсе «упасть».
CASES = [
    case(case_id, question=case_id)
    for case_id in ("by_name-001", "by_name-002", "by_name-003")
]


class FlakyAnswerer:
    """Вместо ассистента: на вопросе fail_on — 429, как у GigaChat на Freemium."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.asked: list[str] = []

    def answer(self, question: str, history: object = ()) -> AnswerResult:
        self.asked.append(question)
        if question == self.fail_on:
            raise LlmError("GigaChat ответил HTTP 429")
        completion = Completion(
            '{"status": "not_found", "statements": []}',
            "stop",
            "GigaChat-2-Pro:1",
            Usage(100, 20, 0, 120),
        )
        return assemble(
            [Hit(CHUNK, 0.6)],
            SOURCES,
            completion,
            prompt_version="answer_v1",
            ref="v3.4.2",
        )


def run(
    answerer: FlakyAnswerer, path: Path, done: dict[str, AnswerCase] | None = None
) -> list[AnswerCase]:
    def save(result: AnswerCase) -> None:
        progress.append(
            path,
            answer_records([result]),
            LlmStats(requests=len(answerer.asked)),
            EmbeddingStats(tokens=1),
        )

    return run_answers(CASES, cast(Answerer, answerer), runs=2, done=done, on_case=save)


def test_interrupted_run_resumes_without_repeating_cases(tmp_path: Path) -> None:
    path = tmp_path / "2026-11-04-errors.partial.jsonl"
    progress.start(path, "abc1234", [c.id for c in CASES], runs=2)

    with pytest.raises(LlmError):
        run(FlakyAnswerer(fail_on="by_name-002"), path)

    journal = progress.load(path)
    assert journal.revision == "abc1234"
    assert list(journal.done) == ["by_name-001"]
    assert len(journal.done["by_name-001"]) == 2

    again = FlakyAnswerer()
    results = run(again, path, done=_resumed_cases(journal, CASES))

    # Готовый кейс не спрашивается снова, в итоге каждый кейс ровно один раз.
    assert again.asked == ["by_name-002"] * 2 + ["by_name-003"] * 2
    assert [r.case.id for r in results] == ["by_name-001", "by_name-002", "by_name-003"]
    assert all(len(r.runs) == 2 for r in results)
    assert list(progress.load(path).done) == [
        "by_name-001",
        "by_name-002",
        "by_name-003",
    ]


def test_stats_are_the_last_cumulative_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "j.partial.jsonl"
    progress.start(path, "abc", ["a", "b"], runs=1)
    progress.append(path, [], LlmStats(requests=1, total_tokens=10), EmbeddingStats())
    progress.append(
        path,
        [],
        LlmStats(requests=2, total_tokens=25, masked=Counter({"email": 1})),
        EmbeddingStats(requests=1, tokens=7),
    )

    journal = progress.load(path)

    assert journal.llm_stats.total_tokens == 25
    assert journal.llm_stats.masked == Counter({"email": 1})
    assert journal.embedding_stats.tokens == 7
    # Статистика нового запуска складывается с той, что была до него.
    total = progress.add_llm_stats(
        journal.llm_stats, LlmStats(requests=3, total_tokens=5)
    )
    assert (total.requests, total.total_tokens) == (5, 30)


def test_line_cut_off_by_a_crash_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "j.partial.jsonl"
    progress.start(path, "abc", ["a"], runs=1)
    with path.open("a", encoding="utf-8") as file:
        file.write('{"records": [{"id": "a", "ru')

    assert progress.load(path).done == {}


def test_unfinished_run_is_not_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "j.partial.jsonl"
    progress.start(path, "abc", [c.id for c in CASES], runs=2)

    assert _open_journal(path, False, "def", CASES, 2) is None
    assert "--resume" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8").startswith('{"revision": "abc"')


def test_resume_needs_the_same_cases_and_runs(tmp_path: Path) -> None:
    path = tmp_path / "j.partial.jsonl"
    progress.start(path, "abc", [c.id for c in CASES], runs=2)

    assert _open_journal(path, True, "def", CASES[:2], 2) is None
    assert _open_journal(path, True, "def", CASES, 3) is None
    journal = _open_journal(path, True, "def", CASES, 2)
    # Коммит — с которого прогон начат, а не текущий.
    assert journal is not None and journal.revision == "abc"


def test_resume_hint_keeps_flags_and_drops_old_resume() -> None:
    assert _without_resume(
        ["--label", "errors", "--resume", "old.jsonl", "--categories", "error"]
    ) == ["--label", "errors", "--categories", "error"]
    assert _without_resume(["--resume=old.jsonl", "--no-judge"]) == ["--no-judge"]
