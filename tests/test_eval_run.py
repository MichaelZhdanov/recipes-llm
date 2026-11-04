import json
from pathlib import Path

import pytest
from eval_helpers import case, result

from assistant.config import Settings
from assistant.generation.answer import Answer
from assistant.generation.citations import CheckedCitation, CheckedStatement
from evals.answer_metrics import AnswerCase, AnswerRun
from evals.run import answer_records, raw_records, rescore_label


def test_raw_data_has_one_record_per_case_and_run() -> None:
    records = raw_records([result("howto-001", [1, None]), result("howto-002", [3])])

    assert [(r["id"], r["run"], r["rank"]) for r in records] == [
        ("howto-001", 0, 1),
        ("howto-001", 1, None),
        ("howto-002", 0, 3),
    ]
    assert records[0]["top"] == [("docs/b.md#y", 0.5)]


def test_answer_records_are_json_with_model_output_and_checks() -> None:
    answer = Answer(
        status="not_found",
        reason="filtered_out",
        statements=[],
        sources=[],
        model="GigaChat-2:1.0",
        prompt_version="answer_v1",
        paperless_ref="v3.4.2",
    )
    checked = [
        CheckedStatement(
            "Выдумка.",
            [CheckedCitation(1, "no such text", "фрагмента нет в источнике")],
            [],
        )
    ]
    run = AnswerRun(
        answer,
        checked,
        [],
        [("docs/a.md#x", 0.4)],
        0.2,
        1.0,
        None,
        "stop",
        '{"status": "answer"}',
    )

    records = answer_records(
        [AnswerCase(case("howto-001", ["docs/a.md#x"]), [run, run])]
    )

    assert [(r["id"], r["run"]) for r in records] == [
        ("howto-001", 0),
        ("howto-001", 1),
    ]
    data = json.loads(json.dumps(records, ensure_ascii=False))
    assert data[0]["checked"][0]["citations"][0]["error"] == "фрагмента нет в источнике"
    assert data[0]["content"] == '{"status": "answer"}'


def test_rescore_report_name_has_one_date() -> None:
    assert (
        rescore_label(Path("raw/2026-11-04-gigachat-2-pro.json"))
        == "gigachat-2-pro-rescore"
    )
    assert rescore_label(Path("raw/2026-11-04.json")) == "rescore"


def test_revision_is_taken_before_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Правки во время долгого прогона не должны попасть в шапку его отчёта."""
    import evals.run as run

    calls: list[str] = []

    def revision() -> str:
        calls.append("revision")
        return "abc"

    monkeypatch.setattr(run, "git_revision", revision)

    def stop(*_: object, **__: object) -> None:
        calls.append("cases")
        raise SystemExit

    monkeypatch.setattr(run, "load_cases", stop)

    with pytest.raises(SystemExit):
        run._answer_stage(Settings(), 1, None, False, "dev")

    assert calls == ["revision", "cases"]
