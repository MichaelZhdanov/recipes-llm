import json
from collections import Counter
from pathlib import Path

import pytest
from eval_helpers import MANIFEST, case, result

from assistant.config import Settings
from assistant.generation.answer import Answer
from assistant.generation.citations import CheckedCitation, CheckedStatement
from assistant.generation.gigachat import LlmStats
from assistant.retrieval.embeddings import EmbeddingStats
from evals.answer_metrics import AnswerCase, AnswerRun
from evals.dataset import Case
from evals.report import RunMeta
from evals.run import (
    _categories,
    answer_records,
    load_meta,
    meta_record,
    pick_cases,
    raw_records,
    rescore_label,
)


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
        dataset_revision="v3.4.2",
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


def dev_cases() -> list[Case]:
    howto = [case(f"howto-{n:03}", ["docs/a.md#x"]) for n in range(1, 13)]
    config = [
        case(f"config-{n:03}", ["docs/a.md#x"]).model_copy(
            update={"category": "config"}
        )
        for n in range(1, 9)
    ]
    oos = [
        Case(
            id=f"out_of_scope-{n:03}",
            category="out_of_scope",
            role="user",
            question="?",
            expected_behavior="abstain",
        )
        for n in range(1, 13)
    ]
    return [*howto, *config, *oos]


def test_sample_takes_every_category_and_is_repeatable() -> None:
    cases = dev_cases()

    picked = pick_cases(cases, 6)

    assert Counter(c.category for c in picked) == {
        "howto": 2,
        "config": 2,
        "out_of_scope": 2,
    }
    assert [c.id for c in picked] == [c.id for c in pick_cases(cases, 6)]
    assert [c.id for c in picked] != [c.id for c in pick_cases(cases, 6, seed=1)]
    # Порядок как в наборе: сначала howto, потом config, потом out_of_scope.
    assert [c.id for c in picked] == sorted(
        (c.id for c in picked), key=[c.id for c in cases].index
    )


def test_sample_larger_than_set_takes_all() -> None:
    cases = dev_cases()

    assert pick_cases(cases, 100) == cases


def test_categories_flag_is_a_comma_list() -> None:
    assert _categories("error, howto,") == frozenset({"error", "howto"})


def test_meta_with_masked_pii_survives_json() -> None:
    """dataclasses.asdict превратил бы Counter в счётчик пар, и json.dumps упал бы."""
    meta = RunMeta(
        manifest=MANIFEST,
        revision="abc1234",
        provider="cloud.ru",
        query_template=None,
        embedding_stats=EmbeddingStats(requests=1, masked=Counter({"phone": 2})),
        llm_stats=LlmStats(requests=3, masked=Counter({"email": 1})),
        llm_model="GigaChat-2-Pro",
        temperature=1e-4,
        prompt_version="answer_v1",
    )

    data = json.loads(json.dumps(meta_record(meta)))
    loaded = load_meta(data, revision="def", rescored_from="raw.json")

    assert loaded.llm_stats.masked == Counter({"email": 1})
    assert loaded.embedding_stats.masked == Counter({"phone": 2})
