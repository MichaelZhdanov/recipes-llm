import json
from collections import Counter
from pathlib import Path

import pytest
from eval_helpers import MANIFEST, case, found, result

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
    plov = found("Плов", ("рис",), recipe_id=7)
    records = raw_records(
        [
            result(case("by_name-001", expected_name=["плов"]), [[plov], []]),
            result(
                case("by_name-002", expected_name=["борщ"]), [[plov]], available=False
            ),
        ]
    )

    assert [(r["id"], r["run"], r["score"], r["available"]) for r in records] == [
        ("by_name-001", 0, 1.0, True),
        ("by_name-001", 1, 0.0, True),
        ("by_name-002", 0, 0.0, False),
    ]
    assert records[0]["top"] == [
        {"recipe_id": 7, "name": "Плов", "ingredients": ("рис",), "score": 0.5}
    ]
    json.dumps(records)  # сырые данные пишутся в JSON


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

    records = answer_records([AnswerCase(case("by_name-001"), [run, run])])

    assert [(r["id"], r["run"]) for r in records] == [
        ("by_name-001", 0),
        ("by_name-001", 1),
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
    by_name = [case(f"by_name-{n:03}") for n in range(1, 13)]
    include = [case(f"include-{n:03}", "include") for n in range(1, 9)]
    oos = [case(f"out_of_scope-{n:03}", "out_of_scope") for n in range(1, 13)]
    return [*by_name, *include, *oos]


def test_sample_takes_every_category_and_is_repeatable() -> None:
    cases = dev_cases()

    picked = pick_cases(cases, 6)

    assert Counter(c.category for c in picked) == {
        "by_name": 2,
        "include": 2,
        "out_of_scope": 2,
    }
    assert [c.id for c in picked] == [c.id for c in pick_cases(cases, 6)]
    assert [c.id for c in picked] != [c.id for c in pick_cases(cases, 6, seed=1)]
    # Порядок как в наборе: сначала by_name, потом include, потом out_of_scope.
    assert [c.id for c in picked] == sorted(
        (c.id for c in picked), key=[c.id for c in cases].index
    )


def test_sample_larger_than_set_takes_all() -> None:
    cases = dev_cases()

    assert pick_cases(cases, 100) == cases


def test_categories_flag_is_a_comma_list() -> None:
    assert _categories("include, by_name,") == frozenset({"include", "by_name"})


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


def test_without_cases_eval_explains_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Пустой набор — понятное сообщение и код 2, ещё до Qdrant и API."""
    import evals.run as run

    monkeypatch.setattr(run, "load_cases", lambda directory=None: [])

    assert run.main(["--stage", "retrieval"]) == 2
    assert run.main(["--split", "holdout"]) == 2
    assert "нет кейсов" in capsys.readouterr().err
