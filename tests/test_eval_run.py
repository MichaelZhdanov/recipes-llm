import dataclasses
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest
from eval_helpers import CHUNK, MANIFEST, SOURCES, answered, case, found, result

from assistant.config import Settings
from assistant.generation import llm as assistant_llm
from assistant.generation.answer import Answer
from assistant.generation.citations import CheckedCitation, CheckedStatement, SourceText
from assistant.generation.gigachat import LlmStats
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import EmbeddingStats
from assistant.retrieval.filters import IngredientFilter
from assistant.tools.search_recipes import SearchArgs
from evals.answer_metrics import AnswerCase, AnswerRun
from evals.dataset import Case
from evals.report import RunMeta
from evals.run import (
    JudgeChoice,
    _categories,
    answer_records,
    load_answer_run,
    load_meta,
    mark_answerable,
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


def test_rescore_keeps_earlier_recipes_and_searches() -> None:
    """Среди источников есть рецепт прошлого хода, а поисков у auto — несколько:
    пересчёт по сырым данным восстанавливает и то и другое."""
    earlier = SourceText(
        2, CHUNK.model_copy(update={"recipe_id": 99}), "Пирог\nМука — 1 стак."
    )
    searches = (
        SearchArgs("пирог", IngredientFilter.of(include=[["мука"]], exclude=["сахар"])),
        SearchArgs("пирог без сахара", IngredientFilter(), dropped=("мука ржаная",)),
    )
    original = dataclasses.replace(
        answered([]), sources=[*SOURCES, earlier], searches=searches
    )
    [record] = answer_records(
        [AnswerCase(case("follow_up-001", "follow_up"), [original])]
    )

    restored = load_answer_run(json.loads(json.dumps(record)))

    assert [s.chunk.recipe_id for s in restored.sources] == [CHUNK.recipe_id, 99]
    assert restored.searches == searches


def test_follow_ups_are_marked_by_whether_their_recipes_are_in_the_index() -> None:
    """Уточнению нужны рецепты из прошлого ответа: нет их в индексе — кейс не проверить."""

    class Index:
        def recipes(self, ids: Sequence[int]) -> list[Chunk]:
            return [CHUNK.model_copy(update={"recipe_id": i}) for i in ids if i == 1]

    def follow_up(case_id: str, url_id: int) -> AnswerCase:
        kase = case(
            case_id,
            "follow_up",
            must_include=["50 г"],
            turns=[
                {"role": "user", "text": "что испечь?"},
                {
                    "role": "assistant",
                    "text": f"https://www.povarenok.ru/recipes/show/{url_id}/",
                },
            ],
        )
        return AnswerCase(kase, [])

    marked = mark_answerable(
        [follow_up("follow_up-001", 1), follow_up("follow_up-002", 2)],
        Index(),
    )

    assert [r.available for r in marked] == [True, False]


def test_default_judge_is_sonnet_and_ultra_is_on_request() -> None:
    """Судья по умолчанию — Sonnet (ручная разметка расхождений, 2026-11-06)."""
    assert JudgeChoice().provider == "anthropic"
    assert JudgeChoice().model_name == "claude-sonnet-5-5"
    assert JudgeChoice("gigachat").model_name == "GigaChat-3-Ultra"


def test_judge_effort_is_explicit_and_shown_in_the_report() -> None:
    """effort задан явно: если вендор сменит умолчание, судья не изменится тихо."""
    assert JudgeChoice().effort_used == "high"
    assert JudgeChoice().label == "claude-sonnet-5-5, effort high"
    # У GigaChat effort нет, в подписи его тоже нет.
    assert JudgeChoice("gigachat", effort="low").label == "GigaChat-3-Ultra"
    # У cloud.ru пусто — по умолчанию модели, иначе — как задан.
    assert JudgeChoice("cloudru").label == "deepseek-ai/DeepSeek-V4-Flash"
    glm = JudgeChoice("cloudru", "z-ai/glm-5.3-flash", "low")
    assert glm.label == "z-ai/glm-5.3-flash, effort low"


def test_judge_llm_gets_the_effort_only_for_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evals.run as run

    seen: list[tuple[str, str]] = []

    def fake_model(settings: Settings, http: object) -> object:
        seen.append((settings.llm_provider, settings.llm_effort))
        return object()

    monkeypatch.setattr(assistant_llm, "make_chat_model", fake_model)
    for choice in (JudgeChoice(), JudgeChoice("gigachat")):
        with run._judge_llm(Settings(llm_effort="low"), choice):
            pass

    assert seen == [("anthropic", "high"), ("gigachat", "")]


def test_stored_judging_round_trips_without_calling_the_judge() -> None:
    """--reuse-judge: отчёт пересобирается по сохранённым вердиктам и канарейкам."""
    from evals.dataset import load_canaries, load_clarity_canaries
    from evals.judge import Judging, Verdict
    from evals.run import judging_record, load_judging

    results = [AnswerCase(case("by_name-001"), [answered([])])]
    verdict = Verdict([True, False], [], ["ок", "выдумано"], [True, True], [[], []])
    canaries = load_canaries()[:2]
    clarity = load_clarity_canaries()[:1]
    judging = Judging(
        model="claude-sonnet-5-5, effort high",
        prompt_version="judge_v6, грамотность clarity_v4",
        verdicts={("by_name-001", 0): verdict},
        canaries=canaries,
        canary_marks=[True, None],
        stats=LlmStats(requests=3, total_tokens=900),
        failures=1,
        clarity_canaries=clarity,
        clarity_marks=[False],
    )
    data = json.loads(
        json.dumps(
            {
                "records": answer_records(results, judging),
                "judging": judging_record(judging),
            },
            ensure_ascii=False,
        )
    )

    loaded = load_judging(data)

    assert loaded.verdicts == {("by_name-001", 0): verdict}
    assert [c.id for c in loaded.canaries] == [c.id for c in canaries]
    assert loaded.canary_marks == [True, None]
    assert loaded.clarity_marks == [False]
    assert (loaded.stats.requests, loaded.stats.total_tokens, loaded.failures) == (
        3,
        900,
        1,
    )


def test_review_survives_raw_data_and_the_judge_sees_the_final_answer() -> None:
    """Итог проверки без LLM не повторить: пересчёт берёт его из сырых данных."""
    from eval_helpers import RUN

    from assistant.generation.review import Mark, Review
    from evals.judge import claims_of
    from evals.run import load_answer_run

    original = RUN
    kept = [s for s in original.checked if s.kept]
    review = Review(
        first={1: Mark(True, "ок"), 2: Mark(False, "выдумано")},
        intro=None,
        revised={2: None},
        second={},
        final=kept[:1],
        intro_kept=False,
        seconds=2.5,
    )
    final_answer = original.answer.model_copy(
        update={"statements": original.answer.statements[:1]}
    )
    run = dataclasses.replace(original, answer=final_answer, review=review)
    [record] = answer_records([AnswerCase(case("by_name-001"), [run])])

    restored = load_answer_run(json.loads(json.dumps(record, ensure_ascii=False)))

    assert restored.review == review
    assert [s.text for s in restored.answer.statements] == [kept[0].text]
    assert [c.text for c in claims_of(restored)] == [kept[0].text]
