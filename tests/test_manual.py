import json
from typing import Any

import pytest
from eval_helpers import RUN, case

from evals.answer_metrics import AnswerCase
from evals.judge import Verdict
from evals.manual import Agreement, compare, render, sample
from evals.run import answer_records


def test_kappa_counts_agreement_beyond_chance() -> None:
    result = Agreement(
        [
            ("howto-001", 0, 1, True, True),
            ("howto-001", 0, 2, True, False),
            ("howto-002", 0, 1, False, False),
            ("howto-002", 0, 2, False, False),
        ]
    )

    assert result.agreement == 0.75
    # Ожидаемое случайное совпадение 0.5, наблюдаемое 0.75: каппа 0.5.
    assert result.kappa == pytest.approx(0.5)


# Вопросы берутся из набора по id кейса; тест не зависит от того, что сейчас в наборе.
QUESTIONS = {"howto-001": "как настроить почту"}


def raw_with_judge() -> dict[str, Any]:
    howto = AnswerCase(case("howto-001", ["docs/usage.md#mail"]), [RUN])
    records = answer_records([howto])
    records[0]["judge"] = json.loads(
        json.dumps(Verdict([True, False], [], ["", ""], [True, True]).__dict__)
    )
    return {"meta": {}, "records": json.loads(json.dumps(records, ensure_ascii=False))}


def test_sample_is_blind_and_compare_finds_disagreement() -> None:
    raw = raw_with_judge()

    [item] = sample(raw, questions=QUESTIONS)

    assert item["id"] == "howto-001"
    assert item["question"] == "как настроить почту"
    assert [s["supported"] for s in item["statements"]] == [None, None]
    assert "judge" not in json.dumps(item, ensure_ascii=False)

    for statement in item["statements"]:
        statement["supported"] = True
    result = compare([item], raw)
    assert result.agreement == 0.5
    assert "howto-001, прогон 1, утверждение 2: автор True, судья False" in render(
        result, marks_file="m.yaml", raw_file="r.json"
    )


def test_unmarked_statement_is_an_error() -> None:
    raw = raw_with_judge()
    [item] = sample(raw, questions=QUESTIONS)

    with pytest.raises(ValueError, match="не размечено"):
        compare([item], raw)
