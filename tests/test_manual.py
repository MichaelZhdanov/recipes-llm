import json
from typing import Any

import pytest
from eval_helpers import RUN, case

from evals.answer_metrics import AnswerCase
from evals.judge import Verdict
from evals.manual import (
    Agreement,
    compare,
    disagreements,
    dump,
    render,
    render_disagreements,
    sample,
    score_disagreements,
)
from evals.run import answer_records


def test_kappa_counts_agreement_beyond_chance() -> None:
    result = Agreement(
        [
            ("by_name-001", 0, 1, True, True),
            ("by_name-001", 0, 2, True, False),
            ("by_name-002", 0, 1, False, False),
            ("by_name-002", 0, 2, False, False),
        ]
    )

    assert result.agreement == 0.75
    # Ожидаемое случайное совпадение 0.5, наблюдаемое 0.75: каппа 0.5.
    assert result.kappa == pytest.approx(0.5)


# Вопросы берутся из набора по id кейса; тест не зависит от того, что сейчас в наборе.
QUESTIONS = {"by_name-001": "как настроить почту"}


def raw_with_judge() -> dict[str, Any]:
    howto = AnswerCase(case("by_name-001"), [RUN])
    records = answer_records([howto])
    records[0]["judge"] = json.loads(
        json.dumps(Verdict([True, False], [], ["", ""], [True, True]).__dict__)
    )
    return {"meta": {}, "records": json.loads(json.dumps(records, ensure_ascii=False))}


def test_sample_is_blind_and_compare_finds_disagreement() -> None:
    raw = raw_with_judge()

    [item] = sample(raw, questions=QUESTIONS)

    assert item["id"] == "by_name-001"
    assert item["question"] == "как настроить почту"
    assert [s["supported"] for s in item["statements"]] == [None, None]
    assert "judge" not in json.dumps(item, ensure_ascii=False)

    for statement in item["statements"]:
        statement["supported"] = True
    result = compare([item], raw)
    assert result.agreement == 0.5
    assert "by_name-001, прогон 1, утверждение 2: автор True, судья False" in render(
        result, marks_file="m.yaml", raw_file="r.json"
    )


def test_unmarked_statement_is_an_error() -> None:
    raw = raw_with_judge()
    [item] = sample(raw, questions=QUESTIONS)

    with pytest.raises(ValueError, match="не размечено"):
        compare([item], raw)


def judged(supported: list[bool], *, model: str, mode: str = "auto") -> dict[str, Any]:
    """Пересчёт одного ответа судьёй `model`: у RUN два утверждения."""
    raw = raw_with_judge()
    raw["records"][0]["judge"]["supported"] = supported
    raw["records"][0]["judge"]["comments"] = [f"{model}: про {n}" for n in (1, 2)]
    raw["meta"] = {"mode": mode}
    raw["judging"] = {"model": model}
    return raw


def test_disagreements_take_each_side_and_hide_the_judges() -> None:
    ultra = judged([False, True], model="Ultra")
    sonnet = judged([True, False], model="Sonnet")

    items = disagreements([(ultra, sonnet)], questions=QUESTIONS)

    assert [(i["mode"], i["statement"], i["supported"]) for i in items] == [
        ("auto", 1, None),
        ("auto", 2, None),
    ]
    assert not items[0]["recipes"][0].startswith("Рецепт целиком")
    text = dump(items)
    assert "Ultra" not in text and "Sonnet" not in text and "про 1" not in text


def test_disagreements_take_at_most_per_side_and_skip_agreement() -> None:
    ultra = judged([False, False], model="Ultra")
    sonnet = judged([True, True], model="Sonnet")

    assert len(disagreements([(ultra, sonnet)], per_side=1, questions=QUESTIONS)) == 1
    assert disagreements([(ultra, ultra)], questions=QUESTIONS) == []


def test_pair_must_judge_the_same_answers() -> None:
    other_mode = judged([True, False], model="Sonnet", mode="always")
    with pytest.raises(ValueError, match="разные режимы"):
        disagreements([(judged([False, True], model="Ultra"), other_mode)])

    other_answer = judged([True, False], model="Sonnet")
    other_answer["records"][0]["content"] = "другой ответ"
    with pytest.raises(ValueError, match="разных ответов"):
        disagreements([(judged([False, True], model="Ultra"), other_answer)])


def test_score_shows_whom_the_author_agrees_with() -> None:
    pairs = [
        (judged([False, True], model="Ultra"), judged([True, False], model="Sonnet"))
    ]
    first, second = disagreements(pairs, questions=QUESTIONS)
    first["supported"] = False  # как A: «нет» только у A подтвердилось
    second["supported"] = True  # как A: «нет» только у B не подтвердилось
    second["note"] = "пересказ"

    disputed = score_disagreements([first, second], pairs)
    report = render_disagreements(disputed, pairs, marks_file="m.yaml")

    assert [(d.author, d.a, d.b) for d in disputed] == [
        (False, False, True),
        (True, True, False),
    ]
    assert "| всего | 2 | 2 | 0 |" in report
    assert "| A — Ultra | 1 | 1 |" in report
    assert "| B — Sonnet | 1 | 0 |" in report
    assert "  - B: Sonnet: про 2" in report
    assert "  - автор: пересказ" in report

    second["supported"] = None
    with pytest.raises(ValueError, match="не размечено утверждение 2"):
        score_disagreements([first, second], pairs)
