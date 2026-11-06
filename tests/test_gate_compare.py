"""Сравнение гейтов: сбой кандидата не роняет отчёт, правило — абсолютные пороги."""

from datetime import date

from evals.gate_compare import GateResult, render, verdicts

MARKINGS: dict[str, list[dict[str, object]]] = {}


def test_broken_candidate_is_shown_and_does_not_break_the_report() -> None:
    whole = GateResult("good", "cloudru", calls=10)
    whole.canaries = {
        "опора": (47, 47, []),
        "релевантность": (34, 34, []),
        "шаги": (24, 24, []),
    }
    broken = GateResult(
        "broken", "cloudru", calls=5, error="LlmError: 403 Project not found"
    )
    broken.canaries = {"опора": (3, 3, [])}

    text = render([whole, broken], MARKINGS, date(2026, 11, 6))
    failed = verdicts([whole, broken], MARKINGS)

    assert "broken (оборвался: LlmError: 403 Project not found)" in text
    assert "| 3/3 | — | — |" in text
    assert failed["broken"] == ["сбой: LlmError: 403 Project not found"]
    # Без разметки 0 совпадений из 0 — ниже порога: правило не пропустит пустой прогон.
    assert "1: разметка" in failed["good"]
