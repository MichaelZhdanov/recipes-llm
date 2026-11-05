"""Сравнение двух судей на одних ответах: канарейки, согласие, каппа Коэна."""

import math

import pytest

from evals.compare_judges import compare, kappa


def test_kappa_is_one_for_full_agreement_and_zero_for_chance() -> None:
    assert kappa([(True, True), (False, False)]) == 1.0
    # Совпали 2 из 4 при 50/50 «да» у обоих — ровно как случайно.
    assert kappa([(True, True), (True, False), (False, True), (False, False)]) == 0.0
    assert math.isnan(kappa([]))
    assert math.isnan(kappa([(True, True), (True, True)]))


def raw(
    model: str, supported: list[bool | None], canary_mark: bool
) -> dict[str, object]:
    statement = {
        "text": "Нужно 250 мл молока.",
        "citations": [{"error": None}],
        "unknown_numbers": [],
    }
    return {
        "records": [
            {
                "id": "include-001",
                "run": 0,
                "checked": [statement] * len(supported),
                "judge": {
                    "supported": supported,
                    "clear": [True] * len(supported),
                    "comments": [f"{model}: ок"] * len(supported),
                },
            }
        ],
        "judging": {
            "model": model,
            "canaries": [{"id": "canary-013", "expected": False, "mark": canary_mark}],
            "clarity_canaries": [
                {"id": "clarity-007", "expected": False, "mark": False}
            ],
        },
    }


def test_compare_reports_canaries_agreement_and_disagreements() -> None:
    text = compare(
        raw("GigaChat-3-Ultra", [True, True, None], canary_mark=False),
        raw("claude-sonnet-5-5", [True, False, True], canary_mark=True),
    )

    assert text.startswith("A — GigaChat-3-Ultra, B — claude-sonnet-5-5")
    assert (
        "- опора на рецепт: A — поймал неверных 1/1, принял верных 0/0; B — поймал неверных 0/1"
        in text
    )
    assert "  - canary-013 (ожидалось False): A False, B True" in text
    # Третье утверждение без вердикта у A в согласие не идёт.
    assert "Утверждения, опора на рецепт: совпало 1 из 2" in text
    assert "- include-001 (прогон 0): «Нужно 250 мл молока.» — A True, B False" in text


@pytest.mark.parametrize("missing", ["a", "b"])
def test_compare_without_canaries_says_so(missing: str) -> None:
    a, b = raw("A", [True], False), raw("B", [True], False)
    (a if missing == "a" else b).pop("judging")

    assert "опора на рецепт: нет данных у одного из судей" in compare(a, b)
