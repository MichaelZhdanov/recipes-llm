"""Оценка шагов: разбор шагов канарейки и итоги в отчёте."""

from assistant.generation.steps import Check, Steps
from evals.steps_eval import CanaryResult, RecipeResult, report, split_steps


def test_split_steps_drops_numbers_and_blank_lines() -> None:
    text = "1. Взбейте яйца.\n\n2) Добавьте 2 ст. л. муки.\n3. Жарьте."

    assert split_steps(text) == ["Взбейте яйца.", "Добавьте 2 ст. л. муки.", "Жарьте."]


def test_report_separates_missed_errors_from_extra_refusals() -> None:
    canaries = [
        CanaryResult("steps-001", True, True, [], 3.0),
        CanaryResult("steps-002", True, False, ["лишняя зелень"], 3.0),
        CanaryResult("steps-013", False, True, [], 3.0),
        CanaryResult("steps-014", False, False, ["180 °C"], 3.0),
    ]
    sample = [
        RecipeResult(
            1,
            "https://x/1/",
            "Блины",
            "Блины",
            Steps("approved", 10.0, steps=["Жарьте."]),
        ),
        RecipeResult(
            2,
            "https://x/2/",
            "Суп",
            "Суп",
            Steps(
                "approved",
                20.0,
                steps=["Варите."],
                first=Check(False, ["время"]),
                revised=["Варите."],
            ),
        ),
        RecipeResult(
            3, "https://x/3/", "Огурцы на зиму", "…", Steps("impossible", 4.0)
        ),
    ]

    text = report(
        canaries,
        sample,
        writer_model="claude-haiku-5-5",
        canary_cost=0.1,
        sample_cost=0.3,
    )

    assert "| верные вердикты | 0.50 (2/4," in text
    assert "| пропущенные ошибки (ждали «нет», проверка «да») | 1 |" in text
    assert "- `steps-013`: без замечаний" in text
    assert "- `steps-002`: лишняя зелень" in text
    assert (
        "| одобрено сразу | 1 |" in text and "| одобрено после доработки | 1 |" in text
    )
    assert "| нельзя расписать | 1 |" in text
    assert "- замечание: время" in text
