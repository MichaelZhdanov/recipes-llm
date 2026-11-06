"""Быстрая проверка судьи на канарейках: выборка и отчёт."""

import json
from collections.abc import Mapping, Sequence
from datetime import date

from assistant.generation.gigachat import (
    Completion,
    FunctionCallMode,
    LlmStats,
    Message,
    Usage,
)
from evals.canary_check import CanaryResult, pick_canaries, render, run_check
from evals.dataset import Canary, load_canaries
from evals.judge import Judge


def test_sample_is_balanced_repeatable_and_in_file_order() -> None:
    canaries = load_canaries()

    picked = pick_canaries(canaries, 10)

    assert sum(c.supported for c in picked) == 5
    assert len(picked) == 10
    assert [c.id for c in picked] == [c.id for c in pick_canaries(canaries, 10)]
    order = [c.id for c in canaries]
    assert [c.id for c in picked] == sorted((c.id for c in picked), key=order.index)


class AlwaysNo:
    def __init__(self) -> None:
        self.stats = LlmStats()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Mapping[str, object] | None = None,
        functions: Sequence[Mapping[str, object]] | None = None,
        function_call: FunctionCallMode | None = None,
    ) -> Completion:
        content = json.dumps(
            {
                "statements": [{"number": 1, "comment": "…", "supported": False}],
                "facts": [],
            }
        )
        usage = Usage(10, 40, 0, 50)
        self.stats.add(usage)
        return Completion(content, "stop", model, usage)


def test_report_shows_misses_finish_reasons_and_tokens() -> None:
    canaries = [
        Canary(id="canary-001", statement="а", quote="б", supported=True, notes=""),
        Canary(id="canary-002", statement="в", quote="г", supported=False, notes=""),
    ]
    judge = Judge(AlwaysNo(), model="z-ai/glm-5.3-flash")

    results = run_check(judge, canaries)
    report = render("glm, effort low", results, judge, "≈0.01 ₽", date(2026, 11, 6))

    assert [r.mark for r in results] == [False, False]
    assert "| верных вердиктов | 1/2 |" in report
    assert "| чем закончились ответы | stop 2 |" in report
    assert "| токенов ответа (с рассуждениями) на вызов | 40 |" in report
    assert "| canary-001 | да | нет ❌ |" in report
    assert isinstance(results[0], CanaryResult)
