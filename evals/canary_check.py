"""Судья только на канарейках — быстрая проверка: `python -m evals.canary_check`.

Перед полной проверкой судьи (канарейки и ручная разметка, `evals/loop.py`) —
дешёвый замер на выборке канареек опоры на рецепт: верны ли вердикты, сколько
секунд и токенов уходит на вызов, сколько сбоев разбора и чем закончились
ответы (stop, length…). Например, сколько думать GLM (`--judge-effort`).
Выборка — поровну подтверждённых и неподтверждённых канареек, сид фиксирован.
"""

import argparse
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from assistant.config import Settings
from assistant.generation.cloudru import cost_rub
from evals.answer_metrics import cost_usd
from evals.dataset import Canary, load_canaries
from evals.judge import JUDGE_PROMPT, Claim, Judge
from evals.manual import SEED
from evals.run import REPORTS_DIR, JudgeChoice, _judge_llm

SAMPLE = 10


def pick_canaries(
    canaries: Sequence[Canary], size: int = SAMPLE, seed: int = SEED
) -> list[Canary]:
    """Поровну подтверждённых и нет; порядок — как в файле."""
    rng = random.Random(seed)
    half = size // 2
    picked = {
        c.id
        for group in (True, False)
        for c in rng.sample(
            [c for c in canaries if c.supported is group],
            min(
                half if group else size - half,
                sum(c.supported is group for c in canaries),
            ),
        )
    }
    return [c for c in canaries if c.id in picked]


@dataclass(frozen=True)
class CanaryResult:
    canary_id: str
    expected: bool
    mark: bool | None
    seconds: float


def run_check(judge: Judge, canaries: Sequence[Canary]) -> list[CanaryResult]:
    results = []
    for canary in canaries:
        started = time.monotonic()
        verdict = judge.judge(
            "—", [Claim(canary.statement, [canary.quote])], [], clarity=False
        )
        results.append(
            CanaryResult(
                canary.id,
                canary.supported,
                verdict.supported[0],
                time.monotonic() - started,
            )
        )
    return results


def render(
    label: str, results: Sequence[CanaryResult], judge: Judge, cost: str, today: date
) -> str:
    right = sum(r.mark is r.expected for r in results)
    seconds = [r.seconds for r in results]
    stats = judge.llm.stats
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(judge.finish_reasons.items()))
    lines = [
        f"# Судья на канарейках: {label} — {today}",
        "",
        f"Рубрика `{JUDGE_PROMPT}`, {len(results)} канареек опоры на рецепт "
        f"(поровну верных и неверных, сид фиксирован). `python -m evals.canary_check`.",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| верных вердиктов | {right}/{len(results)} |",
        f"| сбоев разбора | {judge.failures} |",
        f"| чем закончились ответы | {reasons} |",
        f"| секунд на вызов: медиана / максимум | {statistics.median(seconds):.1f} / {max(seconds):.1f} |",
        f"| токенов ответа (с рассуждениями) на вызов | {stats.completion_tokens // max(stats.requests, 1)} |",
        f"| цена | {cost} |",
        "",
        "| Канарейка | ожидается | судья | с |",
        "|---|---|---|---|",
    ]
    sign = {True: "да", False: "нет", None: "без вердикта"}
    lines += [
        f"| {r.canary_id} | {sign[r.expected]} | {sign[r.mark]}"
        f"{'' if r.mark is r.expected else ' ❌'} | {r.seconds:.1f} |"
        for r in results
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.canary_check", description=__doc__
    )
    parser.add_argument(
        "--judge-provider",
        choices=("anthropic", "gigachat", "cloudru"),
        default="cloudru",
    )
    parser.add_argument("--judge-model", default="")
    parser.add_argument(
        "--judge-effort",
        choices=("", "low", "medium", "high", "xhigh", "max"),
        default="",
    )
    parser.add_argument("--sample", type=int, default=SAMPLE)
    args = parser.parse_args(argv)

    choice = JudgeChoice(args.judge_provider, args.judge_model, args.judge_effort)
    canaries = pick_canaries(load_canaries(), args.sample)
    with _judge_llm(Settings(), choice) as llm:
        judge = Judge(llm, model=choice.model_name)
        results = run_check(judge, canaries)
    if choice.provider == "cloudru":
        cost = f"≈{cost_rub(choice.model_name, llm.stats) or 0:.2f} ₽"
    else:
        cost = f"≈${cost_usd(choice.model_name, llm.stats) or 0:.4f}"
    today = date.today()
    slug = choice.model_name.split("/")[-1] + (
        f"-{choice.effort_used}" if choice.effort_used else "-default"
    )
    report = REPORTS_DIR / f"{today}-canaries-{slug}.md"
    report.write_text(
        render(choice.label, results, judge, cost, today), encoding="utf-8"
    )
    right = sum(r.mark is r.expected for r in results)
    print(f"{choice.label}: верных {right}/{len(results)}, отчёт: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
