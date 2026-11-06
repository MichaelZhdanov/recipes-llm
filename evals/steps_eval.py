"""Оценка шагов приготовления: `python -m evals.steps_eval`.

Шаги пишет ИИ, когда страницы рецепта на сайте нет (`generation/steps.py`).
Две части:

1. **Канарейки** (`judge/steps_canaries.yaml`, независимый субагент): проверка
   шагов видит шаги с известным ответом. Главное — пропущенные ошибки
   (`approved: false`, а проверка пропустила): такие шаги ушли бы
   пользователю. Лишние отказы — меньшая беда: пользователь получит честное
   «не смог расписать».
2. **Выборка рецептов датасета** (сид фиксирован): весь путь — пишет модель
   ассистента, проверяет Sonnet, одна доработка. Страницы почти всех рецептов
   живы, поэтому шаги пишутся принудительно, без проверки страницы. Итоги:
   одобрено сразу, после доработки, отклонено, «нельзя расписать», время, цена.

Отчёт — `evals/reports/<дата>-steps.md`, сырые данные — `raw/<дата>-steps.json`.
Запуск идёт последовательно: так счётчики токенов не смешиваются между потоками.
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

import httpx2

from assistant.config import Settings
from assistant.generation import steps as steps_module
from assistant.generation.review import EFFORT as REVIEW_EFFORT
from assistant.generation.review import MODEL as REVIEW_MODEL
from assistant.generation.steps import (
    Check,
    Steps,
    StepsReviewer,
    StepsWriter,
    make_steps,
    make_steps_models,
)
from assistant.indexing.chunks import recipe_text
from assistant.indexing.dataset import Recipe, read_dataset
from evals.answer_metrics import Rate, cost_usd
from evals.dataset import StepsCanary, load_steps_canaries
from evals.run import REPORTS_DIR

SAMPLE = 20
SEED = 20261106
_NUMBER = re.compile(r"^\s*\d+[.)]\s*")


def split_steps(text: str) -> list[str]:
    """«1. …\\n2. …» → шаги без номеров."""
    return [_NUMBER.sub("", line).strip() for line in text.split("\n") if line.strip()]


@dataclass(frozen=True)
class CanaryResult:
    canary_id: str
    expected: bool
    approved: bool | None
    problems: list[str]
    seconds: float


def check_canaries(
    reviewer: StepsReviewer, canaries: Sequence[StepsCanary]
) -> list[CanaryResult]:
    results = []
    for canary in canaries:
        started = time.monotonic()
        mark: Check = reviewer.check(canary.recipe, split_steps(canary.steps))
        results.append(
            CanaryResult(
                canary.id,
                canary.approved,
                mark.approved,
                mark.problems,
                time.monotonic() - started,
            )
        )
        print(
            f"  {canary.id}: ждали {canary.approved}, проверка {mark.approved}",
            file=sys.stderr,
        )
    return results


def pick_recipes(
    recipes: Sequence[Recipe], size: int, seed: int = SEED
) -> list[Recipe]:
    return random.Random(seed).sample(list(recipes), size)


@dataclass(frozen=True)
class RecipeResult:
    recipe_id: int
    url: str
    name: str
    text: str
    steps: Steps


def run_sample(
    recipes: Sequence[Recipe], writer: StepsWriter, reviewer: StepsReviewer
) -> list[RecipeResult]:
    results = []
    for recipe in recipes:
        text = recipe_text(recipe)
        made = make_steps(text, writer, reviewer)
        results.append(RecipeResult(recipe.id, recipe.url, recipe.name, text, made))
        print(f"  {recipe.id}: {made.status} за {made.seconds:.1f} с", file=sys.stderr)
    return results


def _outcome(result: RecipeResult) -> str:
    made = result.steps
    if made.status == "approved":
        return "одобрено сразу" if made.revised is None else "одобрено после доработки"
    return {
        "rejected": "отклонено",
        "impossible": "нельзя расписать",
        "failed": "сбой разбора",
    }[made.status]


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def report(
    canaries: Sequence[CanaryResult],
    sample: Sequence[RecipeResult],
    *,
    writer_model: str,
    canary_cost: float,
    sample_cost: float,
) -> str:
    right = sum(r.approved is r.expected for r in canaries)
    missed = [r for r in canaries if r.expected is False and r.approved is True]
    refused = [r for r in canaries if r.expected is True and r.approved is not True]
    unparsed = sum(r.approved is None for r in canaries)
    lines = [
        f"# Шаги приготовления: проверка и выборка рецептов ({date.today()})",
        "",
        f"Пишет `{writer_model}` (`{steps_module.PROMPT_VERSION}`), проверяет "
        f"`{REVIEW_MODEL}`, effort {REVIEW_EFFORT} "
        f"(`{steps_module.REVIEW_PROMPT_VERSION}`); одна доработка. "
        "Температура у Claude не задаётся (API её не принимает).",
        "",
        "## Канарейки (`judge/steps_canaries.yaml`)",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| верные вердикты | {Rate(right, len(canaries))} |",
        f"| пропущенные ошибки (ждали «нет», проверка «да») | {len(missed)} |",
        f"| лишние отказы (ждали «да», проверка «нет») | {len(refused)} |",
        f"| сбои разбора | {unparsed} |",
        f"| время на вызов, медиана | {statistics.median(r.seconds for r in canaries):.1f} с |",
        f"| цена прогона | ≈${canary_cost:.3f} |",
        "",
    ]
    for title, group in (("Пропущенные ошибки", missed), ("Лишние отказы", refused)):
        if group:
            lines += [f"**{title}:**", ""]
            lines += [
                f"- `{r.canary_id}`: " + ("; ".join(r.problems) or "без замечаний")
                for r in group
            ]
            lines.append("")
    outcomes = Counter(_outcome(r) for r in sample)
    seconds = [r.steps.seconds for r in sample]
    lines += [
        f"## Выборка рецептов ({len(sample)}, сид {SEED})",
        "",
        "| Итог | Рецептов |",
        "|---|---|",
        *(f"| {name} | {count} |" for name, count in outcomes.most_common()),
        "",
        f"Время на рецепт: медиана {statistics.median(seconds):.1f} с, "
        f"p95 {_quantile(seconds, 0.95):.1f} с. Цена: ≈${sample_cost:.3f} за выборку, "
        f"≈${sample_cost / len(sample):.4f} за рецепт.",
        "",
        "### Шаги по рецептам",
        "",
    ]
    for r in sample:
        made = r.steps
        lines += [f"#### [{r.name}]({r.url}) — {_outcome(r)}", ""]
        if made.first is not None and made.first.problems:
            lines += [f"- замечание: {p}" for p in made.first.problems]
            lines.append("")
        lines += [f"{n}. {step}" for n, step in enumerate(made.steps, start=1)]
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.steps_eval")
    parser.add_argument("--sample", type=int, default=SAMPLE, help="рецептов в выборке")
    parser.add_argument("--name", default=f"{date.today()}-steps", help="имя отчёта")
    args = parser.parse_args(argv)

    settings = Settings()
    csv = (
        settings.data_path
        / settings.dataset_revision
        / "povarenok_recipes_2021_06_16.csv"
    )
    recipes = pick_recipes(read_dataset(csv).recipes, args.sample)
    with httpx2.Client(timeout=120) as http:
        # У канареек и выборки — свои клиенты: свои счётчики токенов и цена.
        _, canary_reviewer = make_steps_models(settings, http)
        writer, reviewer = make_steps_models(settings, http)
        print("канарейки:", file=sys.stderr)
        canaries = check_canaries(canary_reviewer, load_steps_canaries())
        canary_cost = cost_usd(canary_reviewer.model, canary_reviewer.llm.stats) or 0.0
        print("выборка:", file=sys.stderr)
        sample = run_sample(recipes, writer, reviewer)
        sample_cost = (cost_usd(writer.model, writer.llm.stats) or 0.0) + (
            cost_usd(reviewer.model, reviewer.llm.stats) or 0.0
        )
    text = report(
        canaries,
        sample,
        writer_model=writer.model,
        canary_cost=canary_cost,
        sample_cost=sample_cost,
    )
    path = REPORTS_DIR / f"{args.name}.md"
    path.write_text(text + "\n", encoding="utf-8")
    raw = REPORTS_DIR / "raw" / f"{args.name}.json"
    raw.parent.mkdir(exist_ok=True)
    raw.write_text(
        json.dumps(
            {
                "canaries": [asdict(r) for r in canaries],
                "sample": [asdict(r) for r in sample],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"отчёт: {path}\nсырые данные: {raw}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
