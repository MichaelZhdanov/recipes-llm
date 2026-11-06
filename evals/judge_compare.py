"""Судьи на ручной разметке автора и канарейках — `python -m evals.judge_compare`.

Каждый судья с одной и той же рубрикой (`JUDGE_PROMPT`) оценивает все
утверждения, которые автор разметил вслепую (`evals/reports/raw/*-manual.yaml`),
и канарейки опоры на рецепт. Отчёт: совпадение с автором по каждой разметке,
канарейки, секунды на вызов, сбои разбора, чем закончились ответы, цена.

Рубрика выведена из этих же разметок, поэтому абсолютные цифры завышены
одинаково для всех судей; сравнивать судей между собой на них можно.
Судьи работают параллельно: у каждого свой HTTP-клиент.
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import yaml

from assistant.config import Settings
from assistant.generation.cloudru import cost_rub
from evals.answer_metrics import cost_usd
from evals.dataset import load_canaries
from evals.judge import JUDGE_PROMPT, Judge, canary_scores, claims_of, judge_canaries
from evals.manual import dataset_questions
from evals.run import REPORTS_DIR, JudgeChoice, _judge_llm, load_answer_run

RAW = REPORTS_DIR / "raw"
_V1 = {
    "auto": "2026-11-06-answers-auto.json",
    "always": "2026-11-06-answers-always.json",
}
# Разметка → сырые ответы, к которым она относится (по режимам).
MARKINGS: dict[str, dict[str, str]] = {
    "2026-11-06-judges-manual.yaml": _V1,
    "2026-11-06-judge-v4-manual.yaml": _V1,
    "2026-11-06-judge-v7-manual.yaml": {"always": "2026-11-06-always-v2.json"},
}
JUDGES = (
    JudgeChoice("anthropic"),
    JudgeChoice("cloudru", "z-ai/glm-5.3-flash", "high"),
)

Key = tuple[str, str, int]  # файл ответов, кейс, прогон


@dataclass
class JudgeResult:
    label: str
    # (файл ответов, кейс, прогон, номер утверждения с 0) → вердикт
    verdicts: dict[tuple[str, str, int, int], bool | None] = field(default_factory=dict)
    seconds: list[float] = field(default_factory=list)
    canaries: str = ""
    failures: int = 0
    finish: str = ""
    cost: str = ""


def answers_to_judge(markings: Mapping[str, Sequence[dict[str, Any]]]) -> list[Key]:
    """Ответы, в которых есть размеченные утверждения, без повторов."""
    keys = {
        (MARKINGS[name][m["mode"]], m["id"], m["run"])
        for name, marks in markings.items()
        for m in marks
    }
    return sorted(keys)


def agreement(
    marks: Sequence[dict[str, Any]],
    files: Mapping[str, str],
    verdicts: Mapping[tuple[str, str, int, int], bool | None],
) -> tuple[int, int, int, int, int]:
    """Совпало, всего, автор «нет» и судья тоже, автор «нет», автор «да» и судья тоже."""
    same = no_caught = no_total = yes_kept = 0
    for m in marks:
        mark = verdicts.get((files[m["mode"]], m["id"], m["run"], m["statement"] - 1))
        same += mark is m["supported"]
        if m["supported"]:
            yes_kept += mark is True
        else:
            no_total += 1
            no_caught += mark is False
    return same, len(marks), no_caught, no_total, yes_kept


def run_judge(
    choice: JudgeChoice,
    keys: Sequence[Key],
    records: Mapping[Key, dict[str, Any]],
    questions: Mapping[str, str],
) -> JudgeResult:
    result = JudgeResult(choice.label)
    with _judge_llm(Settings(), choice) as llm:
        judge = Judge(llm, model=choice.model_name)
        for key in keys:
            run = load_answer_run(records[key])
            started = time.monotonic()
            verdict = judge.judge(questions[key[1]], claims_of(run), [], clarity=False)
            result.seconds.append(time.monotonic() - started)
            for number, mark in enumerate(verdict.supported):
                result.verdicts[(*key, number)] = mark
        caught, accepted = canary_scores(
            load_canaries(), judge_canaries(load_canaries(), judge)
        )
        result.canaries = f"{caught.k}/{caught.n} · {accepted.k}/{accepted.n}"
        result.failures = judge.failures
        result.finish = ", ".join(
            f"{k} {v}" for k, v in sorted(judge.finish_reasons.items())
        )
        if choice.provider == "cloudru":
            result.cost = f"≈{cost_rub(choice.model_name, llm.stats) or 0:.2f} ₽"
        else:
            result.cost = f"≈${cost_usd(choice.model_name, llm.stats) or 0:.3f}"
    return result


def render(
    results: Sequence[JudgeResult],
    markings: Mapping[str, Sequence[dict[str, Any]]],
    today: date,
) -> str:
    names = " | ".join(r.label for r in results)
    lines = [
        f"# Судьи на разметке автора и канарейках — {today}",
        "",
        f"Рубрика `{JUDGE_PROMPT}` у всех судей. Разметки: "
        + ", ".join(f"`{name}`" for name in markings)
        + ". Правила рубрики выведены из этих разметок — цифры завышены одинаково "
        "для всех судей, сравнивать их между собой можно. `python -m evals.judge_compare`.",
        "",
        f"| | {names} |",
        "|---|" + "---|" * len(results),
    ]
    totals = [[0, 0, 0, 0, 0] for _ in results]
    for name, marks in markings.items():
        cells = []
        for index, r in enumerate(results):
            same, total, caught, no_total, kept = agreement(
                marks, MARKINGS[name], r.verdicts
            )
            for i, v in enumerate((same, total, caught, no_total, kept)):
                totals[index][i] += v
            cells.append(f"{same}/{total}")
        lines.append(f"| совпал с автором: `{name}` | " + " | ".join(cells) + " |")
    lines.append(
        "| **совпал с автором: всего** | "
        + " | ".join(f"**{t[0]}/{t[1]}**" for t in totals)
        + " |"
    )
    lines.append(
        "| автор «нет» — судья тоже | "
        + " | ".join(f"{t[2]}/{t[3]}" for t in totals)
        + " |"
    )
    lines.append(
        "| автор «да» — судья тоже | "
        + " | ".join(f"{t[4]}/{t[1] - t[3]}" for t in totals)
        + " |"
    )
    lines.append(
        "| канарейки: поймал неверные · принял верные | "
        + " | ".join(r.canaries for r in results)
        + " |"
    )
    lines.append(
        "| секунд на ответ: медиана / максимум | "
        + " | ".join(
            f"{statistics.median(r.seconds):.1f} / {max(r.seconds):.1f}"
            for r in results
        )
        + " |"
    )
    lines.append(
        "| сбоев разбора | " + " | ".join(str(r.failures) for r in results) + " |"
    )
    lines.append(
        "| чем закончились ответы | " + " | ".join(r.finish for r in results) + " |"
    )
    lines.append("| цена прогона | " + " | ".join(r.cost for r in results) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.judge_compare", description=__doc__
    )
    parser.parse_args(argv)
    markings = {
        name: yaml.safe_load((RAW / name).read_text(encoding="utf-8"))
        for name in MARKINGS
    }
    keys = answers_to_judge(markings)
    files = {key[0] for key in keys}
    data = {f: json.loads((RAW / f).read_text(encoding="utf-8")) for f in files}
    records = {
        (f, r["id"], r["run"]): r for f, raw in data.items() for r in raw["records"]
    }
    questions = dataset_questions()
    print(f"ответов: {len(keys)}, судей: {len(JUDGES)}", flush=True)
    with ThreadPoolExecutor(max_workers=len(JUDGES)) as pool:
        results = list(
            pool.map(lambda c: run_judge(c, keys, records, questions), JUDGES)
        )
    today = date.today()
    report = REPORTS_DIR / f"{today}-judges-compare.md"
    report.write_text(render(results, markings, today), encoding="utf-8")
    print(f"отчёт: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
