"""Ручная проверка судьи: `python -m evals.manual sample|score …`.

Правило CLAUDE.md: каждую итерацию автор сам проверяет 20 случайных ответов
по той же рубрике, что у модели-судьи (`evals/prompts/judge_v1.md`), а
расхождение судьи с человеком пишется в отчёт.

1. `sample RAW_JSON` выбирает 20 случайных ответов по существу (сид
   фиксирован) и пишет YAML для разметки. Вердиктов судьи в нём нет:
   проверка слепая. В YAML есть строки документации Paperless, поэтому он
   лежит в evals/reports/raw/ и в git не попадает.
2. Автор ставит у каждого утверждения `supported: true` или `false`.
3. `score MARKS_YAML RAW_JSON` сравнивает оценки с судьёй: доля совпадений
   и каппа Коэна (совпадения сверх случайных). Отчёт —
   evals/reports/<имя>-manual.md, без текстов документации.
"""

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evals.dataset import HOLDOUT_DIR, load_cases
from evals.judge import JUDGE_PROMPT, claims_of
from evals.run import REPORTS_DIR, load_answer_run

SAMPLE_SIZE = 20
SEED = 20261104
HEADER = f"""\
# Поставьте supported: true или false у каждого утверждения.
# Правило то же, что у судьи (evals/prompts/{JUDGE_PROMPT}.md, пункт supported):
# true, только если всё сказанное в утверждении прямо следует из фрагментов —
# имена параметров, значения, шаги, условия. Свои знания о Paperless не
# используйте: важно только то, что написано во фрагментах.
# id и run (номер прогона, с нуля) не меняйте: по ним разметка сверяется с судьёй.
"""


class _Dumper(yaml.SafeDumper):
    """Многострочные фрагменты — блоком `|`: так их удобно читать в редакторе."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _represent_str)


def dump(items: Sequence[dict[str, Any]]) -> str:
    return yaml.dump(
        items, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100
    )


def sample(
    raw: dict[str, Any], *, size: int = SAMPLE_SIZE, seed: int = SEED
) -> list[dict[str, Any]]:
    questions = {c.id: c.question for c in [*load_cases(), *load_cases(HOLDOUT_DIR)]}
    answered = [r for r in raw["records"] if r["answer"]["status"] == "answer"]
    picked = random.Random(seed).sample(answered, min(size, len(answered)))
    items = []
    for record in sorted(picked, key=lambda r: (r["id"], r["run"])):
        claims = claims_of(load_answer_run(record))
        items.append(
            {
                "id": record["id"],
                "run": record["run"],
                "question": questions[record["id"]],
                "statements": [
                    {"text": c.text, "fragments": c.fragments, "supported": None}
                    for c in claims
                ],
            }
        )
    return items


@dataclass(frozen=True)
class Agreement:
    pairs: list[tuple[str, int, int, bool, bool]]  # id, прогон, №, автор, судья

    @property
    def agreement(self) -> float:
        return sum(a == j for *_, a, j in self.pairs) / len(self.pairs)

    @property
    def kappa(self) -> float:
        """Каппа Коэна: 1 — полное согласие, 0 — не лучше случайного."""
        n = len(self.pairs)
        observed = self.agreement
        author_yes = sum(a for *_, a, _ in self.pairs) / n
        judge_yes = sum(j for *_, j in self.pairs) / n
        expected = author_yes * judge_yes + (1 - author_yes) * (1 - judge_yes)
        return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def compare(marks: Sequence[dict[str, Any]], raw: dict[str, Any]) -> Agreement:
    verdicts = {
        (r["id"], r["run"]): (r.get("judge") or {}).get("supported", [])
        for r in raw["records"]
    }
    pairs = []
    for item in marks:
        judged = verdicts[(item["id"], item["run"])]
        for number, statement in enumerate(item["statements"]):
            author = statement["supported"]
            judge = judged[number] if number < len(judged) else None
            if author is None:
                raise ValueError(
                    f"{item['id']}, прогон {item['run']}: не размечено утверждение {number + 1}"
                )
            if judge is not None:
                pairs.append(
                    (item["id"], item["run"], number + 1, bool(author), bool(judge))
                )
    return Agreement(pairs)


def render(result: Agreement, *, marks_file: str, raw_file: str) -> str:
    author_no = sum(not a for *_, a, _ in result.pairs)
    lines = [
        "# Ручная проверка судьи",
        "",
        f"Разметка автора — `{marks_file}`, ответы и вердикты судьи — `{raw_file}`.",
        "Рубрика та же, что у судьи (`evals/prompts/judge_v1.md`), разметка слепая.",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| утверждений | {len(result.pairs)} |",
        f"| автор: не подтверждается | {author_no} |",
        f"| совпадение с судьёй | {result.agreement:.2f} |",
        f"| каппа Коэна | {result.kappa:.2f} |",
        "",
        "## Расхождения",
        "",
    ]
    differ = [p for p in result.pairs if p[3] != p[4]]
    lines += [
        f"- {case_id}, прогон {run + 1}, утверждение {number}: автор {author}, судья {judge}"
        for case_id, run, number, author, judge in differ
    ] or ["Нет."]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.manual", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make_sample = commands.add_parser("sample", help="выбрать ответы для разметки")
    make_sample.add_argument("raw", type=Path)
    score = commands.add_parser("score", help="сравнить разметку автора с судьёй")
    score.add_argument("marks", type=Path)
    score.add_argument("raw", type=Path)
    args = parser.parse_args(argv)

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    if args.command == "sample":
        out = REPORTS_DIR / "raw" / f"{args.raw.stem}-manual.yaml"
        out.write_text(HEADER + dump(sample(raw)), encoding="utf-8")
        print(f"разметьте {out}")
        return 0
    marks = yaml.safe_load(args.marks.read_text(encoding="utf-8"))
    result = compare(marks, raw)
    report = REPORTS_DIR / f"{args.raw.stem}-manual.md"
    report.write_text(
        render(result, marks_file=str(args.marks), raw_file=str(args.raw)),
        encoding="utf-8",
    )
    print(
        f"совпадение {result.agreement:.2f}, каппа {result.kappa:.2f}, отчёт: {report}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
