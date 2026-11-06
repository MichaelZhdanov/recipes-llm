"""Ручная проверка судьи: `python -m evals.manual sample|score …`.

Правило CLAUDE.md: каждую итерацию автор сам проверяет 20 случайных ответов
по той же рубрике, что у модели-судьи (`evals/prompts/judge_v2.md`), а
расхождение судьи с человеком пишется в отчёт.

1. `sample RAW_JSON` выбирает 20 случайных ответов по существу (сид
   фиксирован) и пишет YAML для разметки. Вердиктов судьи в нём нет:
   проверка слепая. В YAML есть тексты источников, поэтому он лежит
   в evals/reports/raw/ и в git не попадает.
2. Автор ставит у каждого утверждения `supported: true` или `false`.
3. `score MARKS_YAML RAW_JSON` сравнивает оценки с судьёй: доля совпадений
   и каппа Коэна (совпадения сверх случайных). Отчёт —
   evals/reports/<имя>-manual.md, без текстов источников.

Когда судей два, а их «не подтверждено» почти не пересекаются
(evals/reports/2026-11-06-stage3.md), выбор между ними решает разметка
именно расхождений:
1. `disagreements --pair A B [--pair A2 B2]` — A и B — пересчёты одних
   ответов двумя судьями (`--rescore`), по паре на режим ответа. Из каждой
   пары берутся поровну утверждения, где «нет» сказал только A и только B.
   Вердиктов, комментариев и имён судей в YAML нет.
2. `score-disagreements MARKS --pair A B …` — с кем из судей автор согласен
   и как часто подтверждается «нет» каждого. Отчёт —
   evals/reports/<имя разметки>.md.
"""

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evals.compare_judges import verdicts
from evals.dataset import HOLDOUT_DIR, load_cases
from evals.judge import JUDGE_PROMPT, claims_of
from evals.run import REPORTS_DIR, load_answer_run

SAMPLE_SIZE = 20
SEED = 20261104
# Утверждений с каждой стороны расхождения в каждом режиме: 2 режима × 2
# стороны × 8 — около 30 разметок, если «нет» у судьи меньше 8, берутся все.
PER_SIDE = 8
HEADER = f"""\
# Поставьте supported: true или false у каждого утверждения.
# Правило то же, что у судьи (evals/prompts/{JUDGE_PROMPT}.md, пункт supported):
# true, только если всё сказанное в утверждении прямо следует из фрагментов —
# имена, количества, значения, условия. Свои знания не используйте:
# важно только то, что написано во фрагментах.
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


def dataset_questions() -> dict[str, str]:
    """Вопросы кейсов по id: в сырых данных прогона их нет."""
    return {c.id: c.question for c in [*load_cases(), *load_cases(HOLDOUT_DIR)]}


def sample(
    raw: dict[str, Any],
    *,
    size: int = SAMPLE_SIZE,
    seed: int = SEED,
    questions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    questions = dataset_questions() if questions is None else questions
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
        f"Рубрика та же, что у судьи (`evals/prompts/{JUDGE_PROMPT}.md`, пункт supported), разметка слепая.",
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


DISAGREEMENTS_HEADER = f"""\
# Утверждения, о которых судьи разошлись. Поставьте supported: true или false.
# Вердиктов и имён судей здесь нет: разметка слепая. Правило то же, что у судьи
# (evals/prompts/{JUDGE_PROMPT}.md):
# - true, если всё сказанное следует из рецепта: названия, ингредиенты, количества.
#   Пересказ, другие формы слов, числа прописью («два яйца» при «2 шт») — норма;
# - false за шаг, время, температуру, подачу, совет или замену; за обобщение сильнее
#   данных («постное», «без глютена», «подходит детям»); за «без X», когда X в рецепте
#   есть; за ингредиент или количество из другого рецепта;
# - неполный список ингредиентов — не ошибка; «шагов в данных нет» и «рецепт по
#   ссылке» — верно;
# - своих знаний о блюдах не используйте, только общеизвестное о продуктах
#   (свинина — мясо, в пшеничной муке глютен).
# note — по желанию: почему так, если случай спорный.
# mode, id, run и statement не меняйте: по ним разметка сверяется с судьями.
"""

Pair = tuple[dict[str, Any], dict[str, Any]]


def _check_pair(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Режим пары; A и B должны быть пересчётами одних и тех же ответов."""
    if a["meta"]["mode"] != b["meta"]["mode"]:
        raise ValueError(
            f"в паре разные режимы: {a['meta']['mode']} и {b['meta']['mode']}"
        )
    answers_a = {(r["id"], r["run"]): r["content"] for r in a["records"]}
    answers_b = {(r["id"], r["run"]): r["content"] for r in b["records"]}
    if answers_a != answers_b:
        raise ValueError(
            "A и B — оценки разных ответов: нужны пересчёты одного прогона"
        )
    return str(a["meta"]["mode"])


def disagreements(
    pairs: Sequence[Pair],
    *,
    per_side: int = PER_SIDE,
    seed: int = SEED,
    questions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Утверждения, где «не подтверждено» сказал только один судья, поровну
    с каждой стороны, без вердиктов: автор не знает, кто что сказал."""
    questions = dataset_questions() if questions is None else questions
    rng = random.Random(seed)
    items = []
    for a, b in pairs:
        mode = _check_pair(a, b)
        left, right = verdicts(a, "supported"), verdicts(b, "supported")
        common = sorted(left.keys() & right.keys())
        records = {(r["id"], r["run"]): r for r in a["records"]}
        for says_no in (left, right):
            side = [
                key for key in common if left[key] != right[key] and not says_no[key]
            ]
            for case_id, run, number in rng.sample(side, min(per_side, len(side))):
                claim = claims_of(load_answer_run(records[(case_id, run)]))[number]
                items.append(
                    {
                        "mode": mode,
                        "id": case_id,
                        "run": run,
                        "statement": number + 1,
                        "question": questions[case_id],
                        "text": claim.text,
                        "recipes": [
                            f.removeprefix("Рецепт целиком:\n") for f in claim.fragments
                        ],
                        "supported": None,
                        "note": "",
                    }
                )
    return sorted(items, key=lambda i: (i["mode"], i["id"], i["run"], i["statement"]))


@dataclass(frozen=True)
class Disputed:
    """Утверждение, о котором судьи разошлись, с вердиктами автора и обоих судей."""

    mode: str
    case_id: str
    run: int
    statement: int
    text: str
    author: bool
    a: bool
    b: bool
    comment_a: str
    comment_b: str
    note: str


def _comment(raw: dict[str, Any], case_id: str, run: int, number: int) -> str:
    for record in raw["records"]:
        if (record["id"], record["run"]) == (case_id, run):
            comments = (record.get("judge") or {}).get("comments", [])
            return str(comments[number]) if number < len(comments) else ""
    return ""


def score_disagreements(
    marks: Sequence[dict[str, Any]], pairs: Sequence[Pair]
) -> list[Disputed]:
    by_mode = {_check_pair(a, b): (a, b) for a, b in pairs}
    disputed = []
    for item in marks:
        if item["supported"] is None:
            raise ValueError(
                f"{item['mode']} · {item['id']}, прогон {item['run']}: "
                f"не размечено утверждение {item['statement']}"
            )
        a, b = by_mode[item["mode"]]
        key = (item["id"], item["run"], item["statement"] - 1)
        disputed.append(
            Disputed(
                mode=item["mode"],
                case_id=item["id"],
                run=item["run"],
                statement=item["statement"],
                text=item["text"],
                author=bool(item["supported"]),
                a=verdicts(a, "supported")[key],
                b=verdicts(b, "supported")[key],
                comment_a=_comment(a, *key),
                comment_b=_comment(b, *key),
                note=item.get("note") or "",
            )
        )
    return disputed


def _yes_no(mark: bool) -> str:
    return "да" if mark else "нет"


def render_disagreements(
    disputed: Sequence[Disputed], pairs: Sequence[Pair], *, marks_file: str
) -> str:
    names = {(a["judging"]["model"], b["judging"]["model"]) for a, b in pairs}
    if len(names) != 1:
        raise ValueError(f"в парах разные судьи: {sorted(names)}")
    [(name_a, name_b)] = names
    lines = [
        "# Расхождения судей: ручная разметка",
        "",
        f"Разметка автора — `{marks_file}`; A — {name_a}, B — {name_b}, рубрика "
        f"`{JUDGE_PROMPT}`. Разметка слепая: вердиктов, комментариев и имён судей "
        "в файле не было. Утверждения выбраны поровну из «не подтверждено только "
        f"у A» и «только у B», до {PER_SIDE} с каждой стороны в каждом режиме, "
        "сид фиксирован.",
        "",
        "| Режим | утверждений | автор согласен с A | с B |",
        "|---|---|---|---|",
    ]
    for mode in [*sorted({d.mode for d in disputed}), "всего"]:
        rows = [d for d in disputed if mode in ("всего", d.mode)]
        with_a = sum(d.author == d.a for d in rows)
        lines.append(f"| {mode} | {len(rows)} | {with_a} | {len(rows) - with_a} |")
    lines += [
        "",
        "| «Не подтверждено» только у | утверждений | автор тоже «нет» |",
        "|---|---|---|",
    ]
    for title, side in (
        (f"A — {name_a}", [d for d in disputed if not d.a]),
        (f"B — {name_b}", [d for d in disputed if not d.b]),
    ):
        lines.append(f"| {title} | {len(side)} | {sum(not d.author for d in side)} |")
    lines += ["", "## Утверждения", ""]
    for d in disputed:
        lines.append(
            f"- {d.mode} · {d.case_id}, прогон {d.run + 1}, утверждение {d.statement}: "
            f"«{d.text}» — автор {_yes_no(d.author)}, A {_yes_no(d.a)}, "
            f"B {_yes_no(d.b)}"
        )
        lines.append(f"  - A: {d.comment_a}")
        lines.append(f"  - B: {d.comment_b}")
        if d.note:
            lines.append(f"  - автор: {d.note}")
    return "\n".join(lines) + "\n"


def _load_pairs(pairs: Sequence[Sequence[Path]]) -> list[Pair]:
    return [
        (
            json.loads(a.read_text(encoding="utf-8")),
            json.loads(b.read_text(encoding="utf-8")),
        )
        for a, b in pairs
    ]


def _pair_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pair",
        nargs=2,
        type=Path,
        action="append",
        required=True,
        metavar=("A", "B"),
        help="сырые данные двух судей на одних ответах; по паре на режим",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.manual", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make_sample = commands.add_parser("sample", help="выбрать ответы для разметки")
    make_sample.add_argument("raw", type=Path)
    score = commands.add_parser("score", help="сравнить разметку автора с судьёй")
    score.add_argument("marks", type=Path)
    score.add_argument("raw", type=Path)
    pick = commands.add_parser(
        "disagreements", help="расхождения двух судей для слепой разметки"
    )
    _pair_argument(pick)
    pick.add_argument(
        "--out", type=Path, default=REPORTS_DIR / "raw" / "judges-manual.yaml"
    )
    score_pick = commands.add_parser(
        "score-disagreements", help="с кем из двух судей согласен автор"
    )
    score_pick.add_argument("marks", type=Path)
    _pair_argument(score_pick)
    args = parser.parse_args(argv)

    if args.command == "disagreements":
        items = disagreements(_load_pairs(args.pair))
        args.out.write_text(DISAGREEMENTS_HEADER + dump(items), encoding="utf-8")
        print(f"разметьте {len(items)} утверждений: {args.out}")
        return 0
    if args.command == "score-disagreements":
        pairs = _load_pairs(args.pair)
        marks = yaml.safe_load(args.marks.read_text(encoding="utf-8"))
        disputed = score_disagreements(marks, pairs)
        report = REPORTS_DIR / f"{args.marks.stem}.md"
        report.write_text(
            render_disagreements(disputed, pairs, marks_file=str(args.marks)),
            encoding="utf-8",
        )
        with_a = sum(d.author == d.a for d in disputed)
        print(f"автор согласен с A в {with_a} из {len(disputed)}, отчёт: {report}")
        return 0

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
