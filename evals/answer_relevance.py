"""Рецепты итогового ответа — по вопросу ли они: `python -m evals.answer_relevance RAW…`.

На «что подать к красному вину» бот ответил блюдами, тушёнными в вине: всё
сказанное о них — правда, но это не ответ. groundedness такого не видит,
`semantic_relevance@5` судит выдачу поиска, а не то, что дошло до
пользователя. Эта оценка судит рецепты итогового ответа тем же судьёй
релевантности (`relevance.py`, Sonnet) — по сырым данным одного или
нескольких прогонов `make eval`, рядом: «до» и «после».

Кейсы — вопросы «по смыслу» (semantic) и уточнения без условий (follow_up
без include, exclude и названия: иначе вопрос «а без лука?» без диалога
не понять). Судится первый прогон кейса. Судье тоже нельзя верить на
слово: канарейки релевантности (`judge/relevance_canaries.yaml`) идут в тот
же отчёт.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from assistant.config import Settings
from evals.answer_metrics import Rate
from evals.dataset import Case, load_cases, load_relevance_canaries
from evals.relevance import RelevanceJudge, judge_relevance, relevance_canary_scores
from evals.retrieval_metrics import Found
from evals.run import DEFAULT_JUDGE, REPORTS_DIR, _judge_llm


def in_scope(case: Case) -> bool:
    if case.category == "semantic":
        return True
    return case.category == "follow_up" and not (
        case.expected_include or case.expected_exclude or case.expected_name
    )


@dataclass(frozen=True)
class Answered:
    """Первый прогон кейса: что ответил ассистент и какие рецепты назвал."""

    case_id: str
    question: str
    status: str
    recipes: list[Found]
    new_search: bool


def answered(
    records: Sequence[dict[str, Any]], cases: dict[str, Case]
) -> list[Answered]:
    result = []
    for record in records:
        case = cases.get(record["id"])
        if record["run"] != 0 or case is None or not in_scope(case):
            continue
        chunks = {s["chunk"]["recipe_id"]: s["chunk"] for s in record["sources"]}
        recipes = [
            Found(
                recipe_id=link["recipe_id"],
                name=chunks[link["recipe_id"]]["name"],
                ingredients=tuple(chunks[link["recipe_id"]]["ingredients"]),
                score=0.0,
            )
            for link in record["answer"]["sources"]
            if link["recipe_id"] in chunks
        ]
        if record["answer"]["status"] != "answer":
            recipes = []  # у «не нашёл» — подсказки, а не ответ
        result.append(
            Answered(
                case_id=case.id,
                question=case.question,
                status=record["answer"]["status"],
                recipes=recipes,
                new_search=bool(record.get("first_try")),
            )
        )
    return sorted(result, key=lambda a: a.case_id)


@dataclass(frozen=True)
class Scored:
    label: str
    answers: list[Answered]
    # id кейса → вердикт судьи по каждому рецепту ответа.
    marks: dict[str, list[bool | None]]

    def relevant(self) -> Rate:
        marks = [m for kase in self.marks.values() for m in kase if m is not None]
        return Rate(sum(marks), len(marks))

    def all_relevant(self) -> Rate:
        given = [a for a in self.answers if a.recipes]
        return Rate(
            sum(all(m is True for m in self.marks[a.case_id]) for a in given),
            len(given),
        )


def score(label: str, answers: Sequence[Answered], judge: RelevanceJudge) -> Scored:
    marks = {a.case_id: judge.judge(a.question, a.recipes) for a in answers}
    return Scored(label, list(answers), marks)


def report(scored: Sequence[Scored], canaries: tuple[Rate, Rate], model: str) -> str:
    head = " | ".join(s.label for s in scored)
    lines = [
        f"# Рецепты ответа по вопросу ({date.today()})",
        "",
        f"Судья релевантности `{model}` судит рецепты итогового ответа (первый прогон): "
        "вопросы semantic и уточнения follow_up без условий.",
        "",
        f"| Метрика | {head} |",
        "|---|" + "---|" * len(scored),
        "| рецепты ответа по вопросу | "
        + " | ".join(str(s.relevant()) for s in scored)
        + " |",
        "| ответов, где все рецепты по вопросу | "
        + " | ".join(str(s.all_relevant()) for s in scored)
        + " |",
        "| «не нашёл» | "
        + " | ".join(
            f"{sum(a.status != 'answer' for a in s.answers)} из {len(s.answers)}"
            for s in scored
        )
        + " |",
        "| ответов с новым поиском | "
        + " | ".join(str(sum(a.new_search for a in s.answers)) for s in scored)
        + " |",
        "",
        f"Канарейки судьи релевантности: неподходящие отвергнуты {canaries[0]}, "
        f"подходящие приняты {canaries[1]}.",
    ]
    for s in scored:
        lines += ["", f"## {s.label}", ""]
        for a in s.answers:
            extra = " (новый поиск)" if a.new_search else ""
            if not a.recipes:
                lines.append(f"- `{a.case_id}` «{a.question}» — {a.status}{extra}")
                continue
            names = ", ".join(
                f"{'✅' if m is True else '❌' if m is False else '?'} {r.name}"
                for r, m in zip(a.recipes, s.marks[a.case_id], strict=True)
            )
            lines.append(f"- `{a.case_id}` «{a.question}»{extra}: {names}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.answer_relevance")
    parser.add_argument("raw", type=Path, nargs="+", help="сырые данные make eval")
    parser.add_argument(
        "--labels", nargs="+", help="подписи прогонов в отчёте, по порядку"
    )
    parser.add_argument("--name", default=f"{date.today()}-answer-relevance")
    args = parser.parse_args(argv)
    labels = args.labels or [p.stem for p in args.raw]
    if len(labels) != len(args.raw):
        parser.error("подписей должно быть столько же, сколько файлов")

    cases = {c.id: c for c in load_cases()}
    runs = [
        answered(json.loads(path.read_text(encoding="utf-8"))["records"], cases)
        for path in args.raw
    ]
    with _judge_llm(Settings(), DEFAULT_JUDGE) as llm:
        judge = RelevanceJudge(llm, model=DEFAULT_JUDGE.model_name)
        scored = [
            score(label, run, judge) for label, run in zip(labels, runs, strict=True)
        ]
        canaries = relevance_canary_scores(
            judge_relevance([], judge, load_relevance_canaries())
        )
    text = report(scored, canaries, DEFAULT_JUDGE.label)
    path = REPORTS_DIR / f"{args.name}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"отчёт: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
