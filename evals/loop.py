"""Эксперимент: два судьи другого семейства и исправление ответа — `python -m evals.loop`.

Решение автора 2026-11-06: отвечает Claude Haiku, судят DeepSeek V4 Flash и
GLM-5.3-Flash из каталога cloud.ru (рубрика та же, `judge_v4`). Утверждение
не подтверждено, если «нет» сказал хоть один судья. Тогда Haiku исправляет его
отдельным вызовом (`assistant/generation/revise.py`), и судьи проверяют только
исправленное. Кругов исправления CYCLES; что после них осталось с «нет», удаляется.
Базовая линия — удалить всё с «нет» сразу, без исправления.

1. Проверка судей: канарейки опоры (24) и ручная разметка автора
   (`evals/reports/raw/2026-11-06-judges-manual.yaml`, 30 утверждений).
2. Цикл на CASES сохранённых ответах Haiku (режим auto, первый прогон):
   поровну с «нет» у Sonnet v4 и без него — чтобы циклу было что исправлять
   и было видно, придираются ли судьи к чистым ответам. Сид фиксирован.

Отчёт — evals/reports/<дата>-loop.md, сырые данные — evals/reports/raw/.
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import yaml

from assistant.config import Settings
from assistant.generation import llm as assistant_llm
from assistant.generation.citations import CheckedStatement, SourceText
from assistant.generation.cloudru import cost_rub
from assistant.generation.revise import Remark, Reviser
from evals.answer_metrics import cost_usd
from evals.compare_judges import verdicts
from evals.dataset import load_canaries
from evals.judge import (
    JUDGE_PROMPT,
    Claim,
    Judge,
    Verdict,
    canary_scores,
    claim_of,
    judge_canaries,
)
from evals.manual import SEED, dataset_questions
from evals.run import REPORTS_DIR, JudgeChoice, _judge_llm, load_answer_run

JUDGES = ("deepseek-ai/DeepSeek-V4-Flash", "z-ai/glm-5.3-flash")
CYCLES = 2
CASES = 10
RAW = REPORTS_DIR / "raw"
ANSWERS = {m: RAW / f"2026-11-06-answers-{m}.json" for m in ("auto", "always")}
SONNET_V4 = {m: RAW / f"2026-11-06-{m}-sonnet-v4.json" for m in ("auto", "always")}
MARKS = RAW / "2026-11-06-judges-manual.yaml"


class JudgeLike(Protocol):
    model: str

    def judge(
        self,
        question: str,
        claims: Sequence[Claim],
        facts: Sequence[str],
        *,
        clarity: bool = True,
    ) -> Verdict: ...


class ReviserLike(Protocol):
    def revise(
        self,
        question: str,
        sources: Sequence[SourceText],
        statements: Mapping[int, CheckedStatement],
        remarks: Sequence[Remark],
    ) -> dict[int, CheckedStatement | None]: ...


@dataclass
class Round:
    """Один круг проверки: какие утверждения судили и что сказал каждый судья."""

    texts: dict[int, str]
    supported: dict[str, dict[int, bool | None]]
    comments: dict[str, dict[int, str]]


@dataclass
class CaseLoop:
    case_id: str
    run: int
    question: str
    original: dict[int, str]
    rounds: list[Round] = field(default_factory=list)
    # Номер → итоговый текст; None — утверждение удалено.
    final: dict[int, str | None] = field(default_factory=dict)
    revisions: dict[int, int] = field(default_factory=dict)

    @property
    def flagged_first(self) -> set[int]:
        """С «нет» хоть у одного судьи в первом круге: базовая линия их удаляет."""
        first = self.rounds[0]
        return {n for n in first.texts if not _approved(first, n)}


def _approved(round_: Round, number: int) -> bool:
    return all(marks.get(number) is True for marks in round_.supported.values())


def judge_round(
    question: str,
    sources: Sequence[SourceText],
    statements: Mapping[int, CheckedStatement],
    judges: Sequence[JudgeLike],
) -> Round:
    numbers = sorted(statements)
    claims = [claim_of(statements[n], sources) for n in numbers]
    supported: dict[str, dict[int, bool | None]] = {}
    comments: dict[str, dict[int, str]] = {}
    for judge in judges:
        verdict = judge.judge(question, claims, [], clarity=False)
        supported[judge.model] = dict(zip(numbers, verdict.supported, strict=True))
        texts = verdict.comments or [""] * len(numbers)
        comments[judge.model] = dict(zip(numbers, texts, strict=True))
    return Round({n: statements[n].text for n in numbers}, supported, comments)


def run_case(
    case_id: str,
    run: int,
    question: str,
    sources: Sequence[SourceText],
    statements: Mapping[int, CheckedStatement],
    judges: Sequence[JudgeLike],
    reviser: ReviserLike,
    cycles: int = CYCLES,
) -> CaseLoop:
    """Судьи → исправление утверждений с «нет» → судьи только по исправленным."""
    current = dict(statements)
    result = CaseLoop(case_id, run, question, {n: s.text for n, s in current.items()})
    approved: set[int] = set()
    remarks: dict[int, list[str]] = {}
    to_judge = sorted(current)
    for cycle in range(cycles + 1):
        round_ = judge_round(
            question, sources, {n: current[n] for n in to_judge}, judges
        )
        result.rounds.append(round_)
        for n in to_judge:
            if _approved(round_, n):
                approved.add(n)
                remarks.pop(n, None)
            else:
                remarks[n] = [
                    round_.comments[judge][n]
                    for judge, marks in round_.supported.items()
                    if marks.get(n) is not True and round_.comments[judge].get(n)
                ]
        if not remarks or cycle == cycles:
            break
        revised = reviser.revise(
            question,
            sources,
            current,
            [Remark(n, comments) for n, comments in sorted(remarks.items())],
        )
        to_judge = []
        for n, fixed in revised.items():
            if fixed is None:  # Haiku не исправил или цитата не прошла проверку кодом
                del current[n]
                remarks.pop(n)
            else:
                current[n] = fixed
                result.revisions[n] = result.revisions.get(n, 0) + 1
                to_judge.append(n)
        if not to_judge:
            break
    result.final = {
        n: current[n].text if n in approved and n in current else None
        for n in result.original
    }
    return result


# --- Проверка судей ---


@dataclass(frozen=True)
class JudgeCheck:
    model: str
    canaries_caught: str
    canaries_accepted: str
    # Совпадение с автором на размеченных утверждениях: (совпало, всего).
    with_author: tuple[int, int]
    # Автор «нет»: судья тоже «нет»; автор «да»: судья тоже «да».
    author_no: tuple[int, int]
    author_yes: tuple[int, int]


def _author_agreement(
    marks: Sequence[dict[str, Any]],
    verdict: Mapping[tuple[str, str, int, int], bool | None],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pairs = [
        (
            bool(m["supported"]),
            verdict.get((m["mode"], m["id"], m["run"], m["statement"] - 1)),
        )
        for m in marks
    ]
    no = [judge for author, judge in pairs if not author]
    yes = [judge for author, judge in pairs if author]
    return (
        (sum(author == judge for author, judge in pairs), len(pairs)),
        (sum(j is False for j in no), len(no)),
        (sum(j is True for j in yes), len(yes)),
    )


def check_judge(
    judge: Judge,
    marks: Sequence[dict[str, Any]],
    answers: Mapping[str, dict[str, Any]],
    questions: Mapping[str, str],
) -> JudgeCheck:
    canaries = load_canaries()
    caught, accepted = canary_scores(canaries, judge_canaries(canaries, judge))
    records = {
        (mode, r["id"], r["run"]): r
        for mode, raw in answers.items()
        for r in raw["records"]
    }
    verdict: dict[tuple[str, str, int, int], bool | None] = {}
    for mode, case_id, run in sorted({(m["mode"], m["id"], m["run"]) for m in marks}):
        answer = load_answer_run(records[(mode, case_id, run)])
        claims = [claim_of(s, answer.sources) for s in answer.checked if s.kept]
        marks_ = judge.judge(questions[case_id], claims, [], clarity=False).supported
        for number, mark in enumerate(marks_):
            verdict[(mode, case_id, run, number)] = mark
    with_author, author_no, author_yes = _author_agreement(marks, verdict)
    return JudgeCheck(
        judge.model,
        f"{caught.k}/{caught.n}",
        f"{accepted.k}/{accepted.n}",
        with_author,
        author_no,
        author_yes,
    )


def sonnet_check(marks: Sequence[dict[str, Any]]) -> JudgeCheck:
    """Sonnet v4 на тех же утверждениях — для сравнения, из уже сделанного пересчёта."""
    verdict: dict[tuple[str, str, int, int], bool | None] = {}
    for mode, path in SONNET_V4.items():
        for (case_id, run, number), mark in verdicts(
            json.loads(path.read_text(encoding="utf-8")), "supported"
        ).items():
            verdict[(mode, case_id, run, number)] = mark
    with_author, author_no, author_yes = _author_agreement(marks, verdict)
    # Канарейки Sonnet v4 — из отчётов пересчёта: 12/12 и 12/12 в обоих режимах.
    return JudgeCheck(
        "claude-sonnet-5-5 (judge_v4)",
        "12/12",
        "12/12",
        with_author,
        author_no,
        author_yes,
    )


# --- Выбор вопросов ---


def pick_records(
    answers: dict[str, Any],
    flagged: Mapping[tuple[str, int, int], bool],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Первый прогон, ответ по существу; половина — с «нет» у Sonnet v4."""
    candidates = [
        r
        for r in answers["records"]
        if r["run"] == 0
        and r["answer"]["status"] == "answer"
        and any(
            s["unknown_numbers"] == []
            and any(c["error"] is None for c in s["citations"])
            for s in r["checked"]
        )
    ]

    def has_no(record: dict[str, Any]) -> bool:
        return any(
            not ok
            for (cid, run, _), ok in flagged.items()
            if (cid, run) == (record["id"], 0)
        )

    rng = random.Random(seed)
    with_no = [r for r in candidates if has_no(r)]
    clean = [r for r in candidates if not has_no(r)]
    half = size - size // 2
    picked = rng.sample(with_no, min(half, len(with_no)))
    picked += rng.sample(clean, min(size - len(picked), len(clean)))
    return sorted(picked, key=lambda r: r["id"])


# --- Отчёт ---


def _ratio(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def render(
    checks: Sequence[JudgeCheck],
    loops: Sequence[CaseLoop],
    costs: Mapping[str, str],
    seconds: float,
    today: date,
) -> str:
    original = sum(len(c.original) for c in loops)
    baseline = original - sum(len(c.flagged_first) for c in loops)
    kept = sum(text is not None for c in loops for text in c.final.values())
    lines = [
        f"# Два судьи другого семейства и исправление ответа — {today}",
        "",
        f"Отвечает Claude Haiku 5.5 (сохранённые ответы режима auto), судьи — {', '.join(JUDGES)} "
        f"(cloud.ru, рубрика `{JUDGE_PROMPT}`, effort по умолчанию модели), исправляет Haiku "
        "(промпт `recipes_revise_v1`). «Не подтверждено», если «нет» сказал хоть один судья; "
        f"кругов исправления {CYCLES}; что осталось с «нет», удаляется. `python -m evals.loop`.",
        "",
        "## Проверка судей",
        "",
        "| Судья | канарейки: поймал неверные | принял верные | совпал с автором | автор «нет» — судья тоже | автор «да» — судья тоже |",
        "|---|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c.model} | {c.canaries_caught} | {c.canaries_accepted} | "
            f"{_ratio(c.with_author)} | {_ratio(c.author_no)} | {_ratio(c.author_yes)} |"
        )
    lines += [
        "",
        "Разметка автора — 30 утверждений, где расходились Ultra и Sonnet v3; правила "
        "`judge_v4` выведены из неё, поэтому для Sonnet v4 это не независимая проверка, "
        "а для DeepSeek и GLM — независимая (они её не видели).",
        "",
        "## Цикл",
        "",
        "| | утверждений |",
        "|---|---|",
        f"| в ответах до проверки | {original} |",
        f"| «нет» хоть у одного судьи в первом круге | {original - baseline} |",
        f"| базовая линия: удалить всё с «нет» | осталось {baseline} |",
        f"| после {CYCLES} кругов исправления | осталось {kept} |",
        f"| исправлено и принято обоими судьями | {sum(1 for c in loops for n, t in c.final.items() if t is not None and c.revisions.get(n))} |",
        "",
    ]
    for judge in JUDGES:
        first = sum(
            1
            for c in loops
            for n in c.rounds[0].texts
            if c.rounds[0].supported[judge].get(n) is not True
        )
        lines.append(f"- «нет» в первом круге у {judge}: {first}")
    rounds = max(len(c.rounds) for c in loops)
    for index in range(1, rounds):
        judged = [
            (c, r) for c in loops if len(c.rounds) > index for r in [c.rounds[index]]
        ]
        total = sum(len(r.texts) for _, r in judged)
        ok = sum(_approved(r, n) for _, r in judged for n in r.texts)
        lines.append(
            f"- круг {index + 1}: исправленных утверждений {total}, приняты обоими {ok}"
        )
    lines += ["", "## Цена и время", ""]
    lines += [f"- {name}: {value}" for name, value in costs.items()]
    lines.append(f"- всё вместе: {seconds / 60:.1f} мин")
    lines += ["", "## Утверждения с «нет»", ""]
    for case in loops:
        for n in sorted(case.flagged_first):
            final = case.final.get(n)
            lines.append(
                f"- {case.case_id} · «{case.question}», утверждение {n}: «{case.original[n]}»"
            )
            for index, round_ in enumerate(case.rounds):
                if n not in round_.texts:
                    continue
                if index:
                    lines.append(f"  - исправлено ({index}): «{round_.texts[n]}»")
                for judge in JUDGES:
                    mark = round_.supported[judge].get(n)
                    sign = {True: "да", False: "нет", None: "без вердикта"}[mark]
                    lines.append(
                        f"    - {judge}: {sign} — {round_.comments[judge].get(n, '')}"
                    )
            lines.append(f"  - итог: {'«' + final + '»' if final else 'удалено'}")
    return "\n".join(lines) + "\n"


def _record(loop: CaseLoop) -> dict[str, Any]:
    return {
        "id": loop.case_id,
        "run": loop.run,
        "question": loop.question,
        "original": loop.original,
        "rounds": [r.__dict__ for r in loop.rounds],
        "final": loop.final,
        "revisions": loop.revisions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.loop", description=__doc__)
    parser.add_argument("--cases", type=int, default=CASES)
    args = parser.parse_args(argv)

    settings = Settings()
    questions = dataset_questions()
    answers = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in ANSWERS.items()}
    marks = yaml.safe_load(MARKS.read_text(encoding="utf-8"))
    sonnet_auto = verdicts(
        json.loads(SONNET_V4["auto"].read_text(encoding="utf-8")), "supported"
    )
    started = time.monotonic()
    with (
        _judge_llm(settings, JudgeChoice("cloudru", JUDGES[0])) as first,
        _judge_llm(settings, JudgeChoice("cloudru", JUDGES[1])) as second,
        assistant_llm.make_http_client(settings) as http,
    ):
        judges = [Judge(first, model=JUDGES[0]), Judge(second, model=JUDGES[1])]
        haiku = assistant_llm.make_chat_model(settings, http)
        reviser = Reviser(haiku, model=assistant_llm.llm_model(settings))
        checks = []
        for judge in judges:
            print(
                f"проверка судьи {judge.model}: канарейки и разметка автора", flush=True
            )
            checks.append(check_judge(judge, marks, answers, questions))
        checks.append(sonnet_check(marks))
        loops = []
        for record in pick_records(answers["auto"], sonnet_auto, args.cases, SEED):
            print(f"цикл: {record['id']}", flush=True)
            answer = load_answer_run(record)
            kept = [s for s in answer.checked if s.kept]
            loops.append(
                run_case(
                    record["id"],
                    record["run"],
                    questions[record["id"]],
                    answer.sources,
                    dict(enumerate(kept, start=1)),
                    judges,
                    reviser,
                )
            )
        costs = {
            judge.model: f"{judge.llm.stats.requests} запросов, "
            f"≈{cost_rub(judge.model, judge.llm.stats) or 0:.2f} ₽, сбоев разбора {judge.failures}"
            for judge in judges
        }
        usd = cost_usd(reviser.model, haiku.stats) or 0
        costs[reviser.model] = (
            f"{haiku.stats.requests} исправлений, ≈${usd:.4f}, сбоев разбора {reviser.failures}"
        )
    seconds = time.monotonic() - started
    today = date.today()
    raw_path = RAW / f"{today}-loop.json"
    raw_path.write_text(
        json.dumps(
            {
                "checks": [c.__dict__ for c in checks],
                "loops": [_record(c) for c in loops],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    report = REPORTS_DIR / f"{today}-loop.md"
    report.write_text(render(checks, loops, costs, seconds, today), encoding="utf-8")
    print(f"отчёт: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
