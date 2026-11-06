"""Узкий гейт против вердиктов судьи — `python -m evals.gate_narrow TASK`.

Ответ пишет GLM-5.1 low (evals/reports/2026-11-07-writers.md). У него 14
нарушений на 243 утверждения по вердиктам субагента Sonnet. Гейт за ним —
дешёвая модель с узкой рубрикой `review_narrow_v1`: короткий список
проверяемых нарушений вместо полной рубрики судьи.

Проверяется на тех же ответах. TASK — каталог задания судьи-субагента
(`evals.subagent_judge export`) с его вердиктами (`verdicts*.json`), они и
есть эталон. Гейт видит то же, что в пути ответа: вопрос, прошлые вопросы
диалога и все утверждения ответа с рецептами целиком — один вызов на ответ.
Отдельно канарейки опоры и релевантности, по одной на вызов: они показывают,
что узкая рубрика пропускает за пределами своего списка.

Правило выбора записано до прогона (RULE). Каждый кандидат сохраняется
сразу (`raw/<дата>-narrow-<модель>.json`), отчёт собирается из всех
кандидатов, сохранённых сегодня.
"""

import argparse
import dataclasses
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from assistant.config import Settings
from assistant.generation.cloudru import cost_rub
from assistant.generation.review import Item, Mark, Reviewer
from evals.dataset import load_canaries, load_relevance_canaries
from evals.gate_compare import _choice, _slug
from evals.judge_compare import RAW
from evals.run import REPORTS_DIR, JudgeChoice, _clock_text, _judge_llm

PROMPT = "review_narrow_v1"
# Пороги из RULE: сколько нарушений эталона поймать и доля зря отправленных
# на доработку верных утверждений.
MIN_CAUGHT = 10
MAX_FALSE_SHARE = 0.05
MAX_FAILURE_SHARE = 0.02
RULE = """\
Эталон — вердикты субагента Sonnet (effort high, рубрика `judge_v10`; на
канарейках 47/47 и 34/34) на ответах GLM-5.1 low: 14 нарушений из 243
утверждений. «Отправил на доработку» — как в пути ответа: вердикт не «да»
(«нет» или ответ не разобрался). Кандидат проходит, если выполнено всё сразу:
1. отправил на доработку не меньше 10 из 14 нарушений;
2. зря отправил не больше 5 % верных утверждений (11 из 229);
3. сбоев разбора — не больше 2 % вызовов.
Из прошедших — самый дешёвый. Релевантность, канарейки и задержка — в
отчёте, но не отсекают."""


@dataclass(frozen=True)
class Answer:
    """Ответ из задания судьи и вердикты эталона по нему."""

    item: str
    case: str
    question: str
    dialog: list[str]
    items: list[Item]
    # Эталон по утверждениям: опирается ли на рецепт.
    supported: list[bool]
    # Отвечает ли рецепт утверждения на вопрос; None — вступление или
    # рецепт не нашёлся среди рецептов ответа.
    relevant: list[bool | None]
    comments: list[str]


def _recipe_name(fragment: str) -> str:
    """Первая строка рецепта — название; фрагмент начинается с «Рецепт целиком:»."""
    lines = fragment.split("\n")
    return lines[1] if lines[0].startswith("Рецепт целиком") else lines[0]


def load_task(task: Path) -> list[Answer]:
    key: dict[str, dict[str, Any]] = json.loads(
        (task / "key.json").read_text(encoding="utf-8")
    )
    verdicts: dict[str, dict[str, Any]] = {}
    for path in sorted(task.glob("verdicts*.json")):
        for v in json.loads(path.read_text(encoding="utf-8")):
            verdicts[v["item"]] = v
    answers = []
    for path in sorted(task.glob("part*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            meaning = key[item["item"]]
            verdict = verdicts.get(item["item"])
            if meaning["kind"] != "case" or not item["statements"] or not verdict:
                continue
            by_n = {s["n"]: s for s in verdict["statements"]}
            fits = {r["n"]: r["relevant"] for r in verdict["recipes"]}
            names = {r["name"]: fits.get(r["n"]) for r in item["recipes"]}
            relevant: list[bool | None] = []
            for s in item["statements"]:
                one = len(s["recipes"]) == 1 and not s["intro"]
                relevant.append(
                    names.get(_recipe_name(s["recipes"][0])) if one else None
                )
            answers.append(
                Answer(
                    item=item["item"],
                    case=meaning["id"],
                    question=item["question"],
                    dialog=list(item["dialog"]),
                    items=[
                        Item(s["text"], list(s["recipes"]), s["intro"])
                        for s in item["statements"]
                    ],
                    supported=[
                        by_n[s["n"]]["supported"] is True for s in item["statements"]
                    ],
                    relevant=relevant,
                    comments=[
                        by_n[s["n"]].get("comment", "") for s in item["statements"]
                    ],
                )
            )
    return sorted(answers, key=lambda a: a.item)


@dataclass
class NarrowResult:
    label: str
    # Пункт задания → вердикты кандидата по утверждениям: (supported, relevant, comment).
    marks: dict[str, list[tuple[bool | None, bool | None, str]]] = field(
        default_factory=dict
    )
    # Набор канареек → (верно, всего, id ошибок).
    canaries: dict[str, tuple[int, int, list[str]]] = field(default_factory=dict)
    seconds: list[float] = field(default_factory=list)
    calls: int = 0
    failures: int = 0
    # Цена проверки одного ответа, ₽ (только ответы, без канареек).
    rub_per_answer: float | None = None
    error: str = ""


class _Progress:
    """Строка в stderr на каждый вызов: сколько готово и сколько примерно осталось."""

    def __init__(
        self, label: str, total: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.label, self.total, self.done = label, total, 0
        self._clock = clock
        self._started = clock()

    def step(self, what: str) -> None:
        self.done += 1
        elapsed = self._clock() - self._started
        left = elapsed / self.done * (self.total - self.done)
        print(
            f"[{self.label}] {self.done}/{self.total} {what} · прошло "
            f"{_clock_text(elapsed)} · осталось ≈ {_clock_text(left)}",
            file=sys.stderr,
            flush=True,
        )


def _check(
    result: NarrowResult,
    reviewer: Reviewer,
    question: str,
    items: Sequence[Item],
    context: Sequence[str] = (),
) -> list[Mark]:
    started = time.monotonic()
    marks = reviewer.check(question, items, context)
    result.seconds.append(time.monotonic() - started)
    result.calls += 1
    result.failures += all(m.supported is None for m in marks)
    return marks


def run_candidate(
    choice: JudgeChoice, answers: Sequence[Answer], limit: int | None = None
) -> NarrowResult:
    result = NarrowResult(choice.label)
    try:
        _run(result, choice, answers[:limit], limit)
    except Exception as error:  # кандидат не должен ронять остальных
        result.error = f"{type(error).__name__}: {str(error)[:200]}"
    save(result, probe=limit is not None)
    return result


def _run(
    result: NarrowResult,
    choice: JudgeChoice,
    answers: Sequence[Answer],
    limit: int | None,
) -> None:
    grounding = load_canaries()[:limit]
    relevance = load_relevance_canaries()[:limit]
    progress = _Progress(choice.label, len(answers) + len(grounding) + len(relevance))
    with _judge_llm(Settings(), choice) as llm:
        reviewer = Reviewer(llm, model=choice.model_name, prompt_version=PROMPT)
        for a in answers:
            marks = _check(result, reviewer, a.question, a.items, a.dialog)
            result.marks[a.item] = [(m.supported, m.relevant, m.comment) for m in marks]
            progress.step(a.case)
        if answers and choice.provider == "cloudru":
            cost = cost_rub(choice.model_name, llm.stats)
            result.rub_per_answer = None if cost is None else cost / len(answers)

        wrong = []
        for c in grounding:
            [mark] = _check(result, reviewer, "—", [Item(c.statement, [c.quote])])
            if mark.supported is not c.supported:
                wrong.append(c.id)
            progress.step(c.id)
        result.canaries["опора"] = (len(grounding) - len(wrong), len(grounding), wrong)

        wrong = []
        for r in relevance:
            recipe = "\n".join([r.name, *r.ingredients])
            item = Item(
                f"Можно приготовить «{r.name}».", [f"Рецепт целиком:\n{recipe}"]
            )
            [mark] = _check(result, reviewer, r.question, [item])
            # В пути ответа удаляет только явное «нет»: None — не удаляет.
            if (mark.relevant is False) is r.relevant:
                wrong.append(r.id)
            progress.step(r.id)
        result.canaries["релевантность"] = (
            len(relevance) - len(wrong),
            len(relevance),
            wrong,
        )


def save(result: NarrowResult, probe: bool = False) -> Path:
    """Пробный прогон (--limit) — в свой файл: полный он не затирает."""
    kind = "narrow-probe" if probe else "narrow"
    path = RAW / f"{date.today()}-{kind}-{_slug(result.label)}.json"
    path.write_text(
        json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return path


def load(path: Path) -> NarrowResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["marks"] = {
        k: [(m[0], m[1], m[2]) for m in v] for k, v in data["marks"].items()
    }
    data["canaries"] = {k: (v[0], v[1], v[2]) for k, v in data["canaries"].items()}
    return NarrowResult(**data)


@dataclass(frozen=True)
class Score:
    violations: int
    caught: int
    correct: int
    false_rejects: int
    irrelevant: int
    irrelevant_caught: int
    relevant: int
    relevant_dropped: int
    # Утверждения, которые пользователь бы не увидел зря: верное и по вопросу,
    # но кандидат отправил на доработку или удалил как «не по вопросу».
    lost: int
    judged: int


def score(result: NarrowResult, answers: Sequence[Answer]) -> Score:
    counts = Counter[str]()
    for a in answers:
        marks = result.marks.get(a.item)
        if marks is None:
            continue
        for (supported, relevant, _), ok, fits in zip(
            marks, a.supported, a.relevant, strict=True
        ):
            counts["judged"] += 1
            rejected = supported is not True
            if ok:
                counts["correct"] += 1
                counts["false_rejects"] += rejected
            else:
                counts["violations"] += 1
                counts["caught"] += rejected
            if fits is False:
                counts["irrelevant"] += 1
                counts["irrelevant_caught"] += relevant is False
            elif fits is True:
                counts["relevant"] += 1
                counts["relevant_dropped"] += relevant is False
            if ok and fits is not False:
                counts["lost"] += rejected or relevant is False
    return Score(**{f.name: counts[f.name] for f in dataclasses.fields(Score)})


def failed_rules(result: NarrowResult, answers: Sequence[Answer]) -> list[str]:
    """Какие пункты правила кандидат не прошёл; пусто — прошёл."""
    if result.error:
        return [f"сбой: {result.error}"]
    s = score(result, answers)
    reasons = []
    if s.judged < sum(len(a.items) for a in answers):
        reasons.append("проверены не все ответы")
    if s.caught < MIN_CAUGHT:
        reasons.append(f"1: поймано {s.caught} < {MIN_CAUGHT}")
    if s.false_rejects > math.floor(MAX_FALSE_SHARE * s.correct):
        reasons.append(f"2: зря {s.false_rejects} из {s.correct}")
    if result.calls and result.failures / result.calls > MAX_FAILURE_SHARE:
        reasons.append(f"3: сбои {result.failures}/{result.calls}")
    return reasons


def _rate(part: int, whole: int) -> str:
    return f"{part}/{whole}" if whole else "—"


def render(
    results: Sequence[NarrowResult], answers: Sequence[Answer], today: date
) -> str:
    lines = [
        f"# Узкий гейт за ответами GLM-5.1 low ({today})",
        "",
        f"Рубрика `{PROMPT}`. Эталон — вердикты субагента Sonnet на ответах "
        "`raw/2026-11-07-replay-glm-low.json` (отчёт evals/reports/2026-11-07-writers.md). "
        "Гейт видит ответ целиком, как в пути ответа; канарейки — по одной на вызов.",
        "",
        "## Правило выбора (записано до прогона)",
        "",
        RULE,
        "",
        "| Кандидат | нарушений поймано | зря на доработку | не по вопросу поймано | "
        "по вопросу удалено зря | пользователь потерял бы верных | канарейки опоры | "
        "канарейки релевантности | сбои разбора | секунд на ответ, медиана | ₽ на ответ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = score(r, answers)
        label = f"{r.label} (оборвался: {r.error[:60]})" if r.error else r.label
        cells = [
            f"{r.canaries[k][0]}/{r.canaries[k][1]}" if k in r.canaries else "—"
            for k in ("опора", "релевантность")
        ]
        per_answer = r.seconds[: len(r.marks)]
        median = statistics.median(per_answer) if per_answer else 0.0
        rub = "—" if r.rub_per_answer is None else f"{r.rub_per_answer:.3f}"
        lines.append(
            f"| {label} | {_rate(s.caught, s.violations)} | "
            f"{_rate(s.false_rejects, s.correct)} | "
            f"{_rate(s.irrelevant_caught, s.irrelevant)} | "
            f"{_rate(s.relevant_dropped, s.relevant)} | "
            f"{_rate(s.lost, s.correct)} | "
            + " | ".join(cells)
            + f" | {r.failures}/{r.calls} | {median:.1f} | {rub} |"
        )
    lines += ["", "## Итог по правилу", ""]
    for r in results:
        reasons = failed_rules(r, answers)
        lines.append(
            f"- {r.label}: "
            + ("прошёл" if not reasons else "не прошёл — " + "; ".join(reasons))
        )
    lines += _violations(results, answers)
    lines += _false_rejects(results, answers)
    lines += ["", "## Ошибки на канарейках", ""]
    for r in results:
        wrong = [f"{k}: {', '.join(v[2])}" for k, v in r.canaries.items() if v[2]]
        lines.append(f"- {r.label}: " + ("; ".join(wrong) or "нет"))
    return "\n".join(lines) + "\n"


def _violations(
    results: Sequence[NarrowResult], answers: Sequence[Answer]
) -> list[str]:
    """Каждое нарушение эталона: кто из кандидатов его поймал."""
    lines = ["", "## Нарушения по эталону и кто поймал", ""]
    for a in answers:
        for n, (ok, text, comment) in enumerate(
            zip(a.supported, a.items, a.comments, strict=True)
        ):
            if ok:
                continue
            who = [
                r.label
                for r in results
                if (m := r.marks.get(a.item)) and m[n][0] is not True
            ]
            lines.append(
                f"- `{a.case}` #{n + 1}: «{text.text[:120]}» — эталон: {comment} "
                f"— поймали: {', '.join(who) or 'никто'}"
            )
    return lines


def _false_rejects(
    results: Sequence[NarrowResult], answers: Sequence[Answer]
) -> list[str]:
    """Верные по эталону утверждения, которые кандидат отправил на доработку."""
    lines = ["", "## Зря на доработку (комментарий кандидата)", ""]
    for r in results:
        lines += ["", f"### {r.label}", ""]
        for a in answers:
            for n, (supported, _, comment) in enumerate(r.marks.get(a.item, [])):
                if a.supported[n] and supported is not True:
                    lines.append(
                        f"- `{a.case}` #{n + 1}: «{a.items[n].text[:120]}» — {comment}"
                    )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.gate_narrow", description=__doc__
    )
    parser.add_argument("task", type=Path, help="задание судьи-субагента с вердиктами")
    parser.add_argument(
        "--candidate",
        nargs="+",
        default=[],
        metavar="PROVIDER:MODEL[:EFFORT]",
        help="например cloudru:openai/gpt-oss-120b:medium; без кандидатов — "
        "только отчёт из сохранённых сегодня",
    )
    parser.add_argument("--limit", type=int, help="только первые N ответов и канареек")
    args = parser.parse_args(argv)
    candidates = [_choice(spec) for spec in args.candidate]
    answers = load_task(args.task)
    print(
        f"ответов: {len(answers)}, утверждений: {sum(len(a.items) for a in answers)}, "
        f"нарушений по эталону: {sum(not ok for a in answers for ok in a.supported)}, "
        f"кандидатов: {len(candidates)}",
        file=sys.stderr,
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        fresh = list(
            pool.map(lambda c: run_candidate(c, answers, args.limit), candidates)
        )
    today = date.today()
    kind = "narrow-probe" if args.limit else "narrow"
    # «narrow-*» захватил бы и пробные файлы «narrow-probe-*»: отбираются явно.
    saved = [
        p
        for p in sorted(RAW.glob(f"{today}-narrow*.json"))
        if p.name.startswith(f"{today}-{kind}-")
        and (args.limit or not p.name.startswith(f"{today}-narrow-probe-"))
    ]
    by_label = {r.label: r for r in (load(p) for p in saved)}
    by_label.update({r.label: r for r in fresh})
    results = sorted(by_label.values(), key=lambda r: r.label)
    shown = answers[: args.limit]
    name = f"{today}-gate-narrow{'-probe' if args.limit else ''}.md"
    path = REPORTS_DIR / name
    path.write_text(render(results, shown, today), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
