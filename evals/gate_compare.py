"""Кандидаты в гейт ответа против Sonnet — `python -m evals.gate_compare`.

Гейт — проверка перед отправкой (`review.py`, рубрика `review_v6`, и
`steps.py`, `steps_review_v1`). Сейчас это Sonnet 5.5 с effort high — почти
вся цена ответа (≈ $0.0056 из ≈ $0.0066). Кандидаты — дешевле: Haiku 5.5
(high, max) и модели, которые хостит cloud.ru. Все судят по тем же рубрикам
на одних и тех же данных:

- ручная разметка автора (`evals/reports/raw/*-manual.yaml`) — утверждения
  ответов, гейт видит ответ целиком, как в пути ответа;
- канарейки опоры на рецепт, релевантности (answers_question) и шагов —
  по одной на вызов.

Правило выбора записано до прогона (RULE ниже) и печатается в отчёте.
Кандидаты работают параллельно, у каждого свой HTTP-клиент.
"""

import argparse
import dataclasses
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from assistant.config import Settings
from assistant.generation.cloudru import cost_rub
from assistant.generation.review import Item, Reviewer
from assistant.generation.steps import StepsReviewer
from evals.answer_metrics import cost_usd
from evals.dataset import load_canaries, load_relevance_canaries, load_steps_canaries
from evals.judge import claims_of
from evals.judge_compare import MARKINGS, RAW, agreement, answers_to_judge
from evals.manual import dataset_questions
from evals.run import REPORTS_DIR, JudgeChoice, _judge_llm, load_answer_run
from evals.steps_eval import split_steps

BASELINE = JudgeChoice("anthropic", "claude-sonnet-5-5", "high")
CANDIDATES = (
    BASELINE,
    JudgeChoice("anthropic", "claude-haiku-5-5", "high"),
    JudgeChoice("anthropic", "claude-haiku-5-5", "max"),
    JudgeChoice("cloudru", "openai/gpt-oss-120b", "high"),
    JudgeChoice("cloudru", "ai-sage/GigaChat3.5-432B-A28B-Reasoning"),
    # Сильнейшие открытые модели по Artificial Analysis Intelligence Index v4.0
    # (Kimi 54, DeepSeek 52, GLM 51): кандидаты на переход на cloud.ru целиком.
    JudgeChoice("cloudru", "moonshotai/Kimi-K2.6"),
    JudgeChoice("cloudru", "deepseek-ai/DeepSeek-V4-Pro"),
    JudgeChoice("cloudru", "zai-org/GLM-5.1", "high"),
)
# Прошлые цифры Sonnet 5.5 (effort high) на этих же данных: разметка автора —
# evals/reports/2026-11-06-judges-compare.md (рубрика судьи judge_v8),
# канарейки релевантности гейта и шагов — review_v6 и steps_eval (без ошибок).
# Правило — от них: Anthropic закрыт (403, 2026-11-06), заново Sonnet не прогнать.
REFERENCE_SAME, REFERENCE_CAUGHT = 55, 31
RULE = """\
Пороги — от прошлых цифр Sonnet 5.5 на этих данных (Anthropic закрыт, заново
его не прогнать): разметка автора 55/69, «нет» автора пойман 31/43, канарейки
релевантности и шагов — без ошибок. Кандидат проходит, если выполнено всё сразу:
1. на разметке автора не меньше 52/69 (Sonnet минус 3);
2. «нет» автора пойман не реже 29/43 (Sonnet минус 2: пропуск ошибки хуже придирки);
3. на каждом наборе канареек — не больше одной ошибки и ни одного одобренного
   неверного шага;
4. сбоев разбора — не больше 2 % вызовов.
Задержка не отсекает (автор: «пусть даже ценой ожидания вывода») — она в отчёте.
Из прошедших — лучший по разметке автора; при равенстве — дешевле."""

Key = tuple[str, str, int]


@dataclass
class GateResult:
    label: str
    provider: str
    marks: dict[tuple[str, str, int, int], bool | None] = field(default_factory=dict)
    # Набор канареек → (верно, всего, id ошибок).
    canaries: dict[str, tuple[int, int, list[str]]] = field(default_factory=dict)
    # Шаги с approved: false, которые кандидат одобрил.
    approved_wrong_steps: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)
    calls: int = 0
    failures: int = 0
    # Цена на один ответ (вызов проверки по разметке), $ или ₽.
    cost_per_answer: str = ""
    answers: int = 0
    # Сбой кандидата (например, провайдер закрыт): остальные идут дальше.
    error: str = ""


def _timed[T](result: GateResult, call: Callable[..., T], *args: Any) -> T:
    """Вызов проверки с замером времени: секунды и число вызовов — в отчёт."""
    started = time.monotonic()
    value = call(*args)
    result.seconds.append(time.monotonic() - started)
    result.calls += 1
    return value


def run_gate(
    choice: JudgeChoice,
    keys: Sequence[Key],
    records: Mapping[Key, dict[str, Any]],
    questions: Mapping[str, str],
    limit: int | None = None,
) -> GateResult:
    result = GateResult(choice.label, choice.provider)
    try:
        _run(result, choice, keys, records, questions, limit)
    except Exception as error:  # кандидат не должен ронять остальных
        result.error = f"{type(error).__name__}: {str(error)[:200]}"
    save(result)
    return result


def _run(
    result: GateResult,
    choice: JudgeChoice,
    keys: Sequence[Key],
    records: Mapping[Key, dict[str, Any]],
    questions: Mapping[str, str],
    limit: int | None,
) -> None:
    with _judge_llm(Settings(), choice) as llm:
        reviewer = Reviewer(llm, model=choice.model_name)
        for key in keys[:limit]:
            run = load_answer_run(records[key])
            items = [Item(c.text, c.fragments, c.intro) for c in claims_of(run)]
            marks = _timed(result, reviewer.check, questions[key[1]], items)
            result.failures += all(m.supported is None for m in marks)
            for number, mark in enumerate(marks):
                result.marks[(*key, number)] = mark.supported
        result.answers = len(keys[:limit])
        cost = _cost(choice, llm.stats)
        result.cost_per_answer = _per(cost, result.answers, choice.provider)

        grounding = load_canaries()[:limit]
        wrong = []
        for c in grounding:
            [mark] = _timed(result, reviewer.check, "—", [Item(c.statement, [c.quote])])
            result.failures += mark.supported is None
            if mark.supported is not c.supported:
                wrong.append(c.id)
        result.canaries["опора"] = (len(grounding) - len(wrong), len(grounding), wrong)

        relevance = load_relevance_canaries()[:limit]
        wrong = []
        for r in relevance:
            recipe = "\n".join([r.name, *r.ingredients])
            item = Item(
                f"Можно приготовить «{r.name}».", [f"Рецепт целиком:\n{recipe}"]
            )
            [mark] = _timed(result, reviewer.check, r.question, [item])
            result.failures += mark.relevant is None
            if mark.relevant is not r.relevant:
                wrong.append(r.id)
        result.canaries["релевантность"] = (
            len(relevance) - len(wrong),
            len(relevance),
            wrong,
        )

        steps_reviewer = StepsReviewer(llm, model=choice.model_name)
        steps = load_steps_canaries()[:limit]
        wrong = []
        for s in steps:
            check = _timed(result, steps_reviewer.check, s.recipe, split_steps(s.steps))
            result.failures += check.approved is None
            if check.approved is not s.approved:
                wrong.append(s.id)
                if check.approved is True:
                    result.approved_wrong_steps.append(s.id)
        result.canaries["шаги"] = (len(steps) - len(wrong), len(steps), wrong)


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in label.lower()).strip("-")


def save(result: GateResult) -> Path:
    """Результат кандидата — сразу в сырые данные: долгий прогон не пропадёт,
    если упадёт другой кандидат, а отчёт можно собрать из разных прогонов."""
    data = dataclasses.asdict(result)
    data["marks"] = {"|".join(map(str, k)): v for k, v in result.marks.items()}
    path = RAW / f"{date.today()}-gate-{_slug(result.label)}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load(path: Path) -> GateResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    marks = {}
    for key, value in data.pop("marks").items():
        file, case, run, number = key.split("|")
        marks[(file, case, int(run), int(number))] = value
    data["canaries"] = {k: (v[0], v[1], v[2]) for k, v in data["canaries"].items()}
    return GateResult(marks=marks, **data)


def _cost(choice: JudgeChoice, stats: Any) -> float:
    if choice.provider == "cloudru":
        return cost_rub(choice.model_name, stats) or 0.0
    return cost_usd(choice.model_name, stats) or 0.0


def _per(cost: float, answers: int, provider: str) -> str:
    if not answers:
        return "—"
    per = cost / answers
    return f"≈{per:.3f} ₽" if provider == "cloudru" else f"≈${per:.4f}"


def verdicts(
    results: Sequence[GateResult], markings: Mapping[str, Sequence[dict[str, Any]]]
) -> dict[str, list[str]]:
    """Кандидат → какие пункты правила он не прошёл (пусто — прошёл)."""

    def score(r: GateResult) -> tuple[int, int]:
        same = caught = 0
        for name, marks in markings.items():
            s, _, c, _, _ = agreement(marks, MARKINGS[name], r.marks)
            same, caught = same + s, caught + c
        return same, caught

    failed: dict[str, list[str]] = {}
    for r in results:
        if r.error:
            failed[r.label] = [f"сбой: {r.error}"]
            continue
        same, caught = score(r)
        reasons = []
        if same < REFERENCE_SAME - 3:
            reasons.append("1: разметка")
        if caught < REFERENCE_CAUGHT - 2:
            reasons.append("2: «нет» автора")
        for name, (right, total, _) in r.canaries.items():
            if right < total - 1:
                reasons.append(f"3: канарейки {name}")
        if r.approved_wrong_steps:
            reasons.append("3: одобрен неверный шаг")
        if r.calls and r.failures / r.calls > 0.02:
            reasons.append("4: сбои разбора")
        failed[r.label] = reasons
    return failed


def render(
    results: Sequence[GateResult],
    markings: Mapping[str, Sequence[dict[str, Any]]],
    today: date,
) -> str:
    lines = [
        f"# Кандидаты в гейт ответа ({today})",
        "",
        "Рубрики `review_v6` и `steps_review_v1`; разметка автора — гейт видит ответ "
        "целиком, канарейки — по одной на вызов.",
        "",
        "## Правило выбора (записано до прогона)",
        "",
        RULE,
        "",
        "| Кандидат | разметка автора | «нет» автора пойман | опора | релевантность | шаги | "
        "сбои разбора | секунд на вызов, медиана | цена на ответ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.error and not r.calls:
            lines.append(f"| {r.label} | сбой: {r.error} |" + " |" * 7)
            continue
        # Частичный результат — в таблицу, с пометкой: решать по нему нельзя.
        label = f"{r.label} (оборвался: {r.error[:60]})" if r.error else r.label
        same = total = caught = no_total = 0
        for name, marks in markings.items():
            s, t, c, n, _ = agreement(marks, MARKINGS[name], r.marks)
            same, total, caught, no_total = (
                same + s,
                total + t,
                caught + c,
                no_total + n,
            )
        # Кандидат со сбоем мог не дойти до части канареек.
        cells = [
            f"{r.canaries[k][0]}/{r.canaries[k][1]}" if k in r.canaries else "—"
            for k in ("опора", "релевантность", "шаги")
        ]
        median = statistics.median(r.seconds) if r.seconds else 0.0
        lines.append(
            f"| {label} | {same}/{total} | {caught}/{no_total} | "
            + " | ".join(cells)
            + f" | {r.failures}/{r.calls} | {median:.1f} | {r.cost_per_answer} |"
        )
    lines += ["", "## Итог по правилу", ""]
    for label, reasons in verdicts(results, markings).items():
        lines.append(
            f"- {label}: "
            + ("прошёл" if not reasons else "не прошёл — " + ", ".join(reasons))
        )
    lines += ["", "## Ошибки на канарейках", ""]
    for r in results:
        wrong = [f"{k}: {', '.join(v[2])}" for k, v in r.canaries.items() if v[2]]
        steps = (
            f"; одобрены неверные шаги: {', '.join(r.approved_wrong_steps)}"
            if r.approved_wrong_steps
            else ""
        )
        lines.append(f"- {r.label}: " + ("; ".join(wrong) or "нет") + steps)
    return "\n".join(lines) + "\n"


def _choice(spec: str) -> JudgeChoice:
    """«cloudru:moonshotai/Kimi-K2.6:low» → кандидат; effort можно не задавать."""
    provider, _, rest = spec.partition(":")
    model, _, effort = rest.rpartition(":") if rest.count(":") else (rest, "", "")
    if provider not in ("anthropic", "cloudru", "gigachat") or not model:
        raise SystemExit(f"кандидат {spec!r}: нужно провайдер:модель[:effort]")
    return JudgeChoice(provider, model, effort)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.gate_compare", description=__doc__
    )
    parser.add_argument(
        "--limit", type=int, help="только первые N ответов и канареек — проба"
    )
    parser.add_argument(
        "--only", nargs="+", help="подписи кандидатов (часть имени модели)"
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        default=[],
        metavar="PROVIDER:MODEL[:EFFORT]",
        help="кандидаты не из списка, например cloudru:moonshotai/Kimi-K2.6:low",
    )
    args = parser.parse_args(argv)
    extra = [_choice(spec) for spec in args.candidate]
    candidates = extra or [
        c for c in CANDIDATES if not args.only or any(o in c.label for o in args.only)
    ]
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
    print(f"ответов: {len(keys)}, кандидатов: {len(candidates)}", flush=True)
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        fresh = list(
            pool.map(
                lambda c: run_gate(c, keys, records, questions, args.limit), candidates
            )
        )
    today = date.today()
    # Отчёт — по всем кандидатам, сохранённым сегодня (в том числе другими
    # прогонами); свежий результат важнее старого.
    by_label = {
        r.label: r for r in (load(p) for p in sorted(RAW.glob(f"{today}-gate-*.json")))
    }
    by_label.update({r.label: r for r in fresh})
    order = [c.label for c in CANDIDATES]
    results = sorted(
        by_label.values(),
        key=lambda r: order.index(r.label) if r.label in order else len(order),
    )
    name = f"{today}-gate-compare{'-probe' if args.limit else ''}.md"
    path = REPORTS_DIR / name
    path.write_text(render(results, markings, today), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
