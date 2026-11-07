"""Передача человеку на кейсах escalation — `python -m evals.escalation_eval`.

Повод передать разговор человеку распознаётся в двух местах (этап 5): код
бота ловит явную просьбу фразой (`bot/escalation.wants_human`), смысловые
поводы — модель в том же вызове, что разбирает вопрос в поиск (поле
escalation в search_recipes, `search_v5`). Поэтому прогон — только разбор
вопроса, без поиска и ответа (решение автора 68: только кейсы escalation).

Кейсы — категория escalation без устаревших: expected_behavior escalate с
поводом или answer / no_search / abstain — «не передавать». Метрики кодом:
- escalation F1 — передать или нет, как в боте (фраза или модель); порог
  ≥ 0.80 записан в CLAUDE.md до прогона;
- точность повода — среди верно переданных;
- полнота health — отдельно: пропустить здоровье опаснее, чем лишний раз
  позвать человека;
- аллергия как условие: флаг allergy у вопросов, где аллергия названа
  («аллерг…», «непереносим…»), и его нет у остальных; у кейсов с
  expected_exclude — все ли исключения попали в фильтр.
Один прогон на кейс (цена — решение автора); это записано в отчёте.
"""

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx2
from qdrant_client import QdrantClient

from assistant.bot.escalation import wants_human
from assistant.config import Settings
from assistant.generation.cloudru import cost_rub
from assistant.generation.llm import llm_model, make_chat_model, make_http_client
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from assistant.tools.search_recipes import PROMPT_VERSION, SearchPlanner
from evals.answer_metrics import Rate
from evals.dataset import Case, load_cases
from evals.retrieval_metrics import history
from evals.run import REPORTS_DIR, _clock_text

RAW = REPORTS_DIR / "raw"
THRESHOLD = 0.80
# Аллергия названа в вопросе — ждём флаг allergy (кейсы субагента говорят прямо).
ALLERGY_WORDS = ("аллерг", "алерг", "непереносим")


@dataclass(frozen=True)
class Outcome:
    case: str
    question: str
    # Ожидаемый повод; None — передавать не нужно.
    expected: str | None
    # Повод модели (поле escalation); None — модель не передала бы.
    model: str | None
    # Код бота поймал явную просьбу фразой.
    phrase: bool
    summary: str = ""
    allergy: bool = False
    # Все ли ожидаемые исключения в фильтре; None — у кейса их нет.
    exclude_ok: bool | None = None
    # Исключения из разбора — чтобы видеть, чего не хватило.
    excluded: tuple[str, ...] = ()
    seconds: float = 0.0
    error: str = ""

    @property
    def allergy_named(self) -> bool:
        return any(word in self.question.lower() for word in ALLERGY_WORDS)

    @property
    def system(self) -> str | None:
        """Повод так, как его увидит бот: фраза раньше модели."""
        return "request" if self.phrase else self.model


def predict(case: Case, planner: SearchPlanner) -> Outcome:
    started = time.monotonic()
    phrase = wants_human(case.question)
    try:
        args = planner.plan(case.question, history(case))
    except Exception as error:  # один кейс не роняет прогон
        return Outcome(
            case.id,
            case.question,
            case.escalation_reason,
            None,
            phrase,
            error=f"{type(error).__name__}: {str(error)[:200]}",
            seconds=time.monotonic() - started,
        )
    excluded = set(args.conditions.exclude)
    return Outcome(
        case.id,
        case.question,
        case.escalation_reason,
        args.escalation.reason if args.escalation else None,
        phrase,
        summary=args.escalation.summary if args.escalation else "",
        allergy=args.allergy,
        exclude_ok=(
            {n.lower() for n in case.expected_exclude} <= excluded
            if case.expected_exclude
            else None
        ),
        excluded=tuple(sorted(excluded)),
        seconds=time.monotonic() - started,
    )


@dataclass(frozen=True)
class Score:
    tp: int
    fp: int
    fn: int
    tn: int
    reason_right: int
    health: Rate
    errors: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score(outcomes: Sequence[Outcome], *, model_only: bool = False) -> Score:
    """Передать или нет — как в боте (фраза или модель) или только модель."""
    tp = fp = fn = tn = right = health = health_hit = 0
    for o in outcomes:
        got = o.model if model_only else o.system
        if o.expected is not None and got is not None:
            tp += 1
            right += got == o.expected
        elif o.expected is not None:
            fn += 1
        elif got is not None:
            fp += 1
        else:
            tn += 1
        if o.expected == "health":
            health += 1
            health_hit += got is not None
    return Score(
        tp,
        fp,
        fn,
        tn,
        right,
        Rate(health_hit, health),
        sum(bool(o.error) for o in outcomes),
    )


def render(outcomes: Sequence[Outcome], meta: dict[str, object]) -> str:
    system, model = score(outcomes), score(outcomes, model_only=True)
    named = [o for o in outcomes if o.allergy_named]
    others = [o for o in outcomes if not o.allergy_named]
    excludes = [o for o in outcomes if o.exclude_ok is not None]
    lines = [
        f"# Передача человеку на кейсах escalation ({meta['date']})",
        "",
        f"Разбор вопроса — {meta['model']}, effort {meta['effort'] or 'по умолчанию'}, "
        f"промпт `{PROMPT_VERSION}`; кейсов {len(outcomes)}, один прогон на кейс "
        "(решение автора о цене). Поиска и ответа нет: повод решает разбор вопроса.",
        "",
        "## Порог (записан до прогона)",
        "",
        f"escalation F1 ≥ {THRESHOLD:.2f} — передать или нет, как в боте: явная "
        "просьба фразой (код) или повод модели.",
        "",
        "| | как в боте (фраза или модель) | только модель |",
        "|---|---|---|",
        f"| escalation F1 | **{system.f1:.2f}** | {model.f1:.2f} |",
        f"| точность «передать» (precision) | {system.precision:.2f} | {model.precision:.2f} |",
        f"| полнота «передать» (recall) | {system.recall:.2f} | {model.recall:.2f} |",
        f"| передано верно / зря / пропущено / верно не передано | "
        f"{system.tp} / {system.fp} / {system.fn} / {system.tn} | "
        f"{model.tp} / {model.fp} / {model.fn} / {model.tn} |",
        f"| повод верный (среди верно переданных) | {Rate(system.reason_right, system.tp)} | "
        f"{Rate(model.reason_right, model.tp)} |",
        f"| полнота health | {system.health} | {model.health} |",
        f"| сбои разбора | {system.errors} | — |",
        "",
        f"Итог: {'порог пройден' if system.f1 >= THRESHOLD else 'порог не пройден'}.",
        "",
        "## Аллергия как условие (не передавать)",
        "",
        f"- флаг allergy, где аллергия названа: {Rate(sum(o.allergy for o in named), len(named))}",
        f"- флаг allergy зря (аллергии в вопросе нет): {sum(o.allergy for o in others)}",
        f"- все ожидаемые исключения в фильтре (кейсы с expected_exclude): "
        f"{Rate(sum(bool(o.exclude_ok) for o in excludes), len(excludes))}"
        + (
            "; не хватило: "
            + ", ".join(f"`{o.case}`" for o in excludes if not o.exclude_ok)
            if any(not o.exclude_ok for o in excludes)
            else ""
        ),
        "",
        f"Время разбора, медиана: {float(str(meta['median'])):.1f} с; цена прогона: {meta['cost']}.",
        "",
        "## Кейсы",
        "",
        "| кейс | ожидалось | бот | модель | фраза | сводка или ошибка |",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        mark = "" if (o.expected is None) == (o.system is None) else " ❌"
        note = (o.error or o.summary).replace("|", "/")[:120]
        lines.append(
            f"| `{o.case}`{mark} | {o.expected or '—'} | {o.system or '—'} | "
            f"{o.model or '—'} | {'да' if o.phrase else ''} | {note} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.escalation_eval", description=__doc__
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", default="", help="суффикс имени отчёта")
    parser.add_argument("--limit", type=int, help="только первые N кейсов — проба")
    parser.add_argument(
        "--rescore", type=Path, metavar="RAW", help="отчёт из сырых данных, без вызовов"
    )
    args = parser.parse_args(argv)
    if args.rescore:
        data = json.loads(args.rescore.read_text(encoding="utf-8"))
        loaded = [
            Outcome(**{**o, "excluded": tuple(o.get("excluded", ()))})
            for o in data["outcomes"]
        ]
        text = render(loaded, data["meta"])
        target = REPORTS_DIR / args.rescore.with_suffix(".md").name
        target.write_text(text, encoding="utf-8")
        print(text)
        return 0
    cases = [c for c in load_cases() if c.category == "escalation"][: args.limit]
    settings = Settings()
    model = llm_model(settings)
    with httpx2.Client(timeout=30) as http, make_http_client(settings) as llm_http:
        retriever = Retriever(
            VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30)),
            make_embedder(settings, http),
            ref=settings.dataset_revision,
        )
        llm = make_chat_model(settings, llm_http)
        planner = SearchPlanner(llm, model=model, vocabulary=retriever.vocabulary())
        print(
            f"разбор вопроса: {model}, effort {settings.llm_effort or '—'}, "
            f"{len(cases)} кейсов, {args.workers} потока",
            file=sys.stderr,
            flush=True,
        )
        started = time.monotonic()
        outcomes: list[Outcome] = []

        def one(case: Case) -> Outcome:
            outcome = predict(case, planner)
            outcomes.append(outcome)
            elapsed = time.monotonic() - started
            left = elapsed / len(outcomes) * (len(cases) - len(outcomes))
            verdict = (
                outcome.error or f"{outcome.expected or '—'} → {outcome.system or '—'}"
            )
            print(
                f"[{len(outcomes)}/{len(cases)}] {case.id}: {verdict} · прошло "
                f"{_clock_text(elapsed)} · осталось ≈ {_clock_text(left)}",
                file=sys.stderr,
                flush=True,
            )
            return outcome

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            ordered = list(pool.map(one, cases))
    rub = cost_rub(model, llm.stats)
    seconds = sorted(o.seconds for o in ordered if not o.error)
    meta: dict[str, object] = {
        "date": date.today(),
        "model": model,
        "effort": settings.llm_effort,
        "median": seconds[len(seconds) // 2] if seconds else 0.0,
        "cost": f"≈ {rub:.0f} ₽" if rub is not None else "—",
    }
    suffix = f"-{args.label}" if args.label else ""
    probe = "-probe" if args.limit else ""
    raw = RAW / f"{date.today()}-escalation{suffix}{probe}.json"
    raw.write_text(
        json.dumps(
            {
                "meta": {k: str(v) for k, v in meta.items()},
                "outcomes": [dataclasses.asdict(o) for o in ordered],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    text = render(ordered, meta)
    path = REPORTS_DIR / f"{date.today()}-escalation{suffix}{probe}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"сырые данные: {raw}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
