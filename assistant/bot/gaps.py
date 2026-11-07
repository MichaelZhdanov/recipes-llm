"""Пробелы базы (S1): с чем ассистент не справился — по SQLite бота.

`python -m assistant.bot.gaps [--days 30]` — локально `make gaps`, на сервере
`docker compose exec bot /app/.venv/bin/python -m assistant.bot.gaps`.

Три раздела:
1. «Не нашёл» — по причине и по похожим вопросам: одинаковые после
   нормализации (регистр, «ё», знаки) складываются, видно, что спрашивают
   чаще;
2. 👎 — с причиной («непонятно», «не то», «не получилось») и вопросом;
3. передачи человеку — по поводу, со сводкой для оператора.

Отчёт — Markdown в stdout, не в git: это вопросы живых пользователей. ПДн в
базе уже нет — вопрос маскируется при записи (`Store`). База открывается
только на чтение (`mode=ro`): отчёт не может испортить данные бота.
"""

import argparse
import re
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from assistant.bot.escalation import REASON_LABELS

DATABASE = Path("data/bot/bot.sqlite3")
DAY = 24 * 60 * 60
# Причины «не нашёл» (Answer.reason) — словами для отчёта.
NOT_FOUND = {
    "model_not_found": "среди найденного не подошло ни одно",
    "no_sources": "поиск ничего не нашёл",
    "filtered_out": "утверждения не прошли проверку цитат",
    "length": "ответ оборвался на лимите токенов",
    "format_error": "модель ответила не по формату",
    "blacklist": "модель не стала отвечать",
}
VOTE_REASONS = {
    "unclear": "непонятно",
    "wrong": "не то искал",
    "failed": "не получилось",
}
TOP = 15
# До этапа 6 вопрос в answers не сохранялся (столбец добавлен миграцией).
NO_QUESTION = "— вопрос не сохранялся (ответ до этапа 6)"


def normalize(question: str) -> str:
    """Ключ для группировки похожих вопросов: «Борщ!» и «борщ» — одно."""
    text = question.lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


@dataclass
class Gaps:
    days: int
    answers: int = 0
    not_found: Counter[str] = field(default_factory=Counter)
    # ключ вопроса → (сколько раз, как спросили в последний раз, причины)
    questions: dict[str, tuple[int, str, Counter[str]]] = field(default_factory=dict)
    downvotes: list[tuple[str, str]] = field(default_factory=list)
    tickets: list[tuple[str, str, str]] = field(default_factory=list)


def collect(db: sqlite3.Connection, days: int, now: float) -> Gaps:
    since = now - days * DAY
    gaps = Gaps(days)
    gaps.answers = db.execute(
        "SELECT count(*) FROM answers WHERE created_at >= ?", (since,)
    ).fetchone()[0]
    rows = db.execute(
        "SELECT reason, coalesce(question, '') FROM answers "
        "WHERE status = 'not_found' AND created_at >= ? ORDER BY created_at",
        (since,),
    ).fetchall()
    for reason, question in rows:
        reason = reason or "model_not_found"
        gaps.not_found[reason] += 1
        key = normalize(question)
        count, _, reasons = gaps.questions.get(key, (0, question, Counter[str]()))
        reasons[reason] += 1
        gaps.questions[key] = (count + 1, question, reasons)
    gaps.downvotes = [
        (VOTE_REASONS.get(reason or "", "причину не указали"), question or "")
        for reason, question in db.execute(
            "SELECT vote_reason, question FROM answers "
            "WHERE vote = 'down' AND created_at >= ? ORDER BY created_at DESC",
            (since,),
        )
    ]
    gaps.tickets = [
        (REASON_LABELS.get(reason, reason), summary, status)
        for reason, summary, status in db.execute(
            "SELECT reason, summary, status FROM tickets "
            "WHERE opened_at >= ? ORDER BY opened_at DESC",
            (since,),
        )
    ]
    return gaps


def _cell(text: str, limit: int = 160) -> str:
    text = " ".join(text.split()).replace("|", "/") or NO_QUESTION
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(gaps: Gaps, now: float) -> str:
    total_nf = sum(gaps.not_found.values())
    share = f"{total_nf / gaps.answers:.0%}" if gaps.answers else "—"
    date = datetime.fromtimestamp(now, UTC).date()
    lines = [
        f"# Пробелы базы за {gaps.days} дн. (по {date})",
        "",
        f"Ответов: {gaps.answers}; «не нашёл» — {total_nf} ({share}); 👎 — "
        f"{len(gaps.downvotes)}; передано человеку — {len(gaps.tickets)}.",
        "",
        "## «Не нашёл»: почему",
        "",
    ]
    lines += [
        f"- {NOT_FOUND.get(reason, reason)} — {count}"
        for reason, count in gaps.not_found.most_common()
    ] or ["- не было"]
    lines += ["", "## «Не нашёл»: что спрашивают чаще", ""]
    ranked = sorted(gaps.questions.values(), key=lambda q: (-q[0], q[1]))[:TOP]
    if ranked:
        lines += ["| раз | вопрос | причины |", "|---|---|---|"]
        lines += [
            f"| {count} | {_cell(question)} | "
            + ", ".join(NOT_FOUND.get(r, r) for r in reasons)
            + " |"
            for count, question, reasons in ranked
        ]
    else:
        lines.append("Не было.")
    lines += ["", "## 👎", ""]
    if gaps.downvotes:
        lines += ["| что не так | вопрос |", "|---|---|"]
        lines += [f"| {r} | {_cell(q)} |" for r, q in gaps.downvotes[:TOP]]
    else:
        lines.append("Не было.")
    lines += ["", "## Передано человеку", ""]
    if gaps.tickets:
        by_reason = Counter(reason for reason, _, _ in gaps.tickets)
        lines.append(", ".join(f"{r} — {n}" for r, n in by_reason.most_common()))
        lines += ["", "| повод | сводка | обращение |", "|---|---|---|"]
        lines += [
            f"| {r} | {_cell(s) or '—'} | {'открыто' if st == 'open' else 'закрыто'} |"
            for r, s, st in gaps.tickets[:TOP]
        ]
    else:
        lines.append("Не было.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m assistant.bot.gaps", description=__doc__
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--db", type=Path, default=DATABASE)
    args = parser.parse_args(argv)
    if not args.db.exists():
        print(f"базы бота нет: {args.db}", file=sys.stderr)
        return 1
    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        now = time.time()
        print(render(collect(db, args.days, now), now), end="")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
