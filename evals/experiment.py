"""Эксперимент с контрольной группой (S4): `python -m evals.experiment analyze|simulate`.

Метрика — доля пользователей, решивших задачу без человека: хотя бы один 👍
и ни одного обращения к человеку за период. Единица анализа — пользователь,
как и единица распределения (`assistant/bot/experiment.py`): ответы одного
человека зависимы, и считать их по отдельности — завышать уверенность.

- `analyze --db data/bot/bot.sqlite3 [--days 30]` — по живой базе бота
  (столбец answers.arm); отчёт — в stdout, не в git.
- `simulate --users N --control P --test P` — синтетический трафик без модели:
  пользователи распределяются той же функцией, что в боте, исходы — с
  заданными долями, записываются в SQLite тем же `Store` и считаются тем же
  анализом. Это демонстрация методики, а не результат (решение 86).

Разница долей — тест минус контроль, 95 % ДИ по Ньюкомбу (метод 10:
разность интервалов Уилсона; Newcombe, Statistics in Medicine, 1998) — он
не выходит за [−1, 1] и честен на малых выборках, в отличие от Вальда.
Сколько нужно пользователей — нормальное приближение для двух долей при
α = 0,05 (двусторонний) и мощности 80 %.
"""

import argparse
import math
import random
import sqlite3
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

from pydantic import SecretStr

from assistant.bot import experiment
from assistant.bot.journal import Journal
from assistant.bot.store import Store
from evals.answer_metrics import Rate

DAY = 24 * 60 * 60
Z = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True)
class Arm:
    name: str
    users: int
    solved: int

    @property
    def rate(self) -> Rate:
        return Rate(self.solved, self.users)


def newcombe(test: Rate, control: Rate, z: float = Z) -> tuple[float, float, float]:
    """Разница долей (тест − контроль) и 95 % ДИ по Ньюкомбу (метод 10)."""
    p1, p2 = test.value, control.value
    l1, u1 = test.wilson(z)
    l2, u2 = control.wilson(z)
    diff = p1 - p2
    low = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    high = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return diff, low, high


def users_needed(base: float, lift: float, power: float = 0.8) -> int:
    """Пользователей в каждой группе, чтобы заметить разницу lift (доли) при α = 0,05."""
    p1, p2 = base, base + lift
    mean = (p1 + p2) / 2
    z_beta = NormalDist().inv_cdf(power)
    top = Z * math.sqrt(2 * mean * (1 - mean)) + z_beta * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    return math.ceil(top**2 / lift**2)


def arms(db: sqlite3.Connection, since: float) -> dict[str, Arm]:
    """Пользователи групп за период: решил — хотя бы один 👍 и ни одного обращения."""
    rows = db.execute(
        """
        SELECT a.arm, a.chat_id,
               max(a.vote = 'up') AS liked,
               EXISTS (SELECT 1 FROM tickets t WHERE t.chat_id = a.chat_id
                       AND t.opened_at >= ?) AS handed
        FROM answers a
        WHERE a.arm IS NOT NULL AND a.created_at >= ?
        GROUP BY a.arm, a.chat_id
        """,
        (since, since),
    ).fetchall()
    result: dict[str, Arm] = {}
    for name in (experiment.CONTROL, experiment.TEST):
        users = [r for r in rows if r[0] == name]
        solved = sum(1 for r in users if r[2] and not r[3])
        result[name] = Arm(name, len(users), solved)
    return result


def render(groups: dict[str, Arm], title: str, note: str = "") -> str:
    control, test = groups[experiment.CONTROL], groups[experiment.TEST]
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += [
        "| группа | пользователей | решили без человека, с 👍 |",
        "|---|---|---|",
        f"| контроль — «всегда ищем» | {control.users} | {control.rate} |",
        f"| тест — «модель решает» | {test.users} | {test.rate} |",
        "",
    ]
    if control.users and test.users:
        diff, low, high = newcombe(test.rate, control.rate)
        verdict = (
            "интервал целиком выше нуля — тест лучше"
            if low > 0
            else "интервал целиком ниже нуля — тест хуже"
            if high < 0
            else "интервал накрывает ноль — разницы не видно, данных мало"
        )
        lines += [
            f"**Разница (тест − контроль): {diff * 100:+.1f} п. п., 95 % ДИ "
            f"{low * 100:+.1f} … {high * 100:+.1f} п. п.** — {verdict}.",
            "",
        ]
        base = control.rate.value
        if 0 < base < 1:
            lines += [
                "Сколько пользователей нужно в каждой группе, чтобы заметить разницу "
                f"(α = 0,05, мощность 80 %, доля в контроле {base:.0%}):",
                "",
                "| разница | пользователей в группе |",
                "|---|---|",
            ]
            for lift in (0.05, 0.10, 0.15):
                if base + lift < 1:
                    lines.append(
                        f"| +{lift * 100:.0f} п. п. | {users_needed(base, lift)} |"
                    )
    else:
        lines.append(
            "В одной из групп нет пользователей: эксперимент выключен или пуст."
        )
    return "\n".join(lines) + "\n"


def simulate(
    users: int, control: float, test: float, seed: int, path: Path
) -> dict[str, Arm]:
    """Синтетический трафик в SQLite бота: та же функция группы, тот же Store."""
    rng = random.Random(seed)
    journal = Journal(SecretStr(f"simulation-{seed}"))
    store = Store(path, clock=time.time)
    for chat_id in range(1, users + 1):
        arm = experiment.arm(journal.user(chat_id))
        solved = rng.random() < (test if arm == experiment.TEST else control)
        # 1–3 ответа у пользователя; решил — 👍 на одном из них.
        ids = [
            store.save_answer(
                chat_id,
                status="answer",
                reason=None,
                model="sim",
                recipe_ids=[],
                seconds=1.0,
                arm=arm,
            )
            for _ in range(rng.randint(1, 3))
        ]
        if solved:
            store.vote(rng.choice(ids), chat_id, "up")
        elif rng.random() < 0.3:
            # Не решил: часть уходит к человеку, часть ставит 👎 или молчит.
            store.open_ticket(chat_id, "request", "синтетика")
        elif rng.random() < 0.5:
            store.vote(ids[0], chat_id, "down")
    store.close()
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return arms(db, 0.0)
    finally:
        db.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.experiment", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    an = commands.add_parser("analyze", help="по живой базе бота")
    an.add_argument("--db", type=Path, default=Path("data/bot/bot.sqlite3"))
    an.add_argument("--days", type=int, default=30)
    sim = commands.add_parser("simulate", help="синтетический трафик без модели")
    sim.add_argument("--users", type=int, default=200)
    sim.add_argument(
        "--control", type=float, default=0.40, help="доля решивших в контроле"
    )
    sim.add_argument("--test", type=float, default=0.45, help="доля решивших в тесте")
    sim.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        if not args.db.exists():
            print(f"базы бота нет: {args.db}", file=sys.stderr)
            return 1
        db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        try:
            groups = arms(db, time.time() - args.days * DAY)
        finally:
            db.close()
        print(render(groups, f"Эксперимент S4 за {args.days} дн."), end="")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        groups = simulate(
            args.users, args.control, args.test, args.seed, Path(tmp) / "sim.sqlite3"
        )
    note = (
        f"Синтетический трафик без модели: {args.users} пользователей, истинная доля "
        f"решивших — контроль {args.control:.0%}, тест {args.test:.0%}, сид {args.seed}. "
        "Демонстрация методики, а не результат."
    )
    print(render(groups, "Эксперимент S4: симуляция", note), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
