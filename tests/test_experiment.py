"""Анализ эксперимента S4: интервал Ньюкомба (сверка со статьёй), объём выборки,
успех пользователя по SQLite бота, симуляция тем же путём, что бот."""

import sqlite3
from pathlib import Path

import pytest

from assistant.bot.store import Store
from evals.answer_metrics import Rate
from evals.experiment import arms, newcombe, render, simulate, users_needed


def test_newcombe_matches_the_paper() -> None:
    """Newcombe (1998), пример a: 56/70 − 48/80, метод 10 — 0,0524…0,3339."""
    diff, low, high = newcombe(Rate(56, 70), Rate(48, 80))

    assert diff == pytest.approx(0.2)
    assert (round(low, 4), round(high, 4)) == (0.0524, 0.3339)


def test_users_needed_for_two_proportions() -> None:
    # 50 % → 60 % при α = 0,05 и мощности 80 % — 385–388 на группу (по формуле
    # с объединённой дисперсией под нулевой гипотезой).
    assert 385 <= users_needed(0.5, 0.1) <= 388
    assert users_needed(0.4, 0.05) > users_needed(0.4, 0.10) > users_needed(0.4, 0.15)


def test_solved_means_a_thumbs_up_and_no_human(tmp_path: Path) -> None:
    store = Store(tmp_path / "bot.sqlite3")
    # 1 — 👍, решил; 2 — 👍, но ушёл к человеку; 3 — без оценки; 4 — вне эксперимента.
    for chat, arm in ((1, "auto"), (2, "auto"), (3, "always"), (4, None)):
        answer = store.save_answer(
            chat,
            status="answer",
            reason=None,
            model="m",
            recipe_ids=[],
            seconds=1,
            arm=arm,
        )
        if chat in (1, 2, 4):
            store.vote(answer, chat, "up")
    store.open_ticket(2, "request", "")
    store.close()
    db = sqlite3.connect(f"file:{tmp_path / 'bot.sqlite3'}?mode=ro", uri=True)

    groups = arms(db, 0.0)

    assert (groups["auto"].users, groups["auto"].solved) == (2, 1)
    assert (groups["always"].users, groups["always"].solved) == (1, 0)


def test_simulation_goes_through_the_bot_store_and_is_reproducible(
    tmp_path: Path,
) -> None:
    first = simulate(400, 0.4, 0.4, seed=1, path=tmp_path / "a.sqlite3")
    again = simulate(400, 0.4, 0.4, seed=1, path=tmp_path / "b.sqlite3")

    assert first == again
    assert first["always"].users + first["auto"].users == 400
    assert 150 < first["auto"].users < 250


def test_report_says_when_the_interval_covers_zero() -> None:
    from assistant.bot import experiment
    from evals.experiment import Arm

    groups: dict[str, Arm] = {
        experiment.CONTROL: Arm("always", 100, 40),
        experiment.TEST: Arm("auto", 100, 45),
    }

    text = render(groups, "т")

    assert "интервал накрывает ноль — разницы не видно, данных мало" in text
    assert "| +5 п. п. |" in text
