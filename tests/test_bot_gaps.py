"""Пробелы базы (S1): «не нашёл» по причинам и похожим вопросам, 👎 с причиной,
передачи человеку; период; база — только на чтение."""

import sqlite3
from pathlib import Path

import pytest

from assistant.bot.gaps import DAY, collect, main, normalize, render
from assistant.bot.store import Store

NOW = 1_800_000_000.0


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def database(tmp_path: Path) -> Path:
    path = tmp_path / "bot.sqlite3"
    clock = Clock(NOW - 40 * DAY)
    store = Store(path, clock)
    # Старше периода — в отчёт не попадает.
    store.save_answer(
        1,
        status="not_found",
        reason="no_sources",
        model="m",
        recipe_ids=[],
        seconds=1,
        question="старый вопрос",
    )
    clock.now = NOW - DAY
    for question, reason in (
        ("Борщ без свёклы!", "model_not_found"),
        ("борщ без свеклы", "no_sources"),
        ("что-то с кальмарами", "length"),
    ):
        store.save_answer(
            1,
            status="not_found",
            reason=reason,
            model="m",
            recipe_ids=[],
            seconds=1,
            question=question,
        )
    answered = store.save_answer(
        2,
        status="answer",
        reason=None,
        model="m",
        recipe_ids=[7],
        seconds=1,
        question="омлет на завтрак",
    )
    store.vote(answered, 2, "down")
    store.vote_reason(answered, 2, "wrong")
    store.open_ticket(3, "health", "После грибов тошнит, просит совета.")
    store.close()
    return path


def test_normalize_groups_the_same_question_written_differently() -> None:
    assert (
        normalize("Борщ без свёклы!")
        == normalize("  борщ   без свеклы ")
        == "борщ без свеклы"
    )


def test_report_groups_not_found_and_lists_downvotes_and_tickets(
    tmp_path: Path,
) -> None:
    db = sqlite3.connect(f"file:{database(tmp_path)}?mode=ro", uri=True)

    gaps = collect(db, days=30, now=NOW)
    text = render(gaps, NOW)

    assert gaps.answers == 4  # старый вопрос — за пределами 30 дней
    assert "Ответов: 4; «не нашёл» — 3 (75%); 👎 — 1; передано человеку — 1." in text
    assert (
        "| 2 | борщ без свеклы | среди найденного не подошло ни одно, поиск ничего не нашёл |"
        in text
    )
    assert "ответ оборвался на лимите токенов — 1" in text
    assert "| не то искал | омлет на завтрак |" in text
    assert "| 🔴 здоровье | После грибов тошнит, просит совета. | открыто |" in text
    assert "старый вопрос" not in text


def test_answers_from_before_questions_were_stored_say_so(tmp_path: Path) -> None:
    store = Store(tmp_path / "bot.sqlite3", Clock(NOW - DAY))
    store.save_answer(
        1, status="not_found", reason=None, model="m", recipe_ids=[], seconds=1
    )
    store.close()
    db = sqlite3.connect(f"file:{tmp_path / 'bot.sqlite3'}?mode=ro", uri=True)

    text = render(collect(db, days=30, now=NOW), NOW)

    assert "| 1 | — вопрос не сохранялся (ответ до этапа 6) |" in text


def test_report_opens_the_database_read_only(tmp_path: Path) -> None:
    db = sqlite3.connect(f"file:{database(tmp_path)}?mode=ro", uri=True)

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        db.execute("DELETE FROM answers")


def test_main_prints_the_report_or_says_there_is_no_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--db", str(database(tmp_path)), "--days", "3650"]) == 0
    assert "# Пробелы базы за 3650 дн." in capsys.readouterr().out
    assert main(["--db", str(tmp_path / "нет.sqlite3")]) == 1
    assert "базы бота нет" in capsys.readouterr().err
