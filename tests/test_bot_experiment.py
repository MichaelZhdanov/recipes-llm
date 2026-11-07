"""Эксперимент S4 в боте: группа по псевдониму — постоянная и ≈ 50/50; выключен —
всё как было; включён — тест отвечает «модель решает», группа в журнале и SQLite."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import httpx2
from pydantic import SecretStr

from assistant.bot import experiment
from assistant.bot.journal import Journal
from assistant.bot.pages import PageChecker
from assistant.bot.service import BotService
from assistant.bot.store import Store
from assistant.generation.answer import (
    Answer,
    AnswerResult,
    Progress,
    StockRequest,
    Trace,
)
from assistant.generation.gigachat import Message

KEY = SecretStr("experiment-key")


class Answerer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.questions: list[str] = []

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        self.questions.append(question)
        answer = Answer(
            status="not_found",
            reason="model_not_found",
            statements=[],
            sources=[],
            model=self.name,
            prompt_version="p",
            dataset_revision="r",
        )
        return AnswerResult(answer, Trace([], [], None, [], 0.0, 0.0))


def chat_in(arm: experiment.Arm) -> int:
    journal = Journal(KEY)
    return next(c for c in range(1, 1000) if experiment.arm(journal.user(c)) == arm)


def service(
    control: Answerer,
    test: Answerer | None,
    lines: list[str],
    store: Store | None = None,
) -> BotService:
    store = store or Store(":memory:")
    http = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(200)))
    return BotService(
        control,
        store,
        PageChecker(http, store),
        recipes=lambda recipe_id: None,
        make_steps=None,
        fetch_page=lambda url: None,
        fetch_photo=lambda url: None,
        journal=Journal(KEY, write=lines.append),
        test_answerer=test,
    )


def test_arm_is_stable_per_user_and_splits_users_about_in_half() -> None:
    journal = Journal(KEY)
    arms = [experiment.arm(journal.user(chat)) for chat in range(2000)]

    assert experiment.arm(journal.user(42)) == experiment.arm(Journal(KEY).user(42))
    assert 0.45 < arms.count(experiment.TEST) / len(arms) < 0.55


def test_without_the_experiment_everyone_gets_the_usual_assistant() -> None:
    control = Answerer("always")
    lines: list[str] = []
    store = Store(":memory:")

    service(control, None, lines, store).ask(chat_in(experiment.TEST), "борщ")

    assert control.questions == ["борщ"]
    assert "arm" not in json.loads(lines[0])
    assert store._db.execute("SELECT arm FROM answers").fetchone() == (None,)


def test_with_the_experiment_the_test_arm_gets_model_decides(tmp_path: Path) -> None:
    control, test = Answerer("always"), Answerer("auto")
    lines: list[str] = []
    store = Store(tmp_path / "bot.sqlite3")
    bot = service(control, test, lines, store)

    bot.ask(chat_in(experiment.TEST), "что-нибудь к чаю")
    bot.ask(chat_in(experiment.CONTROL), "борщ")

    assert test.questions == ["что-нибудь к чаю"] and control.questions == ["борщ"]
    assert [json.loads(line)["arm"] for line in lines] == ["auto", "always"]
    rows = store._db.execute("SELECT arm, model FROM answers ORDER BY id").fetchall()
    assert rows == [("auto", "auto"), ("always", "always")]


def test_migration_adds_the_arm_column_to_an_old_database(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE answers (id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, "
        "message_id INTEGER, status TEXT NOT NULL, reason TEXT, model TEXT NOT NULL, "
        "recipe_ids TEXT NOT NULL, seconds REAL NOT NULL, vote TEXT, vote_reason TEXT, "
        "created_at REAL NOT NULL, voted_at REAL)"
    )
    old.commit()
    old.close()

    store = Store(path)

    columns = {row[1] for row in store._db.execute("PRAGMA table_info(answers)")}
    assert {"question", "missing", "arm"} <= columns
