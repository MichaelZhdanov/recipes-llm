"""Хранилище бота: история с окном и сбросом, оценки, кэши, обращения к человеку."""

from pathlib import Path

from assistant.bot.store import HISTORY_LIMIT, SESSION_TIMEOUT, Store, Ticket


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def store(clock: Clock) -> Store:
    return Store(":memory:", clock=clock)


def test_history_keeps_pairs_and_the_last_messages_in_order() -> None:
    clock = Clock()
    db = store(clock)
    for n in range(7):
        db.add_turn(1, f"вопрос {n}", f"ответ {n}")

    history = db.history(1)

    assert len(history) == HISTORY_LIMIT
    assert history[0] == {"role": "user", "content": "вопрос 2"}
    assert history[-1] == {"role": "assistant", "content": "ответ 6"}
    assert db.history(2) == []


def test_new_command_and_silence_start_a_new_session() -> None:
    clock = Clock()
    db = store(clock)
    db.add_turn(1, "блины на молоке", "Блины: https://www.povarenok.ru/recipes/show/1/")

    db.reset(1)
    assert db.history(1) == []

    db.add_turn(1, "а без яиц?", "…")
    clock.now += SESSION_TIMEOUT + 1
    assert db.history(1) == []
    db.add_turn(1, "оладьи", "…")
    assert [m["content"] for m in db.history(1)] == ["оладьи", "…"]


def test_question_is_stored_without_personal_data() -> None:
    db = store(Clock())

    db.add_turn(1, "пишите на ivan@example.com, что приготовить", "…")

    assert "ivan@example.com" not in db.history(1)[0]["content"]


def test_vote_and_reason_belong_to_the_chat_of_the_answer() -> None:
    db = store(Clock())
    answer_id = db.save_answer(
        1,
        status="answer",
        reason=None,
        model="claude-haiku-5-5",
        recipe_ids=[7, 9],
        seconds=11.0,
    )

    assert not db.vote(answer_id, 2, "down")  # чужой чат
    assert not db.vote_reason(answer_id, 1, "failed")  # причина — только после 👎
    assert db.vote(answer_id, 1, "down")
    assert db.vote_reason(answer_id, 1, "failed")
    assert db.recipe_ids(answer_id, 1) == [7, 9]
    assert db.recipe_ids(answer_id, 2) is None


def test_page_cache_expires_and_steps_cache_is_per_prompt_version() -> None:
    clock = Clock()
    db = store(clock)
    url = "https://www.povarenok.ru/recipes/show/1/"

    db.save_page(url, ok=False)
    assert db.page_ok(url, max_age=60) is False
    clock.now += 61
    assert db.page_ok(url, max_age=60) is None

    db.save_steps(1, "1. Смешайте…", approved=True, prompt_version="steps_v1")
    saved = db.steps(1, "steps_v1")
    assert saved is not None and saved.approved and saved.text.startswith("1.")
    assert db.steps(1, "steps_v2") is None


def test_database_file_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "bot" / "bot.sqlite3"
    first = Store(path, clock=Clock())
    first.add_turn(1, "вопрос", "ответ")
    first.close()

    second = Store(path, clock=Clock())

    assert len(second.history(1)) == 2
    second.close()


# --- Обращения к человеку ---


def test_one_open_ticket_per_chat_and_it_is_found_by_chat_and_thread() -> None:
    db = store(Clock())

    first = db.open_ticket(7, "health", "тошнит после грибов")
    again = db.open_ticket(7, "request", "просит человека")
    db.set_thread(first.id, 55)

    assert again == first
    assert first.status == "open" and first.thread_id is None
    found = db.open_ticket_of(7)
    assert found is not None and found.thread_id == 55
    assert db.ticket_of_thread(55) == found
    assert db.open_ticket_of(8) is None


def test_closed_ticket_frees_the_chat_and_is_closed_once() -> None:
    db = store(Clock())
    ticket = db.open_ticket(7, "request", "")
    db.set_thread(ticket.id, 55)

    assert db.close_ticket(ticket.id) is True
    # /close в теме и /bot у пользователя могут прийти одновременно.
    assert db.close_ticket(ticket.id) is False
    assert db.open_ticket_of(7) is None
    assert db.ticket_of_thread(55) is None
    assert db.open_ticket(7, "feedback", "").id != ticket.id


def test_only_own_closed_ticket_can_be_rated() -> None:
    db = store(Clock())
    ticket = db.open_ticket(7, "request", "")

    assert db.vote_ticket(ticket.id, 7, "up") is False
    db.close_ticket(ticket.id)
    assert db.vote_ticket(ticket.id, 8, "up") is False
    assert db.vote_ticket(ticket.id, 7, "down") is True


def test_recent_statuses_are_newest_first_within_the_session() -> None:
    clock = Clock()
    db = store(clock)
    for status in ("answer", "not_found"):
        db.save_answer(
            7, status=status, reason=None, model="m", recipe_ids=[], seconds=1
        )
    clock.now += SESSION_TIMEOUT + 1
    db.save_answer(
        7, status="not_found", reason=None, model="m", recipe_ids=[], seconds=1
    )

    # Старый «не нашёл» — из прошлой сессии: «два подряд» он не даёт.
    assert db.recent_statuses(7) == ["not_found"]
    assert db.recent_statuses(8) == []


def test_tickets_survive_reopening(tmp_path: Path) -> None:
    path = tmp_path / "bot.sqlite3"
    db = Store(path)
    ticket = db.open_ticket(7, "health", "сыпь после торта")
    db.set_thread(ticket.id, 55)
    db.close()

    reopened = Store(path)
    assert reopened.open_ticket_of(7) == Ticket(
        ticket.id, 7, 55, "health", "сыпь после торта", "open"
    )


def test_idle_tickets_count_silence_from_the_last_message() -> None:
    clock = Clock()
    db = store(clock)
    ticket = db.open_ticket(7, "request", "")
    clock.now += 5 * 3600
    db.touch_ticket(ticket.id)  # сообщение через 5 часов

    clock.now += 11 * 3600  # 16 часов от открытия, 11 — от сообщения
    assert db.idle_tickets(12 * 3600) == []
    clock.now += 2 * 3600
    assert [t.id for t in db.idle_tickets(12 * 3600)] == [ticket.id]
    db.close_ticket(ticket.id)
    assert db.idle_tickets(12 * 3600) == []


def test_database_without_last_at_is_migrated(tmp_path: Path) -> None:
    """База версии 3 (этап 5 до автозакрытия): CREATE TABLE IF NOT EXISTS
    столбец не добавит — его добавляет миграция, время — от открытия."""
    import sqlite3

    path = tmp_path / "bot.sqlite3"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE tickets (id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, "
        "thread_id INTEGER, reason TEXT NOT NULL, summary TEXT NOT NULL, "
        "status TEXT NOT NULL, vote TEXT, opened_at REAL NOT NULL, closed_at REAL);"
        "INSERT INTO tickets (chat_id, reason, summary, status, opened_at) "
        "VALUES (7, 'request', '', 'open', 100.0);"
        "PRAGMA user_version = 3;"
    )
    old.commit()
    old.close()

    clock = Clock()  # 1000.0 — через 900 с после открытия
    db = Store(path, clock=clock)

    assert [t.id for t in db.idle_tickets(800)] == [1]
    assert db.idle_tickets(1000) == []


# --- Склад Grocy ---


def test_grocy_choice_lives_in_the_conversation_and_resets_on_toggle() -> None:
    clock = Clock()
    db = store(clock)
    assert not db.grocy_enabled(7) and db.source(7) is None

    db.set_grocy(7, True)
    db.set_source(7, "stock")
    db.add_turn(7, "что на ужин", "…")
    assert db.grocy_enabled(7) and db.source(7) == "stock"

    db.reset(7)  # /new — новый разговор: спросить заново
    assert db.source(7) is None
    db.set_source(7, "all")
    clock.now += SESSION_TIMEOUT + 1  # 30 минут тишины — тоже новый
    assert db.source(7) is None
    db.set_source(7, "stock")
    db.set_grocy(7, False)
    assert not db.grocy_enabled(7) and db.source(7) is None


def test_pending_question_is_taken_once_and_masked() -> None:
    db = store(Clock())

    db.save_pending(7, "что на ужин? мой тел. +7 916 123-45-67")
    question = db.pop_pending(7)

    assert question is not None and question.startswith("что на ужин?")
    assert "916" not in question
    assert db.pop_pending(7) is None


def test_answer_keeps_its_question_for_the_extended_search() -> None:
    db = store(Clock())
    answer_id = db.save_answer(
        7,
        status="answer",
        reason=None,
        model="m",
        recipe_ids=[],
        seconds=1,
        question="ужин",
    )

    assert db.question_of(answer_id, 7) == "ужин"
    assert db.question_of(answer_id, 8) is None


def test_answers_without_question_column_are_migrated(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "bot.sqlite3"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE answers (id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, "
        "message_id INTEGER, status TEXT NOT NULL, reason TEXT, model TEXT NOT NULL, "
        "recipe_ids TEXT NOT NULL, seconds REAL NOT NULL, vote TEXT, vote_reason TEXT, "
        "created_at REAL NOT NULL, voted_at REAL);"
        "INSERT INTO answers (chat_id, status, model, recipe_ids, seconds, created_at) "
        "VALUES (7, 'answer', 'm', '[]', 1, 1);"
    )
    old.commit()
    old.close()

    db = Store(path)

    assert db.question_of(1, 7) is None
    new = db.save_answer(
        7,
        status="answer",
        reason=None,
        model="m",
        recipe_ids=[],
        seconds=1,
        question="суп",
    )
    assert db.question_of(new, 7) == "суп"


def test_answer_remembers_what_its_recipes_lack() -> None:
    db = store(Clock())
    near = db.save_answer(
        7,
        status="answer",
        reason=None,
        model="m",
        recipe_ids=[1, 2],
        seconds=1,
        missing={1: ["мука пшеничная", "сахар"]},
    )
    plain = db.save_answer(
        7, status="answer", reason=None, model="m", recipe_ids=[1], seconds=1
    )

    assert db.missing_of(near, 7, 1) == ["мука пшеничная", "сахар"]
    assert db.missing_of(near, 7, 2) == [] and db.missing_of(plain, 7, 1) == []
    assert db.missing_of(near, 8, 1) == []  # чужой чат
