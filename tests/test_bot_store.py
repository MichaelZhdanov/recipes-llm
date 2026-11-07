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
