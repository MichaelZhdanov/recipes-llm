"""Хранилище бота: история с окном и сбросом, оценки, кэши."""

from pathlib import Path

from assistant.bot.store import HISTORY_LIMIT, SESSION_TIMEOUT, Store


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
