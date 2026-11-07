"""Лимит частоты (этап 7, F7; решение 75): 10 обращений к модели за 10 минут
и 50 в сутки на чат; бесплатные действия не считаются; лимит переживает
перезапуск."""

import json
from collections.abc import Sequence
from pathlib import Path

import httpx2
from pydantic import SecretStr

from assistant.bot.journal import Journal
from assistant.bot.limits import LIMITS, Limit, exceeded, wait_text
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
from assistant.grocy.client import StockItem

CHAT = 7
MINUTE = 60.0


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class Answerer:
    def __init__(self) -> None:
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
            message="",
            statements=[],
            sources=[],
            model="m",
            prompt_version="recipes_v5",
            dataset_revision="ffaf49ee",
        )
        return AnswerResult(answer, Trace([], [], None, [], 0.0, 0.0))


def service(
    store: Store, answerer: Answerer, lines: list[str] | None = None
) -> BotService:
    http = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(200)))
    return BotService(
        answerer,
        store,
        PageChecker(http, store),
        recipes=lambda recipe_id: None,
        make_steps=None,
        fetch_page=lambda url: None,
        fetch_photo=lambda url: None,
        stock=lambda: [StockItem("яйцо куриное", 10)],
        journal=Journal(
            SecretStr("k"), write=(lines if lines is not None else []).append
        ),
    )


def test_limits_are_the_authors_numbers() -> None:
    assert (
        Limit(10, 10 * MINUTE, "за 10 минут"),
        Limit(50, 24 * 60 * MINUTE, "в сутки"),
    ) == LIMITS


def test_eleventh_question_in_ten_minutes_waits_for_the_oldest_to_expire() -> None:
    clock, answerer = Clock(), Answerer()
    lines: list[str] = []
    bot = service(Store(":memory:", clock), answerer, lines)
    for n in range(10):
        bot.ask(CHAT, f"вопрос {n}")
        clock.now += 30  # десять вопросов за 5 минут

    limited = bot.ask(CHAT, "одиннадцатый")

    # Первый вопрос выйдет из окна через 10 − 5 = 5 минут.
    assert limited.text == (
        "⏳ Слишком много вопросов: не больше 10 за 10 минут. "
        "Следующий можно задать через 5 мин."
    )
    assert limited.answer_id is None and len(answerer.questions) == 10
    assert json.loads(lines[-1])["event"] == "limited"
    # Другой чат не ждёт.
    assert bot.ask(CHAT + 1, "вопрос").text != limited.text

    clock.now += 5 * MINUTE
    assert bot.ask(CHAT, "снова можно").text != limited.text
    assert answerer.questions[-1] == "снова можно"


def test_fifty_a_day_even_when_spread_out() -> None:
    clock, answerer = Clock(), Answerer()
    bot = service(Store(":memory:", clock), answerer)
    for n in range(50):
        bot.ask(CHAT, f"вопрос {n}")
        clock.now += 2 * MINUTE  # в окно 10 минут никогда не больше пяти

    limited = bot.ask(CHAT, "пятьдесят первый")

    # Первый вопрос — 100 минут назад: выйдет из суток через 22 ч 20 мин.
    assert "не больше 50 в сутки" in limited.text
    assert limited.text.endswith("через 23 ч.")
    assert len(answerer.questions) == 50


def test_only_actions_that_call_the_model_are_counted() -> None:
    clock, answerer = Clock(), Answerer()
    store = Store(":memory:", clock)
    bot = service(store, answerer)
    bot.connect_grocy(CHAT)
    for _ in range(5):
        bot.start(CHAT)
        bot.reset(CHAT)
        bot.ask(CHAT, "позовите человека")  # передача человеку — кодом, без модели
    # Вопрос нового разговора ждёт выбора источника — модель ещё не звали.
    bot.ask(CHAT, "что на ужин")
    assert store.ask_times(CHAT, 60 * MINUTE) == []

    bot.choose_source(CHAT, "stock")

    assert store.ask_times(CHAT, 60 * MINUTE) == [clock.now]


def test_limit_survives_a_restart(tmp_path: Path) -> None:
    clock, path = Clock(), tmp_path / "bot.sqlite3"
    first = Store(path, clock)
    bot = service(first, Answerer())
    for n in range(10):
        bot.ask(CHAT, f"вопрос {n}")
    first.close()

    again = service(Store(path, clock), Answerer())

    assert again.ask(CHAT, "после перезапуска").text.startswith("⏳")


def test_old_asks_are_pruned() -> None:
    clock = Clock()
    store = Store(":memory:", clock)
    store.add_ask(CHAT, keep=60)
    clock.now += 61
    store.add_ask(CHAT + 1, keep=60)

    assert store.ask_times(CHAT, 1e9) == []


def test_exceeded_picks_the_longest_wait_and_wait_rounds_up() -> None:
    limits = (Limit(2, 100, "a"), Limit(3, 1000, "b"))

    assert exceeded([10, 20], now=30, limits=limits) == exceeded(
        [10, 20], 30, limits[:1]
    )
    both = exceeded([0, 10, 20], now=30, limits=limits)
    assert both is not None and both.limit.label == "b" and both.wait == 970
    assert exceeded([10], now=30, limits=limits) is None
    assert wait_text(1) == "через 1 мин"
    assert wait_text(61) == "через 2 мин"
    assert wait_text(3600) == "через 1 ч"
    assert wait_text(3601) == "через 2 ч"
