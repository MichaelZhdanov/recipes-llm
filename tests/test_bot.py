"""Бот: логика BotService и обработчики aiogram на фейковом Bot API.

Обработчики проверяются целиком: апдейт Telegram (словарь, как из getUpdates)
→ Dispatcher.feed_raw_update → исходящие вызовы Bot API, которые записывает
фейковая сессия. Сети и настоящего токена нет; async — через asyncio.run.
"""

import asyncio
import json
import logging
import sys
import threading
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import httpx2
import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import EditMessageText, SendChatAction, SendMessage, TelegramMethod
from aiogram.methods.base import TelegramType
from pydantic import SecretStr

from assistant.bot.app import Handlers, build_dispatcher, make_bot
from assistant.bot.format import (
    BUSY_TEXT,
    ERROR_TEXT,
    HELP_TEXT,
    NEW_TEXT,
    NOT_TEXT,
    START_TEXT,
    STATUS,
    TOO_LONG_TEXT,
)
from assistant.bot.logs import MASK, RedactingFormatter, secrets_of
from assistant.bot.pages import PageChecker
from assistant.bot.service import BotService
from assistant.bot.store import Store
from assistant.config import Settings
from assistant.generation.answer import (
    Answer,
    AnswerResult,
    AnswerStatement,
    Progress,
    SourceLink,
    Trace,
)
from assistant.generation.gigachat import Message
from assistant.retrieval.embeddings import EmbeddingError

TOKEN = "42:TEST-secret-token"
ALIVE = "https://www.povarenok.ru/recipes/show/1/"
DEAD = "https://www.povarenok.ru/recipes/show/2/"
SLOW = "долгий вопрос"


def recipe_answer() -> Answer:
    return Answer(
        status="answer",
        reason=None,
        message="🥞 Вот что нашлось:",
        statements=[AnswerStatement(text="🥛 Молока — 500 мл", sources=[1])],
        sources=[
            SourceLink(number=1, recipe_id=1, name="Блины на молоке", url=ALIVE),
            SourceLink(number=2, recipe_id=2, name="Оладьи", url=DEAD),
        ],
        model="claude-haiku-5-5",
        prompt_version="recipes_v3",
        dataset_revision="ffaf49ee",
    )


class FakeAnswerer:
    """Отвечает одним и тем же рецептом, проходя стадии, как настоящий."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.histories: list[list[Message]] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
    ) -> AnswerResult:
        self.histories.append(list(history))
        if question == SLOW:
            self.entered.set()
            self.release.wait(timeout=5)
        for stage in ("search", "answer", "review"):
            if progress is not None:
                progress(stage)
        if self.error is not None:
            raise self.error
        trace = Trace(
            hits=[],
            sources=[],
            completion=None,
            checked=[],
            retrieval_seconds=0.0,
            generation_seconds=0.0,
        )
        return AnswerResult(answer=recipe_answer(), trace=trace)


class RecordingStore(Store):
    """Хранилище, которое помнит, к какому сообщению привязан ответ."""

    def __init__(self) -> None:
        super().__init__(":memory:")
        self.message_ids: list[tuple[int, int]] = []

    def set_message_id(self, answer_id: int, message_id: int) -> None:
        self.message_ids.append((answer_id, message_id))
        super().set_message_id(answer_id, message_id)


def make_service(answerer: FakeAnswerer) -> BotService:
    store = RecordingStore()
    pages = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200 if str(request.url) == ALIVE else 404)
        )
    )
    return BotService(answerer, store, PageChecker(pages, store))


# --- BotService ---


def test_follow_up_question_gets_previous_turn_with_recipe_links() -> None:
    answerer = FakeAnswerer()
    service = make_service(answerer)

    service.ask(7, "блины на молоке")
    service.ask(7, "а сколько молока?")

    assert answerer.histories[0] == []
    previous = answerer.histories[1]
    assert [m["role"] for m in previous] == ["user", "assistant"]
    assert previous[0]["content"] == "блины на молоке"
    assert ALIVE in previous[1]["content"]
    # У другого чата — своя история.
    service.ask(8, "оладьи")
    assert answerer.histories[2] == []


def test_answer_links_live_pages_and_offers_steps_for_dead_ones() -> None:
    reply = make_service(FakeAnswerer()).ask(7, "блины")

    assert reply.answer_id is not None
    assert f'<a href="{ALIVE}">способ приготовления</a>' in reply.text
    assert "Оладьи</b> — страница рецепта на сайте недоступна" in reply.text
    assert reply.buttons[0][0] == ("👍", f"vote:{reply.answer_id}:up")
    assert reply.buttons[1] == [("🍳 Шаги: Оладьи", f"steps:{reply.answer_id}:2")]


def test_failure_of_a_dependency_leaves_history_untouched() -> None:
    answerer = FakeAnswerer(error=EmbeddingError("cloud.ru не ответил"))
    service = make_service(answerer)

    reply = service.ask(7, "блины")

    assert reply.text == ERROR_TEXT
    assert reply.buttons == [] and reply.answer_id is None
    assert service.store.history(7) == []


def test_reset_forgets_the_conversation() -> None:
    service = make_service(FakeAnswerer())
    service.ask(7, "блины")

    assert service.reset(7) == NEW_TEXT
    assert service.store.history(7) == []


# --- Обработчики aiogram на фейковом Bot API ---


class FakeTelegram(BaseSession):
    """Bot API без сети: записывает вызовы и отвечает, как ответил бы Telegram."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []
        self._message_id = 100

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: int | None = None,
    ) -> TelegramType:
        self.calls.append(method)
        result: object = True
        if isinstance(method, SendMessage):
            self._message_id += 1
            result = self._message(method.chat_id, self._message_id, method.text)
        elif isinstance(method, EditMessageText):
            result = self._message(method.chat_id, method.message_id, method.text)
        content = json.dumps({"ok": True, "result": result})
        return self.check_response(bot, method, 200, content).result  # type: ignore[return-value]

    @staticmethod
    def _message(
        chat_id: object, message_id: object, text: str | None
    ) -> dict[str, object]:
        return {
            "message_id": message_id,
            "date": int(time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        }

    async def close(self) -> None:
        pass

    async def stream_content(
        self, *args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        raise NotImplementedError
        yield b""

    def sent(self) -> list[str]:
        return [m.text for m in self.calls if isinstance(m, SendMessage)]

    def edits(self) -> list[EditMessageText]:
        return [m for m in self.calls if isinstance(m, EditMessageText)]


def update(
    text: str | None,
    *,
    chat_id: int = 7,
    chat_type: str = "private",
    age: float = 0.0,
    update_id: int = 1,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": update_id,
        "date": int(time.time() - age),
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": chat_id, "is_bot": False, "first_name": "Тест"},
    }
    if text is None:
        message["photo"] = [
            {"file_id": "p", "file_unique_id": "p", "width": 1, "height": 1}
        ]
    else:
        message["text"] = text
    return {"update_id": update_id, "message": message}


def feed(service: BotService, *updates: dict[str, Any]) -> FakeTelegram:
    telegram = FakeTelegram()
    bot = make_bot(TOKEN, session=telegram)
    dispatcher = build_dispatcher(Handlers(service))

    async def scenario() -> None:
        for raw in updates:
            await dispatcher.feed_raw_update(bot, raw)

    asyncio.run(scenario())
    return telegram


def test_commands_answer_with_their_texts() -> None:
    telegram = feed(
        make_service(FakeAnswerer()),
        update("/start"),
        update("/help", update_id=2),
        update("/new", update_id=3),
        update(None, update_id=4),
        update("x" * 2001, update_id=5),
    )

    assert telegram.sent() == [START_TEXT, HELP_TEXT, NEW_TEXT, NOT_TEXT, TOO_LONG_TEXT]


def test_question_shows_status_by_stage_then_the_answer_in_its_place() -> None:
    service = make_service(FakeAnswerer())

    telegram = feed(service, update("блины на молоке"))

    assert telegram.sent()[0] in STATUS["search"]
    assert any(
        isinstance(m, SendChatAction) and m.action == "typing" for m in telegram.calls
    )
    *statuses, final = telegram.edits()
    # «search» уже на экране: дальше по одной правке на стадию.
    assert len(statuses) == 2
    assert statuses[0].text in STATUS["answer"]
    assert statuses[1].text in STATUS["review"]
    assert {e.message_id for e in telegram.edits()} == {101}
    assert "Молока — 500 мл" in (final.text or "")
    assert final.reply_markup is not None
    assert final.reply_markup.inline_keyboard[0][0].callback_data == "vote:1:up"
    assert isinstance(service.store, RecordingStore)
    assert service.store.message_ids == [(1, 101)]


def test_failed_answer_replaces_status_with_apology() -> None:
    for error in (EmbeddingError("нет сети"), RuntimeError("ошибка в коде")):
        telegram = feed(make_service(FakeAnswerer(error=error)), update("блины"))

        final = telegram.edits()[-1]
        assert final.text == ERROR_TEXT
        assert final.reply_markup is None


def test_groups_and_stale_messages_are_ignored() -> None:
    answerer = FakeAnswerer()
    telegram = feed(
        make_service(answerer),
        update("блины", chat_type="supergroup", chat_id=-100),
        update("/start", chat_type="group", chat_id=-5, update_id=2),
        update("блины", age=11 * 60, update_id=3),
    )

    assert telegram.calls == []
    assert answerer.histories == []


def test_second_question_in_the_same_chat_waits_other_chats_do_not() -> None:
    answerer = FakeAnswerer()
    service = make_service(answerer)
    telegram = FakeTelegram()
    bot = make_bot(TOKEN, session=telegram)
    dispatcher = build_dispatcher(Handlers(service))

    async def scenario() -> None:
        first = asyncio.create_task(dispatcher.feed_raw_update(bot, update(SLOW)))
        while not answerer.entered.is_set():
            await asyncio.sleep(0.01)
        await dispatcher.feed_raw_update(bot, update("а без яиц?", update_id=2))
        await dispatcher.feed_raw_update(bot, update("оладьи", chat_id=8, update_id=3))
        answerer.release.set()
        await first

    asyncio.run(scenario())

    assert BUSY_TEXT in telegram.sent()
    # Вопрос чата 8 ответился, пока чат 7 ждал; «а без яиц?» в ассистент не ушёл.
    assert len(answerer.histories) == 2
    finals = [e for e in telegram.edits() if "Молока" in (e.text or "")]
    assert {e.chat_id for e in finals} == {7, 8}


# --- Секреты в логах ---


def test_token_is_not_logged_by_the_bot_or_aiogram(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        feed(
            make_service(FakeAnswerer()), update("/start"), update("блины", update_id=2)
        )

    assert caplog.records
    assert TOKEN not in caplog.text


def test_formatter_masks_secrets_in_message_and_traceback() -> None:
    settings = Settings(
        telegram_bot_token=SecretStr(TOKEN), anthropic_api_key=SecretStr("sk-test")
    )
    secrets = secrets_of(settings)
    formatter = RedactingFormatter(secrets)
    try:
        raise ConnectionError(f"https://api.telegram.org/bot{TOKEN}/getUpdates")
    except ConnectionError:
        record = logging.LogRecord(
            "bot", logging.ERROR, __file__, 1, "ключ %s", ("sk-test",), sys.exc_info()
        )

    text = formatter.format(record)

    assert sorted(secrets) == sorted([TOKEN, "sk-test"])
    assert TOKEN not in text and "sk-test" not in text
    assert f"bot{MASK}/getUpdates" in text and f"ключ {MASK}" in text
