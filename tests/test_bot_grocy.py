"""Склад Grocy в боте (этап 6): включение, выбор источника на разговор,
расширенный поиск «не хватает 1–4» и честный ответ, если Grocy недоступен."""

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx2
from aiogram.methods import EditMessageReplyMarkup, EditMessageText, SendMessage
from bot_helpers import FakeTelegram, press, update

from assistant.bot.app import Handlers, build_dispatcher, make_bot
from assistant.bot.format import (
    ASK_AGAIN_TEXT,
    GONE_TEXT,
    GROCY_ALREADY_OFF_TEXT,
    GROCY_ALREADY_ON_TEXT,
    GROCY_OFF_TEXT,
    GROCY_ON_TEXT,
    GROCY_UNAVAILABLE_TEXT,
    NEAR_BUTTON,
    NEAR_OFFER_TEXT,
    SOURCE_QUESTION_TEXT,
    STOCK_DOWN_TEXT,
)
from assistant.bot.pages import PageChecker
from assistant.bot.service import BotService
from assistant.bot.store import Store
from assistant.generation.answer import (
    Answer,
    AnswerResult,
    AnswerStatement,
    Progress,
    SourceLink,
    StockRequest,
    Trace,
)
from assistant.generation.gigachat import Message
from assistant.grocy.client import GrocyError, StockItem

TOKEN = "42:TEST-secret-token"
USER = 7
URL = "https://www.povarenok.ru/recipes/show/1/"


def answer(**fields: Any) -> Answer:
    return Answer(
        status="answer",
        reason=None,
        message="🍳 Вот что нашлось:",
        statements=[AnswerStatement(text="🥚 Омлет из яиц и молока", sources=[1])],
        sources=[SourceLink(number=1, recipe_id=1, name="Омлет", url=URL)],
        model="m",
        prompt_version="recipes_v5",
        dataset_revision="ffaf49ee",
        **fields,
    )


class RecordingAnswerer:
    """Запоминает, с каким складом спросили; отвечает заданным ответом."""

    def __init__(self, reply: Answer | None = None) -> None:
        self.reply = reply or answer()
        self.calls: list[tuple[str, StockRequest | None]] = []

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        self.calls.append((question, stock))
        trace = Trace([], [], None, [], 0.0, 0.0)
        return AnswerResult(answer=self.reply, trace=trace)


def stock_reader(fail: bool = False) -> Any:
    def read() -> list[StockItem]:
        if fail:
            raise GrocyError("Grocy недоступен: ConnectError")
        return [StockItem("яйцо куриное", 10), StockItem("молоко", 2)]

    return read


def service(
    answerer: RecordingAnswerer | None = None,
    *,
    stock: Any = None,
    grocy: bool = True,
    recipes: Any = None,
) -> BotService:
    store = Store(":memory:")
    http = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(200)))
    return BotService(
        answerer or RecordingAnswerer(),
        store,
        PageChecker(http, store),
        recipes=recipes or (lambda recipe_id: None),
        make_steps=None,
        fetch_page=lambda url: None,
        fetch_photo=lambda url: None,
        stock=(stock or stock_reader()) if grocy else None,
    )


def test_connect_and_disconnect_are_separate_and_idempotent() -> None:
    assert service(grocy=False).connect_grocy(USER) == GROCY_UNAVAILABLE_TEXT
    bot_service = service()

    assert bot_service.disconnect_grocy(USER) == GROCY_ALREADY_OFF_TEXT
    assert bot_service.connect_grocy(USER) == GROCY_ON_TEXT
    assert bot_service.connect_grocy(USER) == GROCY_ALREADY_ON_TEXT
    assert bot_service.disconnect_grocy(USER) == GROCY_OFF_TEXT
    assert not bot_service.store.grocy_enabled(USER)


def test_without_grocy_enabled_the_bot_answers_as_before() -> None:
    answerer = RecordingAnswerer()

    service(answerer).ask(USER, "что на ужин")

    assert answerer.calls == [("что на ужин", None)]


def test_new_conversation_asks_the_source_once_and_follow_ups_keep_it() -> None:
    answerer = RecordingAnswerer()
    bot_service = service(answerer)
    bot_service.connect_grocy(USER)

    asked = bot_service.ask(USER, "что на ужин")
    assert asked.text == SOURCE_QUESTION_TEXT and answerer.calls == []
    assert [b[1] for b in asked.buttons[0]] == ["source:stock", "source:all"]

    bot_service.choose_source(USER, "stock")
    bot_service.ask(USER, "а без молока?")

    (first, first_stock), (second, second_stock) = answerer.calls
    assert first == "что на ужин" and second == "а без молока?"
    assert first_stock == StockRequest(names=("яйцо куриное", "молоко"))
    assert second_stock == first_stock
    # /new — новый разговор: снова спросить.
    bot_service.reset(USER)
    assert bot_service.ask(USER, "десерт").text == SOURCE_QUESTION_TEXT


def test_all_recipes_choice_searches_without_the_stock() -> None:
    answerer = RecordingAnswerer()
    bot_service = service(answerer)
    bot_service.connect_grocy(USER)
    bot_service.ask(USER, "что на ужин")

    bot_service.choose_source(USER, "all")

    assert answerer.calls == [("что на ужин", None)]
    assert service().choose_source(USER, "all").text == ASK_AGAIN_TEXT


def test_strict_answer_offers_near_and_near_lists_what_is_missing() -> None:
    answerer = RecordingAnswerer(answer(near_available=True))
    bot_service = service(answerer)
    bot_service.connect_grocy(USER)
    bot_service.ask(USER, "что испечь")
    strict = bot_service.choose_source(USER, "stock")
    assert strict.answer_id is not None
    # Строка автора и под ней кнопка «Поискать ещё».
    assert strict.text.endswith("Есть рецепты, в которых не хватает пары продуктов.")
    assert strict.text.endswith(NEAR_OFFER_TEXT)
    assert strict.buttons[-1] == [(NEAR_BUTTON, f"near:{strict.answer_id}")]
    assert NEAR_BUTTON == "🔎 Поискать ещё"

    answerer.reply = answer(missing={1: ["мука пшеничная", "сахар"]})
    near = bot_service.near(USER, strict.answer_id)

    question, request = answerer.calls[-1]
    assert question == "что испечь" and request is not None and request.near
    assert "🛒 Не хватает: мука пшеничная, сахар" in near.text


def test_forged_near_button_with_another_chats_answer_reveals_nothing() -> None:
    """Ловушка: клиент Telegram может прислать любые callback_data (Bot API
    предупреждает об этом). Чужой ответ — «уже не найти», модель не вызывается."""
    answerer = RecordingAnswerer(answer(near_available=True))
    bot_service = service(answerer)
    bot_service.connect_grocy(USER)
    bot_service.ask(USER, "что испечь")
    strict = bot_service.choose_source(USER, "stock")
    assert strict.answer_id is not None
    calls = len(answerer.calls)

    forged = bot_service.near(USER + 1, strict.answer_id)

    assert forged.text == GONE_TEXT and forged.answer_id is None
    assert len(answerer.calls) == calls


def test_grocy_down_is_said_honestly_and_search_goes_over_all_recipes() -> None:
    answerer = RecordingAnswerer()
    bot_service = service(answerer, stock=stock_reader(fail=True))
    bot_service.connect_grocy(USER)
    bot_service.ask(USER, "что на ужин")

    reply = bot_service.choose_source(USER, "stock")

    assert reply.text.startswith(STOCK_DOWN_TEXT)
    assert answerer.calls == [("что на ужин", None)]


def test_source_button_answers_in_place_of_a_status_and_disappears() -> None:
    bot_service = service()
    telegram = FakeTelegram()
    bot = make_bot(TOKEN, session=telegram)
    dispatcher = build_dispatcher(Handlers(bot_service))
    markup = {
        "inline_keyboard": [
            [
                {"text": "🧺", "callback_data": "source:stock"},
                {"text": "📖", "callback_data": "source:all"},
            ]
        ]
    }

    async def scenario() -> None:
        for raw in (
            update("/grocy"),
            update("что на ужин", update_id=2),
            press("source:stock", message_id=102, markup=markup),
        ):
            await dispatcher.feed_raw_update(bot, raw)

    asyncio.run(scenario())

    sent = [m.text for m in telegram.calls if isinstance(m, SendMessage)]
    assert sent[0] == GROCY_ON_TEXT
    edits = [m for m in telegram.calls if isinstance(m, EditMessageText)]
    assert edits[0].text == SOURCE_QUESTION_TEXT
    assert (edits[-1].text or "").startswith("🍳 Вот что нашлось:")
    (removed,) = [m for m in telegram.calls if isinstance(m, EditMessageReplyMarkup)]
    assert removed.message_id == 102 and removed.reply_markup is None


def test_start_offers_to_connect_with_a_button_and_help_explains_both_commands() -> (
    None
):
    from aiogram.types import InlineKeyboardMarkup

    from assistant.bot.format import HELP_TEXT, START_GROCY_OFFER_TEXT, START_TEXT

    def run(bot_service: BotService, *raws: dict[str, Any]) -> FakeTelegram:
        telegram = FakeTelegram()
        bot = make_bot(TOKEN, session=telegram)
        dispatcher = build_dispatcher(Handlers(bot_service))

        async def scenario() -> None:
            for raw in raws:
                await dispatcher.feed_raw_update(bot, raw)

        asyncio.run(scenario())
        return telegram

    bot_service = service()
    telegram = run(
        bot_service,
        update("/start"),
        update("/help", update_id=2),
        press("grocy:on", message_id=101),
        update("/start", update_id=3),
        update("/grocy_off", update_id=4),
    )

    start, help_, connected, start_again, off = [
        m for m in telegram.calls if isinstance(m, SendMessage)
    ]
    assert START_GROCY_OFFER_TEXT in (start.text or "")
    assert isinstance(start.reply_markup, InlineKeyboardMarkup)
    assert start.reply_markup.inline_keyboard[0][0].callback_data == "grocy:on"
    assert "/grocy_off — отключить склад Grocy" in (help_.text or "")
    assert "общий склад на сервере" in (help_.text or "")
    assert connected.text == GROCY_ON_TEXT
    # Подключён — под /start кнопки нет, а текст напоминает, как отключить.
    assert start_again.reply_markup is None and "/grocy_off" in (start_again.text or "")
    assert off.text == GROCY_OFF_TEXT
    # Без Grocy на сервере — всё как раньше: ни предложения, ни команд склада.
    plain = run(service(grocy=False), update("/start"), update("/help", update_id=2))
    assert [m.text for m in plain.calls if isinstance(m, SendMessage)] == [
        START_TEXT,
        HELP_TEXT,
    ]


def test_steps_of_a_near_recipe_say_what_is_missing() -> None:
    """«📖 Шаги» у рецепта из «поискать ещё»: в конце заголовка — чего не хватает."""
    from chunk_helpers import make_chunk

    salad = make_chunk(
        "Салат-гарнир из перловой крупы", {"Перловка": "200 г"}, recipe_id=1
    )
    answerer = RecordingAnswerer(answer(near_available=True))
    bot_service = service(answerer, recipes=lambda recipe_id: salad)
    bot_service.connect_grocy(USER)
    bot_service.ask(USER, "гарнир")
    strict = bot_service.choose_source(USER, "stock")
    assert strict.answer_id is not None
    answerer.reply = answer(missing={1: ["крупа перловая", "укроп"]})
    near = bot_service.near(USER, strict.answer_id)
    assert near.answer_id is not None

    header, *_ = bot_service.page_steps(USER, near.answer_id, salad.recipe_id)
    plain, *_ = bot_service.page_steps(USER, strict.answer_id, salad.recipe_id)

    assert isinstance(header, str) and header.startswith(
        "📖 <b>Салат-гарнир из перловой крупы</b>"
    )
    assert header.endswith("🛒 Не хватает: крупа перловая, укроп")
    # У строгого ответа ничего докупать не нужно — строки нет.
    assert isinstance(plain, str) and "Не хватает" not in plain
