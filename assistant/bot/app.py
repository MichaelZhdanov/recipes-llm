"""Бот на aiogram 3: тонкие обработчики над `BotService` и запуск long polling.

- Отвечает только в личных чатах: в группе эскалации бот — администратор и
  видит все её сообщения, отвечать там ассистенту нельзя.
- Пока готовится ответ (~11 с), статус-сообщение меняется по стадиям ответа,
  а Telegram показывает «печатает…»; готовый ответ заменяет статус.
- В одном чате — по одному вопросу: новое сообщение во время ответа получает
  «секунду, ещё отвечаю». Разные чаты отвечаются параллельно.
- Сообщения старше STALE_AFTER пропускаются: после простоя бота отвечать на
  вчерашнее странно.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from pathlib import Path

import httpx2
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.base import BaseSession
from aiogram.enums import ChatType, ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender

from assistant.bot.format import (
    BUSY_TEXT,
    HELP_TEXT,
    NOT_TEXT,
    START_TEXT,
    TOO_LONG_TEXT,
    Keyboard,
    status_text,
)
from assistant.bot.pages import PageChecker
from assistant.bot.service import ERROR_REPLY, BotService, Reply
from assistant.bot.store import Store
from assistant.config import Settings
from assistant.generation.answer import Stage
from assistant.service import MAX_QUESTION_CHARS, LazyAnswerer

logger = logging.getLogger(__name__)

DATABASE = Path("data/bot/bot.sqlite3")
STALE_AFTER = 10 * 60
# Сколько поток ассистента ждёт правку статуса.
STATUS_TIMEOUT = 5.0


def keyboard(rows: Keyboard) -> InlineKeyboardMarkup | None:
    if not rows:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=data) for text, data in row]
            for row in rows
        ]
    )


class Handlers:
    def __init__(
        self, service: BotService, clock: Callable[[], float] = time.time
    ) -> None:
        self.service = service
        self.busy: set[int] = set()
        self._clock = clock

    async def start(self, message: Message) -> None:
        await message.answer(START_TEXT)

    async def help(self, message: Message) -> None:
        await message.answer(HELP_TEXT)

    async def new(self, message: Message) -> None:
        text = await asyncio.to_thread(self.service.reset, message.chat.id)
        await message.answer(text)

    async def other(self, message: Message) -> None:
        await message.answer(NOT_TEXT)

    async def question(self, message: Message, bot: Bot) -> None:
        if message.date.timestamp() < self._clock() - STALE_AFTER:
            return
        text = (message.text or "").strip()
        if len(text) > MAX_QUESTION_CHARS:
            await message.answer(TOO_LONG_TEXT)
            return
        chat_id = message.chat.id
        # Проверка и захват — без await между ними: в одном цикле событий
        # второй обработчик того же чата не может вклиниться.
        if chat_id in self.busy:
            await message.answer(BUSY_TEXT)
            return
        self.busy.add(chat_id)
        try:
            await self._answer(message, bot, text)
        finally:
            self.busy.discard(chat_id)

    async def _answer(self, message: Message, bot: Bot, text: str) -> None:
        chat_id = message.chat.id
        status = await message.answer(status_text("search"))
        loop = asyncio.get_running_loop()
        shown: set[Stage] = {"search"}

        def progress(stage: Stage) -> None:
            # Вызывается из потока ассистента. Правка ждёт завершения, чтобы
            # запоздавший статус не затёр уже готовый ответ.
            if stage in shown:
                return
            shown.add(stage)
            future = asyncio.run_coroutine_threadsafe(self._status(status, stage), loop)
            try:
                future.result(timeout=STATUS_TIMEOUT)
            except Exception:
                logger.debug("статус не обновился", exc_info=True)

        async with ChatActionSender.typing(chat_id=chat_id, bot=bot):
            try:
                reply = await asyncio.to_thread(
                    self.service.ask, chat_id, text, progress
                )
            except Exception:
                # Ошибка в коде, а не во внешнем API: пользователь не должен
                # смотреть на вечное «ищу рецепты…».
                logger.exception("ответ не собран")
                reply = ERROR_REPLY
        await self._show(status, reply)

    async def _show(self, status: Message, reply: Reply) -> None:
        await status.edit_text(reply.text, reply_markup=keyboard(reply.buttons))
        if reply.answer_id is not None:
            await asyncio.to_thread(
                self.service.store.set_message_id, reply.answer_id, status.message_id
            )

    @staticmethod
    async def _status(status: Message, stage: Stage) -> None:
        # Та же фраза или сообщение уже заменено ответом — не страшно.
        with contextlib.suppress(TelegramBadRequest):
            await status.edit_text(status_text(stage))


def build_dispatcher(handlers: Handlers) -> Dispatcher:
    router = Router()
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    router.message.register(handlers.start, CommandStart())
    router.message.register(handlers.help, Command("help"))
    router.message.register(handlers.new, Command("new"))
    router.message.register(handlers.question, F.text)
    router.message.register(handlers.other)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


def make_bot(token: str, session: BaseSession | None = None) -> Bot:
    """session — для тестов: фейковый Bot API без сети."""
    return Bot(
        token,
        session=session,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
            # Пять ссылок на рецепты — превью первой только путает.
            link_preview=LinkPreviewOptions(is_disabled=True),
        ),
    )


COMMANDS = [
    BotCommand(command="new", description="начать разговор заново"),
    BotCommand(command="help", description="что я умею"),
]


async def run(settings: Settings) -> None:
    token = settings.telegram_bot_token.get_secret_value()
    if not token:
        raise SystemExit("не задан TELEGRAM_BOT_TOKEN: токен бота — у @BotFather")
    store = Store(DATABASE)
    answerer = LazyAnswerer(settings)
    pages_http = httpx2.Client()
    service = BotService(
        answerer,
        store,
        PageChecker(pages_http, store),
        gigachat_note=settings.llm_provider == "gigachat",
    )
    bot = make_bot(token)
    dispatcher = build_dispatcher(Handlers(service))
    try:
        await bot.set_my_commands(COMMANDS)
        await dispatcher.start_polling(
            bot, allowed_updates=["message", "callback_query"]
        )
    finally:
        await bot.session.close()
        answerer.close()
        pages_http.close()
        store.close()
