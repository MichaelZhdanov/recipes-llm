"""Бот на aiogram 3: тонкие обработчики над `BotService` и запуск long polling.

- Отвечает только в личных чатах: в группе эскалации бот — администратор и
  видит все её сообщения, отвечать там ассистенту нельзя.
- Пока готовится ответ (~11 с), статус-сообщение меняется по стадиям ответа,
  а Telegram показывает «печатает…»; готовый ответ заменяет статус.
- В одном чате — по одному вопросу: новое сообщение во время ответа получает
  «секунду, ещё отвечаю». Разные чаты отвечаются параллельно.
- Сообщения старше STALE_AFTER пропускаются: после простоя бота отвечать на
  вчерашнее странно.
- 👍/👎 под ответом: после оценки они исчезают, после 👎 бот спрашивает
  «что не так?» тремя кнопками.
- «🍳 Шаги» — только у рецептов без живой страницы: шаги пишет и проверяет
  `generation/steps.py` (~20 с), бот присылает их отдельным сообщением.
- «📖 Шаги с фото» — у рецептов с живой страницей: фото блюда и сколько на
  странице шагов, сами шаги — по ссылке (`recipe_page.py`).
- Здоровье для Docker — файл-сердцебиение (`health.py`): раз в минуту бот
  спрашивает Telegram getMe и при успехе обновляет файл.
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
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender
from qdrant_client import QdrantClient

from assistant.bot.format import (
    BUSY_TEXT,
    HELP_TEXT,
    NOT_TEXT,
    START_TEXT,
    STEPS_STATUS_TEXT,
    TOO_LONG_TEXT,
    Keyboard,
    status_text,
)
from assistant.bot.health import HEARTBEAT, HEARTBEAT_EVERY
from assistant.bot.pages import PageChecker
from assistant.bot.recipe_page import fetch_card
from assistant.bot.service import ERROR_REPLY, BotService, RecipeLookup, Reply
from assistant.bot.store import Store, Vote, VoteReason
from assistant.config import Settings
from assistant.generation.answer import Stage
from assistant.generation.llm import make_http_client
from assistant.generation.steps import make_steps, make_steps_models
from assistant.indexing.chunks import Chunk
from assistant.retrieval.store import VectorStore
from assistant.service import MAX_QUESTION_CHARS, LazyAnswerer

logger = logging.getLogger(__name__)

DATABASE = Path("data/bot/bot.sqlite3")
STALE_AFTER = 10 * 60
# Сколько поток ассистента ждёт правку статуса.
STATUS_TIMEOUT = 5.0
VOTES: dict[str, Vote] = {"up": "up", "down": "down"}
REASONS: dict[str, VoteReason] = {
    "unclear": "unclear",
    "wrong": "wrong",
    "failed": "failed",
}


def keyboard(rows: Keyboard) -> InlineKeyboardMarkup | None:
    if not rows:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=data) for text, data in row]
            for row in rows
        ]
    )


def without_votes(markup: InlineKeyboardMarkup | None) -> InlineKeyboardMarkup | None:
    """Клавиатура ответа после оценки: 👍/👎 убираются, кнопки шагов остаются."""
    if markup is None:
        return None
    rows = [
        row
        for row in markup.inline_keyboard
        if not any((button.callback_data or "").startswith("vote:") for button in row)
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def parse_callback(data: str | None, prefix: str) -> tuple[int, str] | None:
    """«prefix:<id ответа>:<значение>» → (id, значение); чужое — None."""
    parts = (data or "").split(":")
    if len(parts) != 3 or parts[0] != prefix or not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2]


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

    async def vote(self, callback: CallbackQuery) -> None:
        parsed = parse_callback(callback.data, "vote")
        message = callback.message
        if parsed is None or parsed[1] not in VOTES or message is None:
            await callback.answer()
            return
        answer_id, value = parsed
        toast, follow = await asyncio.to_thread(
            self.service.vote, message.chat.id, answer_id, VOTES[value]
        )
        await callback.answer(toast)
        # Сообщение старше 48 часов Telegram отдаёт без текста и кнопок.
        if not isinstance(message, Message):
            return
        await message.edit_reply_markup(
            reply_markup=without_votes(message.reply_markup)
        )
        if follow is not None:
            await message.answer(follow.text, reply_markup=keyboard(follow.buttons))

    async def why(self, callback: CallbackQuery) -> None:
        parsed = parse_callback(callback.data, "why")
        message = callback.message
        if parsed is None or parsed[1] not in REASONS or message is None:
            await callback.answer()
            return
        answer_id, value = parsed
        text = await asyncio.to_thread(
            self.service.vote_reason, message.chat.id, answer_id, REASONS[value]
        )
        await callback.answer(text)
        if isinstance(message, Message):
            await message.edit_text(text)

    async def card(self, callback: CallbackQuery, bot: Bot) -> None:
        parsed = parse_callback(callback.data, "card")
        message = callback.message
        if parsed is None or not parsed[1].isdigit() or message is None:
            await callback.answer()
            return
        await callback.answer()
        chat_id = message.chat.id
        try:
            reply = await asyncio.to_thread(
                self.service.card, chat_id, parsed[0], int(parsed[1])
            )
        except Exception:
            logger.exception("карточка рецепта не собрана")
            reply = ERROR_REPLY
        if reply.photo is not None:
            try:
                await bot.send_photo(chat_id, reply.photo, caption=reply.text)
                return
            except TelegramBadRequest:
                # Telegram не смог забрать фото по адресу — тогда без фото.
                logger.warning("фото рецепта не отправилось", exc_info=True)
        await bot.send_message(chat_id, reply.text)

    async def steps(self, callback: CallbackQuery, bot: Bot) -> None:
        parsed = parse_callback(callback.data, "steps")
        message = callback.message
        if parsed is None or not parsed[1].isdigit() or message is None:
            await callback.answer()
            return
        chat_id = message.chat.id
        if chat_id in self.busy:
            await callback.answer(BUSY_TEXT)
            return
        self.busy.add(chat_id)
        try:
            await callback.answer()
            status = await bot.send_message(chat_id, STEPS_STATUS_TEXT)
            async with ChatActionSender.typing(chat_id=chat_id, bot=bot):
                try:
                    reply = await asyncio.to_thread(
                        self.service.steps, chat_id, parsed[0], int(parsed[1])
                    )
                except Exception:
                    logger.exception("шаги не собраны")
                    reply = ERROR_REPLY
            await status.edit_text(reply.text, reply_markup=keyboard(reply.buttons))
        finally:
            self.busy.discard(chat_id)


def build_dispatcher(handlers: Handlers) -> Dispatcher:
    router = Router()
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    router.callback_query.filter(F.message.chat.type == ChatType.PRIVATE)
    router.message.register(handlers.start, CommandStart())
    router.message.register(handlers.help, Command("help"))
    router.message.register(handlers.new, Command("new"))
    router.message.register(handlers.question, F.text)
    router.message.register(handlers.other)
    router.callback_query.register(handlers.vote, F.data.startswith("vote:"))
    router.callback_query.register(handlers.why, F.data.startswith("why:"))
    router.callback_query.register(handlers.steps, F.data.startswith("steps:"))
    router.callback_query.register(handlers.card, F.data.startswith("card:"))
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


async def heartbeat(
    bot: Bot, path: Path = HEARTBEAT, every: float = HEARTBEAT_EVERY
) -> None:
    while True:
        try:
            await bot.get_me()
            path.touch()
        except Exception:
            # Сеть до Telegram или сбой API: проверка здоровья это увидит.
            logger.warning("Telegram не ответил на getMe", exc_info=True)
        await asyncio.sleep(every)


def recipe_lookup(store: VectorStore) -> RecipeLookup:
    def lookup(recipe_id: int) -> Chunk | None:
        found = store.recipes([recipe_id])
        return found[0] if found else None

    return lookup


async def run(settings: Settings) -> None:
    token = settings.telegram_bot_token.get_secret_value()
    if not token:
        raise SystemExit("не задан TELEGRAM_BOT_TOKEN: токен бота — у @BotFather")
    store = Store(DATABASE)
    answerer = LazyAnswerer(settings)
    pages_http = httpx2.Client()
    steps_http = make_http_client(settings)
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=30)
    writer, reviewer = make_steps_models(settings, steps_http)
    service = BotService(
        answerer,
        store,
        PageChecker(pages_http, store),
        recipes=recipe_lookup(VectorStore(qdrant)),
        make_steps=lambda text: make_steps(text, writer, reviewer),
        fetch_card=lambda url: fetch_card(pages_http, url),
        gigachat_note=settings.llm_provider == "gigachat",
    )
    bot = make_bot(token)
    dispatcher = build_dispatcher(Handlers(service))
    beat = asyncio.create_task(heartbeat(bot))
    try:
        await bot.set_my_commands(COMMANDS)
        await dispatcher.start_polling(
            bot, allowed_updates=["message", "callback_query"]
        )
    finally:
        beat.cancel()
        await bot.session.close()
        answerer.close()
        pages_http.close()
        steps_http.close()
        qdrant.close()
        store.close()
