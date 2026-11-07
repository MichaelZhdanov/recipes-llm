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
- «📖 Шаги с фото» — у рецептов с живой страницей: шаги со страницы по
  порядку, по сообщению на шаг — фото с текстом шага (`recipe_page.py`).
- Здоровье для Docker — файл-сердцебиение (`health.py`): раз в минуту бот
  спрашивает Telegram getMe и при успехе обновляет файл.
- Передача человеку (этап 5): в группе операторов (ESCALATION_CHAT_ID,
  супергруппа с темами) бот создаёт тему «#N · повод» с карточкой и дальше
  копирует сообщения (copyMessage) в обе стороны — оператор не видит, кто
  пишет, ассистент молчит. /close в теме, закрытие темы кнопкой Telegram или
  /bot у пользователя возвращают ассистента. Других групп бот не слушает.
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
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
)
from aiogram.utils.chat_action import ChatActionSender
from qdrant_client import QdrantClient

from assistant.bot.escalation import topic_name
from assistant.bot.format import (
    ALREADY_WITH_HUMAN_TEXT,
    BUSY_TEXT,
    HELP_TEXT,
    NO_OPERATORS_TEXT,
    NOT_TEXT,
    OPERATOR_ONLY_TEXT,
    START_TEXT,
    STEPS_STATUS_TEXT,
    TOO_LONG_TEXT,
    TOPIC_CLOSED_TEXT,
    USER_LEFT_TEXT,
    Keyboard,
    status_text,
)
from assistant.bot.health import HEARTBEAT, HEARTBEAT_EVERY
from assistant.bot.pages import PageChecker
from assistant.bot.recipe_page import fetch_page, fetch_photo
from assistant.bot.service import (
    ERROR_REPLY,
    BotService,
    Part,
    RecipeLookup,
    Reply,
)
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
# Пауза между сообщениями шагов со страницы рецепта, с.
STEP_PAUSE = 0.5
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
        self,
        service: BotService,
        clock: Callable[[], float] = time.time,
        *,
        operators_chat: int | None = None,
    ) -> None:
        """operators_chat — супергруппа операторов с темами (ESCALATION_CHAT_ID)."""
        self.service = service
        self.busy: set[int] = set()
        self._clock = clock
        self.operators_chat = operators_chat

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
        if reply.handoff is not None:
            reply = await self._hand_over(bot, chat_id, reply)
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
        reply = await asyncio.to_thread(
            self.service.vote_reason, message.chat.id, answer_id, REASONS[value]
        )
        await callback.answer()
        if isinstance(message, Message):
            await message.edit_text(reply.text, reply_markup=keyboard(reply.buttons))

    async def page_steps(self, callback: CallbackQuery, bot: Bot) -> None:
        parsed = parse_callback(callback.data, "card")
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
            async with ChatActionSender.upload_photo(chat_id=chat_id, bot=bot):
                try:
                    parts = await asyncio.to_thread(
                        self.service.page_steps, chat_id, parsed[0], int(parsed[1])
                    )
                except Exception:
                    logger.exception("шаги со страницы не собраны")
                    parts = [ERROR_REPLY.text]
                for number, part in enumerate(parts):
                    if number:
                        # Telegram ограничивает частоту сообщений в один чат.
                        await asyncio.sleep(STEP_PAUSE)
                    await _send_part(bot, chat_id, part)
        finally:
            self.busy.discard(chat_id)

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

    # --- Передача человеку ---

    async def human(self, message: Message, bot: Bot) -> None:
        """/human — явная просьба позвать человека."""
        reply = await asyncio.to_thread(self.service.human, message.chat.id, "request")
        await self._send(bot, message.chat.id, reply)

    async def human_button(self, callback: CallbackQuery, bot: Bot) -> None:
        """«Позвать повара» после сбоя, второго «не нашёл» или 👎 «не получилось»."""
        message = callback.message
        reason = (callback.data or "").partition(":")[2]
        await callback.answer()
        if message is None:
            return
        reply = await asyncio.to_thread(self.service.human, message.chat.id, reason)
        if isinstance(message, Message):
            # Кнопка нажата — второй раз звать не нужно.
            with contextlib.suppress(TelegramBadRequest):
                await message.edit_reply_markup(reply_markup=None)
        await self._send(bot, message.chat.id, reply)

    async def _send(self, bot: Bot, chat_id: int, reply: Reply) -> None:
        if reply.handoff is not None:
            reply = await self._hand_over(bot, chat_id, reply)
        await bot.send_message(
            chat_id, reply.text, reply_markup=keyboard(reply.buttons)
        )

    async def _hand_over(self, bot: Bot, chat_id: int, reply: Reply) -> Reply:
        """Тема «#N · повод» с карточкой в группе операторов; не вышло — обращения
        нет, пользователь остаётся с ботом и узнаёт об этом честно."""
        handoff = reply.handoff
        if handoff is None or self.operators_chat is None:
            return Reply(NO_OPERATORS_TEXT)
        ticket, card = await asyncio.to_thread(
            self.service.open_ticket, chat_id, handoff
        )
        if ticket.thread_id is not None:
            return Reply(ALREADY_WITH_HUMAN_TEXT)
        try:
            topic = await bot.create_forum_topic(
                self.operators_chat, topic_name(ticket.id, handoff.reason)
            )
            await asyncio.to_thread(
                self.service.store.set_thread, ticket.id, topic.message_thread_id
            )
            await bot.send_message(
                self.operators_chat, card, message_thread_id=topic.message_thread_id
            )
        except TelegramAPIError:
            logger.exception("тема обращения %s не создана", ticket.id)
            await asyncio.to_thread(self.service.cancel_ticket, ticket)
            return Reply(NO_OPERATORS_TEXT)
        return reply

    async def in_handover(self, message: Message) -> bool:
        """Фильтр: у чата открыто обращение — сообщения идут человеку."""
        ticket = await asyncio.to_thread(
            self.service.store.open_ticket_of, message.chat.id
        )
        return ticket is not None

    async def to_operator(self, message: Message, bot: Bot) -> None:
        """Сообщение пользователя — копией в тему обращения: кто пишет, не видно."""
        ticket = await asyncio.to_thread(
            self.service.store.open_ticket_of, message.chat.id
        )
        if ticket is None or ticket.thread_id is None or self.operators_chat is None:
            return
        try:
            await bot.copy_message(
                self.operators_chat,
                message.chat.id,
                message.message_id,
                message_thread_id=ticket.thread_id,
            )
        except TelegramBadRequest:
            # Тему удалили или закрыли, а служебное сообщение не дошло.
            logger.warning("тема обращения %s недоступна", ticket.id, exc_info=True)
            await self._closed(bot, ticket.thread_id)

    async def from_operator(self, message: Message, bot: Bot) -> None:
        """Сообщение оператора в теме — копией пользователю."""
        thread = message.message_thread_id
        if thread is None or not message.is_topic_message:
            return
        ticket = await asyncio.to_thread(self.service.store.ticket_of_thread, thread)
        if ticket is None:
            return
        try:
            await bot.copy_message(ticket.chat_id, message.chat.id, message.message_id)
        except TelegramForbiddenError:
            await message.reply("Пользователь недоступен: он остановил бота.")

    async def close(self, message: Message, bot: Bot) -> None:
        """/close в теме обращения."""
        thread = message.message_thread_id
        if thread is None or not await self._closed(bot, thread):
            await message.reply(OPERATOR_ONLY_TEXT)
            return
        await message.reply(TOPIC_CLOSED_TEXT)
        if self.operators_chat is not None:
            with contextlib.suppress(TelegramBadRequest):
                await bot.close_forum_topic(self.operators_chat, thread)

    async def topic_closed(self, message: Message, bot: Bot) -> None:
        """Тему закрыли кнопкой Telegram — как /close."""
        if message.message_thread_id is not None:
            await self._closed(bot, message.message_thread_id)

    async def _closed(self, bot: Bot, thread: int) -> bool:
        """Закрыть обращение темы и вернуть пользователя ассистенту."""
        closed = await asyncio.to_thread(self.service.close_by_operator, thread)
        if closed is None:
            return False
        reply, ticket = closed
        with contextlib.suppress(TelegramForbiddenError):
            await bot.send_message(
                ticket.chat_id, reply.text, reply_markup=keyboard(reply.buttons)
            )
        return True

    async def back(self, message: Message, bot: Bot) -> None:
        """/bot у пользователя: обращение закрывается, ассистент снова отвечает."""
        reply, ticket = await asyncio.to_thread(
            self.service.back_to_bot, message.chat.id
        )
        await message.answer(reply.text, reply_markup=keyboard(reply.buttons))
        if ticket is None or ticket.thread_id is None or self.operators_chat is None:
            return
        with contextlib.suppress(TelegramBadRequest):
            await bot.send_message(
                self.operators_chat, USER_LEFT_TEXT, message_thread_id=ticket.thread_id
            )
            await bot.close_forum_topic(self.operators_chat, ticket.thread_id)

    async def rate(self, callback: CallbackQuery) -> None:
        """👍/👎 разговору с поваром."""
        parsed = parse_callback(callback.data, "rate")
        message = callback.message
        if parsed is None or parsed[1] not in VOTES or message is None:
            await callback.answer()
            return
        text = await asyncio.to_thread(
            self.service.rate, message.chat.id, parsed[0], VOTES[parsed[1]]
        )
        await callback.answer(text)
        if isinstance(message, Message):
            with contextlib.suppress(TelegramBadRequest):
                await message.edit_reply_markup(reply_markup=None)


async def _send_part(bot: Bot, chat_id: int, part: Part) -> None:
    """Текст — сообщением, шаг с фото — фото с подписью. Telegram не принял
    фото — подпись текстом; «слишком часто» — ждём, сколько он сказал."""
    for attempt in range(2):
        try:
            if isinstance(part, str):
                await bot.send_message(chat_id, part)
            else:
                data, caption = part
                photo = BufferedInputFile(data, filename="step.jpg")
                try:
                    await bot.send_photo(chat_id, photo, caption=caption)
                except TelegramBadRequest:
                    logger.warning("фото шага не отправилось", exc_info=True)
                    await bot.send_message(chat_id, caption)
            return
        except TelegramRetryAfter as error:
            if attempt:
                raise
            await asyncio.sleep(error.retry_after)


def build_dispatcher(handlers: Handlers) -> Dispatcher:
    router = Router()
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    router.callback_query.filter(F.message.chat.type == ChatType.PRIVATE)
    # Порядок важен: /bot — раньше пересылки, пересылка — раньше ассистента.
    router.message.register(handlers.back, Command("bot"))
    router.message.register(handlers.to_operator, handlers.in_handover)
    router.message.register(handlers.start, CommandStart())
    router.message.register(handlers.help, Command("help"))
    router.message.register(handlers.new, Command("new"))
    router.message.register(handlers.human, Command("human"))
    router.message.register(handlers.question, F.text)
    router.message.register(handlers.other)
    router.callback_query.register(handlers.vote, F.data.startswith("vote:"))
    router.callback_query.register(handlers.why, F.data.startswith("why:"))
    router.callback_query.register(handlers.steps, F.data.startswith("steps:"))
    # «card:» — с прошлой версии кнопки: старые сообщения тоже работают.
    router.callback_query.register(handlers.page_steps, F.data.startswith("card:"))
    router.callback_query.register(handlers.human_button, F.data.startswith("human:"))
    router.callback_query.register(handlers.rate, F.data.startswith("rate:"))
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    if handlers.operators_chat is not None:
        # Группа операторов: только её темы, только люди, без команд, кроме /close.
        group = Router()
        group.message.filter(F.chat.id == handlers.operators_chat)
        group.message.register(handlers.close, Command("close"))
        group.message.register(handlers.topic_closed, F.forum_topic_closed)
        group.message.register(
            handlers.from_operator,
            F.message_thread_id,
            ~F.from_user.is_bot,
            ~F.text.startswith("/"),
        )
        dispatcher.include_router(group)
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
    BotCommand(command="human", description="позвать повара-консультанта"),
    BotCommand(command="bot", description="вернуться к ассистенту от повара"),
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
    # Шаги от ИИ проверяет Sonnet, как и ответ (принцип 1): без проверки ответа
    # (ANSWER_REVIEW=false) их не пишем — у рецепта без страницы нет кнопки.
    steps_models = (
        make_steps_models(settings, steps_http) if settings.answer_review else None
    )
    service = BotService(
        answerer,
        store,
        PageChecker(pages_http, store),
        recipes=recipe_lookup(VectorStore(qdrant)),
        make_steps=(
            (lambda text: make_steps(text, *steps_models)) if steps_models else None
        ),
        fetch_page=lambda url: fetch_page(pages_http, url),
        fetch_photo=lambda url: fetch_photo(pages_http, url),
        gigachat_note=settings.llm_provider == "gigachat",
        operators=settings.escalation_chat_id is not None,
    )
    bot = make_bot(token)
    dispatcher = build_dispatcher(
        Handlers(service, operators_chat=settings.escalation_chat_id)
    )
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
