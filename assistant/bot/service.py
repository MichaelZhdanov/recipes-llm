"""Логика бота без aiogram: что ответить на вопрос, /new, оценку и кнопку шагов.

Синхронная, как и весь ассистент: обработчики aiogram (`app.py`) вызывают её
через `asyncio.to_thread`. Здесь всё, что стоит проверить тестами: история
диалога, ссылки на живые страницы, сохранение ответа и оценки, ошибки.

Передача человеку (этап 5): сервис решает, когда передать и что сказать
пользователю (Reply.handoff), и ведёт обращения в SQLite; тему в группе
операторов создаёт и сообщения пересылает приложение — это вызовы Telegram.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from assistant.bot.escalation import (
    OFFERED_REASONS,
    TICKET_IDLE,
    card_html,
    wants_human,
)
from assistant.bot.format import (
    ALREADY_WITH_HUMAN_TEXT,
    ASK_AGAIN_TEXT,
    BACK_TO_BOT_TEXT,
    CLOSED_BY_OPERATOR_TEXT,
    ERROR_TEXT,
    EXPIRED_TEXT,
    FAILED_TEXT,
    GONE_TEXT,
    GROCY_ALREADY_OFF_TEXT,
    GROCY_ALREADY_ON_TEXT,
    GROCY_OFF_TEXT,
    GROCY_ON_TEXT,
    GROCY_UNAVAILABLE_TEXT,
    HANDOFF_TEXT,
    HEALTH_TEXT,
    LIMIT_TEXT,
    NEAR_OFFER_TEXT,
    NEW_TEXT,
    NO_OPERATORS_TEXT,
    NOT_FOUND_TWICE_TEXT,
    NOT_WITH_HUMAN_TEXT,
    SOURCE_QUESTION_TEXT,
    STOCK_DOWN_TEXT,
    THANKS_TEXT,
    WHAT_WRONG_TEXT,
    Keyboard,
    answer_buttons,
    answer_html,
    human_buttons,
    near_buttons,
    page_steps_header,
    rate_buttons,
    reason_buttons,
    source_buttons,
    start_buttons,
    start_text,
    step_caption,
    steps_html,
    steps_refused_html,
)
from assistant.bot.journal import Journal, question_text, search_fields
from assistant.bot.limits import KEEP, LIMITS, exceeded, wait_text
from assistant.bot.pages import PageChecker
from assistant.bot.recipe_page import RecipePage
from assistant.bot.store import SavedSteps, Source, Store, Ticket, Vote, VoteReason
from assistant.generation import steps as steps_module
from assistant.generation.answer import (
    AnswerResult,
    Progress,
    StockRequest,
    render_markdown,
)
from assistant.generation.steps import Steps
from assistant.grocy.client import GrocyError, StockItem
from assistant.indexing.chunks import Chunk
from assistant.logs import REQUEST_ID
from assistant.metrics import dependency_kind
from assistant.service import DEPENDENCY_ERRORS, QuestionAnswerer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Record:
    """Что известно о вопросе до ответа — для журнала."""

    chat_id: int
    request_id: str
    question: str
    # all — по всем рецептам; stock — из того, что есть; near — «поискать ещё».
    source: str
    stock_items: int | None
    # Время начала (часы сервиса) и «просили склад, а Grocy не ответил».
    started: float = 0.0
    stock_down: bool = False


@dataclass(frozen=True)
class Handoff:
    """Передать разговор человеку: повод и суть обращения для оператора."""

    reason: str
    summary: str = ""


@dataclass(frozen=True)
class Reply:
    """Что отправить пользователю: HTML и кнопки; answer_id — для оценки;
    handoff — после этого сообщения разговор переходит к человеку."""

    text: str
    buttons: Keyboard = field(default_factory=list)
    answer_id: int | None = None
    handoff: Handoff | None = None


ERROR_REPLY = Reply(ERROR_TEXT)
# Сбой после повторов — «что-то пошло не так» и кнопка «Позвать повара».
FAILURE_REPLY = Reply(ERROR_TEXT, human_buttons("failure"))
GONE_REPLY = Reply(GONE_TEXT)

# Рецепт по id из индекса; None — в индексе его нет.
RecipeLookup = Callable[[int], Chunk | None]
# Шаги по тексту рецепта: steps.make_steps с моделями бота.
StepsMaker = Callable[[str], Steps]
# Шаги страницы рецепта по адресу: recipe_page.fetch_page; None — не вышло.
PageFetcher = Callable[[str], RecipePage | None]
# Фото по адресу: recipe_page.fetch_photo; None — не вышло.
PhotoFetcher = Callable[[str], bytes | None]
# Что отправить: текст или фото с подписью.
Part = str | tuple[bytes, str]
# Что сейчас на складе: grocy.client.Grocy.stock.
StockReader = Callable[[], list[StockItem]]
# Кэш шагов сбрасывается, когда меняется любой из двух промптов.
STEPS_VERSION = f"{steps_module.PROMPT_VERSION}+{steps_module.REVIEW_PROMPT_VERSION}"


class BotService:
    def __init__(
        self,
        answerer: QuestionAnswerer,
        store: Store,
        pages: PageChecker,
        *,
        recipes: RecipeLookup,
        make_steps: StepsMaker | None,
        fetch_page: PageFetcher,
        fetch_photo: PhotoFetcher,
        gigachat_note: bool = False,
        operators: bool = False,
        stock: StockReader | None = None,
        journal: Journal | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """operators — есть ли группа операторов (ESCALATION_CHAT_ID): без неё
        бот честно говорит, что передать человеку сейчас не может. stock —
        чтение склада Grocy; None — Grocy не подключён (нет GROCY_API_KEY)."""
        self.answerer = answerer
        self.store = store
        self.pages = pages
        self.recipes = recipes
        self.make_steps = make_steps
        self.fetch_page = fetch_page
        self.fetch_photo = fetch_photo
        self.gigachat_note = gigachat_note
        self.operators = operators
        self.stock = stock
        # Журнал событий (journal.py); без ключа — случайный ключ процесса.
        self.journal = journal or Journal()
        self._clock = clock

    def ask(
        self, chat_id: int, question: str, progress: Progress | None = None
    ) -> Reply:
        """Ответ на вопрос с учётом истории диалога; при сбое история не меняется."""
        if wants_human(question):
            # Явная просьба — сразу человеку, без модели (решение 67).
            self.store.add_turn(chat_id, question, HANDOFF_TEXT)
            return self.handoff(chat_id, "request")
        if self.stock is not None and self.store.grocy_enabled(chat_id):
            source = self.store.source(chat_id)
            if source is None:
                # Новый разговор — сначала спросить, откуда искать (решение 71).
                self.store.save_pending(chat_id, question)
                return Reply(SOURCE_QUESTION_TEXT, source_buttons())
            return self._answer(chat_id, question, progress, source == "stock")
        return self._answer(chat_id, question, progress)

    def choose_source(
        self, chat_id: int, source: Source, progress: Progress | None = None
    ) -> Reply:
        """Кнопка «из того, что есть» / «по всем рецептам»: ответ на вопрос в ожидании."""
        self.store.set_source(chat_id, source)
        question = self.store.pop_pending(chat_id)
        if question is None:
            return Reply(ASK_AGAIN_TEXT)
        return self._answer(chat_id, question, progress, source == "stock")

    def near(
        self, chat_id: int, answer_id: int, progress: Progress | None = None
    ) -> Reply:
        """Расширенный поиск: рецепты, которым не хватает 1–4 продуктов со склада."""
        question = self.store.question_of(answer_id, chat_id)
        if question is None or self.stock is None:
            return Reply(GONE_TEXT)
        return self._answer(chat_id, question, progress, True, near=True)

    def start(self, chat_id: int) -> Reply:
        """/start: приветствие и, если есть Grocy, предложение подключить склад."""
        grocy = self.stock is not None
        connected = grocy and self.store.grocy_enabled(chat_id)
        return Reply(start_text(grocy, connected), start_buttons(grocy, connected))

    def connect_grocy(self, chat_id: int) -> str:
        """/grocy и кнопка «Подключить Grocy»: в MVP — общий склад на сервере."""
        if self.stock is None:
            return GROCY_UNAVAILABLE_TEXT
        if self.store.grocy_enabled(chat_id):
            return GROCY_ALREADY_ON_TEXT
        self.store.set_grocy(chat_id, True)
        self.journal.event("grocy", chat_id, enabled=True)
        return GROCY_ON_TEXT

    def disconnect_grocy(self, chat_id: int) -> str:
        """/grocy_off: дальше — поиск по всем рецептам, как без склада."""
        if self.stock is None or not self.store.grocy_enabled(chat_id):
            return GROCY_ALREADY_OFF_TEXT
        self.store.set_grocy(chat_id, False)
        self.journal.event("grocy", chat_id, enabled=False)
        return GROCY_OFF_TEXT

    def _stock_request(self, near: bool) -> StockRequest | None:
        """Склад из Grocy; None — Grocy недоступен (тогда ищем по всем рецептам)."""
        assert self.stock is not None
        try:
            items = self.stock()
        except GrocyError as error:
            logger.error("склад не прочитан: %s", error)
            return None
        return StockRequest(names=tuple(item.name for item in items), near=near)

    def _answer(
        self,
        chat_id: int,
        question: str,
        progress: Progress | None,
        from_stock: bool = False,
        *,
        near: bool = False,
    ) -> Reply:
        """Ответ под своей меткой request_id: её видят все записи лога по пути
        ответа и строка журнала (assistant/logs.py)."""
        request_id = uuid.uuid4().hex[:12]
        token = REQUEST_ID.set(request_id)
        try:
            return self._answer_as(
                request_id, chat_id, question, progress, from_stock, near=near
            )
        finally:
            REQUEST_ID.reset(token)

    def _answer_as(
        self,
        request_id: str,
        chat_id: int,
        question: str,
        progress: Progress | None,
        from_stock: bool = False,
        *,
        near: bool = False,
    ) -> Reply:
        limited = self._limited(chat_id)
        if limited is not None:
            return limited
        # Считается до вызова: сбой модели тоже тратит деньги.
        self.store.add_ask(chat_id, KEEP)
        started = self._clock()
        stock = self._stock_request(near) if from_stock else None
        note = STOCK_DOWN_TEXT if from_stock and stock is None else ""
        record = Record(
            chat_id,
            request_id,
            question,
            ("near" if near else "stock") if stock else "all",
            len(stock.names) if stock else None,
            started,
            from_stock and stock is None,
        )
        try:
            result = self.answerer.answer(
                question, self.store.history(chat_id), progress, stock
            )
        except DEPENDENCY_ERRORS as error:
            # Текст ошибки — код HTTP и ответ API, без вопроса пользователя;
            # секреты из лога вычищает форматтер (assistant/logs.py).
            logger.error("ассистент не ответил: %s: %s", type(error).__name__, error)
            self._journal(
                record,
                None,
                "error",
                error=type(error).__name__,
                error_kind=dependency_kind(error),
            )
            return FAILURE_REPLY if self.operators else ERROR_REPLY
        answer = result.answer
        if answer.escalation is not None:
            # Повод из разбора вопроса: искать и отвечать не нужно (решение 65).
            self._journal(
                record, result, "escalate", escalation=answer.escalation.reason
            )
            self.store.add_turn(chat_id, question, HANDOFF_TEXT)
            return self.handoff(
                chat_id, answer.escalation.reason, answer.escalation.summary
            )
        alive = self.pages.alive_many([link.url for link in answer.sources])
        seconds = self._clock() - started
        answer_id = self.store.save_answer(
            chat_id,
            status=answer.status,
            reason=answer.reason,
            model=answer.model,
            recipe_ids=[link.recipe_id for link in answer.sources],
            seconds=seconds,
            question=question,
            missing=answer.missing,
        )
        # Ответ в истории — текстом со ссылками: по ним ассистент находит
        # рецепты для уточнений («а во втором сколько муки?»).
        self.store.add_turn(chat_id, question, render_markdown(answer))
        self._journal(
            record,
            result,
            answer.status,
            answer_id=answer_id,
            reason=answer.reason,
            allergy=answer.allergy,
        )
        text = answer_html(answer, alive, gigachat_note=self.gigachat_note)
        if note:
            text = f"{note}\n\n{text}"
        buttons = answer_buttons(
            answer_id, answer, alive, ai_steps=self.make_steps is not None
        )
        if answer.near_available:
            # «Поискать ещё» — рецепты, где не хватает 1–4 продуктов (решение 72).
            text = f"{text}\n\n{NEAR_OFFER_TEXT}"
            buttons = [*buttons, *near_buttons(answer_id)]
        if self.operators and self.store.recent_statuses(chat_id) == [
            "not_found",
            "not_found",
        ]:
            # Второй «не нашёл» подряд — предложить человека, не передавать сразу.
            text = f"{text}\n\n{NOT_FOUND_TWICE_TEXT}"
            buttons = [*buttons, *human_buttons("not_found_twice")]
        return Reply(text, buttons, answer_id)

    def _limited(self, chat_id: int) -> Reply | None:
        """Лимит частоты (решение 75): ответ «подождите» или None."""
        hit = exceeded(self.store.ask_times(chat_id, KEEP), self.store.now(), LIMITS)
        if hit is None:
            return None
        self.journal.event(
            "limited",
            chat_id,
            window=hit.limit.window,
            wait=round(hit.wait),
        )
        limit = hit.limit
        return Reply(
            LIMIT_TEXT.format(
                count=limit.count, label=limit.label, wait=wait_text(hit.wait)
            )
        )

    def _journal(
        self,
        record: Record,
        result: AnswerResult | None,
        status: str,
        **fields: object,
    ) -> None:
        """Событие «ответ» в журнале: вопрос, разбор, найденное и показанное."""
        trace = result.trace if result else None
        self.journal.event(
            "answer",
            record.chat_id,
            request_id=record.request_id,
            question=question_text(record.question),
            source=record.source,
            stock_items=record.stock_items,
            search=search_fields(trace.args if trace else None),
            found=[hit.chunk.recipe_id for hit in trace.hits] if trace else [],
            shown=[link.recipe_id for link in result.answer.sources] if result else [],
            status=status,
            model=result.answer.model if result else None,
            prompt=result.answer.prompt_version if result else None,
            seconds=round(self._clock() - record.started, 1),
            retrieval_seconds=round(trace.retrieval_seconds, 1) if trace else None,
            generation_seconds=round(trace.generation_seconds, 1) if trace else None,
            stock_down=record.stock_down,
            **fields,
        )

    def reset(self, chat_id: int) -> str:
        self.store.reset(chat_id)
        return NEW_TEXT

    def vote(
        self, chat_id: int, answer_id: int, vote: Vote
    ) -> tuple[str, Reply | None]:
        """Подсказка на кнопке и, после 👎, вопрос «что не так?»."""
        if not self.store.vote(answer_id, chat_id, vote):
            return GONE_TEXT, None
        self.journal.event("vote", chat_id, answer_id=answer_id, vote=vote)
        if vote == "down":
            return THANKS_TEXT, Reply(WHAT_WRONG_TEXT, reason_buttons(answer_id))
        return THANKS_TEXT, None

    def vote_reason(self, chat_id: int, answer_id: int, reason: VoteReason) -> Reply:
        """Причина 👎; «не получилось» — предложить повара."""
        if not self.store.vote_reason(answer_id, chat_id, reason):
            return Reply(GONE_TEXT)
        self.journal.event("vote", chat_id, answer_id=answer_id, reason=reason)
        if reason == "failed" and self.operators:
            return Reply(FAILED_TEXT, human_buttons("feedback"))
        return Reply(THANKS_TEXT)

    # --- Передача человеку ---

    def handoff(self, chat_id: int, reason: str, summary: str = "") -> Reply:
        """Что сказать пользователю и кому передать; обращение откроет приложение."""
        if not self.operators:
            return Reply(NO_OPERATORS_TEXT)
        if self.store.open_ticket_of(chat_id) is not None:
            return Reply(ALREADY_WITH_HUMAN_TEXT)
        text = (
            HANDOFF_TEXT if reason != "health" else f"{HEALTH_TEXT}\n\n{HANDOFF_TEXT}"
        )
        return Reply(text, handoff=Handoff(reason, summary))

    def human(self, chat_id: int, reason: str) -> Reply:
        """/human и кнопка «Позвать повара»: повод — просьба или предложенный кодом."""
        if reason != "request" and reason not in OFFERED_REASONS:
            return Reply(GONE_TEXT)
        return self.handoff(chat_id, reason)

    def open_ticket(self, chat_id: int, handoff: Handoff) -> tuple[Ticket, str]:
        """Обращение в базе и карточка для операторов (повод, суть, реплики)."""
        ticket = self.store.open_ticket(chat_id, handoff.reason, handoff.summary)
        self.journal.event(
            "ticket_opened", chat_id, ticket=ticket.id, reason=handoff.reason
        )
        card = card_html(
            ticket.id, handoff.reason, handoff.summary, self.store.history(chat_id)
        )
        return ticket, card

    def cancel_ticket(self, ticket: Ticket) -> None:
        """Тему создать не вышло — обращения нет, пользователь остаётся с ботом."""
        self.store.close_ticket(ticket.id)
        logger.warning("обращение %s не открылось в группе операторов", ticket.id)

    def back_to_bot(self, chat_id: int) -> tuple[Reply, Ticket | None]:
        """/bot у пользователя: обращение закрывается, ассистент снова отвечает."""
        ticket = self.store.open_ticket_of(chat_id)
        if ticket is None or not self.store.close_ticket(ticket.id):
            return Reply(NOT_WITH_HUMAN_TEXT), None
        self.journal.event("ticket_closed", chat_id, ticket=ticket.id, by="user")
        return Reply(BACK_TO_BOT_TEXT, rate_buttons(ticket.id)), ticket

    def close_by_operator(self, thread_id: int) -> tuple[Reply, Ticket] | None:
        """/close в теме или тему закрыли кнопкой Telegram; None — обращения нет."""
        ticket = self.store.ticket_of_thread(thread_id)
        if ticket is None or not self.store.close_ticket(ticket.id):
            return None
        self.journal.event(
            "ticket_closed", ticket.chat_id, ticket=ticket.id, by="operator"
        )
        return Reply(CLOSED_BY_OPERATOR_TEXT, rate_buttons(ticket.id)), ticket

    def expire_idle(self, idle: float = TICKET_IDLE) -> list[tuple[Reply, Ticket]]:
        """Закрыть обращения без сообщений дольше idle: что сказать пользователю."""
        expired = []
        for ticket in self.store.idle_tickets(idle):
            if self.store.close_ticket(ticket.id):
                self.journal.event(
                    "ticket_closed", ticket.chat_id, ticket=ticket.id, by="idle"
                )
                expired.append((Reply(EXPIRED_TEXT, rate_buttons(ticket.id)), ticket))
        return expired

    def rate(self, chat_id: int, ticket_id: int, vote: Vote) -> str:
        if not self.store.vote_ticket(ticket_id, chat_id, vote):
            return GONE_TEXT
        self.journal.event("ticket_vote", chat_id, ticket=ticket_id, vote=vote)
        return THANKS_TEXT

    def steps(self, chat_id: int, answer_id: int, recipe_id: int) -> Reply:
        """Шаги для рецепта из ответа этого чата: из кэша или Haiku + Sonnet.

        Шаги одного рецепта одинаковы для всех, поэтому кэш общий; «не смог
        расписать» тоже кэшируется. Сбой ответа модели — нет: попробуем снова.
        """
        ids = self.store.recipe_ids(answer_id, chat_id)
        if ids is None or recipe_id not in ids:
            return GONE_REPLY
        try:
            recipe = self.recipes(recipe_id)
            if recipe is None:
                return GONE_REPLY
            saved = self.store.steps(recipe_id, STEPS_VERSION)
            if saved is None and self.make_steps is None:
                # Без проверки шаги от ИИ не пишутся (кнопка — от прошлых ответов);
                # проверенные раньше — из кэша, как обычно.
                return Reply(steps_refused_html(recipe.name))
            if saved is None:
                saved = self._new_steps(recipe)
        except DEPENDENCY_ERRORS as error:
            logger.error("шаги не написаны: %s: %s", type(error).__name__, error)
            return ERROR_REPLY
        if saved is None:
            return ERROR_REPLY
        if saved.approved:
            return Reply(steps_html(recipe.name, json.loads(saved.text)))
        return Reply(steps_refused_html(recipe.name))

    def page_steps(self, chat_id: int, answer_id: int, recipe_id: int) -> list[Part]:
        """Шаги со страницы рецепта по порядку: заголовок с источником, затем
        по сообщению на шаг — фото с текстом шага в подписи (решение автора:
        фото и шаг — одно сообщение); шаг без фото — текстом.

        Фото скачивается здесь, параллельно; не скачалось — шаг идёт текстом.
        Страница не разобралась — заголовок со ссылкой, шагов нет.
        """
        ids = self.store.recipe_ids(answer_id, chat_id)
        if ids is None or recipe_id not in ids:
            return [GONE_TEXT]
        try:
            recipe = self.recipes(recipe_id)
        except DEPENDENCY_ERRORS as error:
            logger.error("рецепт не найден: %s: %s", type(error).__name__, error)
            return [ERROR_TEXT]
        if recipe is None:
            return [GONE_TEXT]
        page = self.fetch_page(recipe.url)
        steps = page.steps if page else []
        urls = [s.photo for s in steps if s.photo]
        with ThreadPoolExecutor(max_workers=max(1, min(len(urls), 10))) as pool:
            photos = dict(zip(urls, pool.map(self.fetch_photo, urls), strict=True))
        missing = self.store.missing_of(answer_id, chat_id, recipe_id)
        parts: list[Part] = [
            page_steps_header(recipe.name, recipe.url, len(steps), missing)
        ]
        for number, step in enumerate(steps, start=1):
            caption = step_caption(number, step.text)
            data = photos.get(step.photo) if step.photo else None
            parts.append((data, caption) if data is not None else caption)
        logger.info("шаги со страницы рецепта %s: %s", recipe.recipe_id, len(steps))
        return parts

    def _new_steps(self, recipe: Chunk) -> SavedSteps | None:
        assert self.make_steps is not None
        result = self.make_steps(recipe.text)
        logger.info(
            "шаги рецепта %s: %s за %.1f с",
            recipe.recipe_id,
            result.status,
            result.seconds,
        )
        if result.status == "failed":
            return None
        saved = SavedSteps(
            json.dumps(result.steps, ensure_ascii=False),
            result.status == "approved",
            STEPS_VERSION,
        )
        self.store.save_steps(
            recipe.recipe_id, saved.text, saved.approved, saved.prompt_version
        )
        return saved
