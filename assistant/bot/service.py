"""Логика бота без aiogram: что ответить на вопрос, /new, оценку и кнопку шагов.

Синхронная, как и весь ассистент: обработчики aiogram (`app.py`) вызывают её
через `asyncio.to_thread`. Здесь всё, что стоит проверить тестами: история
диалога, ссылки на живые страницы, сохранение ответа и оценки, ошибки.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from assistant.bot.format import (
    ERROR_TEXT,
    GONE_TEXT,
    NEW_TEXT,
    THANKS_TEXT,
    WHAT_WRONG_TEXT,
    Keyboard,
    answer_buttons,
    answer_html,
    reason_buttons,
    steps_html,
    steps_refused_html,
)
from assistant.bot.pages import PageChecker
from assistant.bot.store import SavedSteps, Store, Vote, VoteReason
from assistant.generation import steps as steps_module
from assistant.generation.answer import Progress, render_markdown
from assistant.generation.steps import Steps
from assistant.indexing.chunks import Chunk
from assistant.service import DEPENDENCY_ERRORS, QuestionAnswerer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reply:
    """Что отправить пользователю: HTML и кнопки; answer_id — для оценки."""

    text: str
    buttons: Keyboard = field(default_factory=list)
    answer_id: int | None = None


ERROR_REPLY = Reply(ERROR_TEXT)
GONE_REPLY = Reply(GONE_TEXT)

# Рецепт по id из индекса; None — в индексе его нет.
RecipeLookup = Callable[[int], Chunk | None]
# Шаги по тексту рецепта: steps.make_steps с моделями бота.
StepsMaker = Callable[[str], Steps]
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
        make_steps: StepsMaker,
        gigachat_note: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.answerer = answerer
        self.store = store
        self.pages = pages
        self.recipes = recipes
        self.make_steps = make_steps
        self.gigachat_note = gigachat_note
        self._clock = clock

    def ask(
        self, chat_id: int, question: str, progress: Progress | None = None
    ) -> Reply:
        """Ответ на вопрос с учётом истории диалога; при сбое история не меняется."""
        started = self._clock()
        try:
            result = self.answerer.answer(
                question, self.store.history(chat_id), progress
            )
        except DEPENDENCY_ERRORS as error:
            # Текст ошибки — код HTTP и ответ API, без вопроса пользователя;
            # секреты из лога вычищает форматтер (logs.py).
            logger.error("ассистент не ответил: %s: %s", type(error).__name__, error)
            return ERROR_REPLY
        answer = result.answer
        alive = self.pages.alive_many([link.url for link in answer.sources])
        seconds = self._clock() - started
        answer_id = self.store.save_answer(
            chat_id,
            status=answer.status,
            reason=answer.reason,
            model=answer.model,
            recipe_ids=[link.recipe_id for link in answer.sources],
            seconds=seconds,
        )
        # Ответ в истории — текстом со ссылками: по ним ассистент находит
        # рецепты для уточнений («а во втором сколько муки?»).
        self.store.add_turn(chat_id, question, render_markdown(answer))
        logger.info(
            "ответ %s: %s%s за %.1f с",
            answer_id,
            answer.status,
            f" ({answer.reason})" if answer.reason else "",
            seconds,
        )
        return Reply(
            answer_html(answer, alive, gigachat_note=self.gigachat_note),
            answer_buttons(answer_id, answer, alive),
            answer_id,
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
        logger.info("оценка ответа %s: %s", answer_id, vote)
        if vote == "down":
            return THANKS_TEXT, Reply(WHAT_WRONG_TEXT, reason_buttons(answer_id))
        return THANKS_TEXT, None

    def vote_reason(self, chat_id: int, answer_id: int, reason: VoteReason) -> str:
        """Причина 👎; повод передать человеку — на этапе 5."""
        if not self.store.vote_reason(answer_id, chat_id, reason):
            return GONE_TEXT
        logger.info("причина 👎 ответа %s: %s", answer_id, reason)
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

    def _new_steps(self, recipe: Chunk) -> SavedSteps | None:
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
