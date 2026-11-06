"""Логика бота без aiogram: что ответить на вопрос и на /new.

Синхронная, как и весь ассистент: обработчики aiogram (`app.py`) вызывают её
через `asyncio.to_thread`. Здесь всё, что стоит проверить тестами: история
диалога, ссылки на живые страницы, сохранение ответа, ошибки.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from assistant.bot.format import (
    ERROR_TEXT,
    NEW_TEXT,
    Keyboard,
    answer_buttons,
    answer_html,
)
from assistant.bot.pages import PageChecker
from assistant.bot.store import Store
from assistant.generation.answer import Progress, render_markdown
from assistant.service import DEPENDENCY_ERRORS, QuestionAnswerer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reply:
    """Что отправить пользователю: HTML и кнопки; answer_id — для оценки."""

    text: str
    buttons: Keyboard = field(default_factory=list)
    answer_id: int | None = None


ERROR_REPLY = Reply(ERROR_TEXT)


class BotService:
    def __init__(
        self,
        answerer: QuestionAnswerer,
        store: Store,
        pages: PageChecker,
        *,
        gigachat_note: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.answerer = answerer
        self.store = store
        self.pages = pages
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
            logger.error("ассистент не ответил: %s", type(error).__name__)
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
