"""Журнал событий бота (этап 7, F7): кто спросил — псевдоним, вопрос — без
ПДн, разбор, найденное и показанное — одной JSON-строкой; секретов нет."""

import json
import logging
from collections.abc import Sequence
from typing import Any

import httpx2
import pytest
from chunk_helpers import CHICKEN, COCKTAIL
from pydantic import SecretStr

from assistant.bot.journal import QUESTION_CHARS, Journal
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
    StockRequest,
    Trace,
)
from assistant.generation.gigachat import LlmError, Message
from assistant.logs import JOURNAL, MASK, REQUEST_ID, setup_logging
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.store import Hit
from assistant.tools.search_recipes import SearchArgs

CHAT = 7
KEY = SecretStr("journal-test-key")
PHONE = "+7 912 345-67-89"
EMAIL = "olga.smirnova@example.com"


class Answerer:
    """Нашёл два рецепта, показал один; разбор — «курица без лука»."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        if self.error is not None:
            raise self.error
        args = SearchArgs(
            "курица",
            IngredientFilter(include=(("курица",),), exclude=("лук репчатый",)),
            queries=("курица с рисом",),
        )
        answer = Answer(
            status="answer",
            reason=None,
            message="🍗 Вот что нашлось:",
            statements=[AnswerStatement(text="Курица с рисом", sources=[1])],
            sources=[
                SourceLink(
                    number=1,
                    recipe_id=CHICKEN.recipe_id,
                    name="Курица",
                    url=CHICKEN.url,
                )
            ],
            model="zai-org/GLM-5.1",
            prompt_version="recipes_v5",
            dataset_revision="ffaf49ee",
        )
        hits = [Hit(CHICKEN, 0.9), Hit(COCKTAIL, 0.4)]
        trace = Trace(hits, [], None, [], 0.0, 0.0, args=args)
        return AnswerResult(answer=answer, trace=trace)


def service(answerer: Answerer, lines: list[str]) -> BotService:
    store = Store(":memory:")
    http = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(200)))
    return BotService(
        answerer,
        store,
        PageChecker(http, store),
        recipes=lambda recipe_id: None,
        make_steps=None,
        fetch_page=lambda url: None,
        fetch_photo=lambda url: None,
        journal=Journal(
            KEY, write=lines.append, now=lambda: "2026-11-07T20:00:00+00:00"
        ),
    )


def records(lines: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in lines]


def test_pseudonym_is_stable_per_key_and_does_not_reveal_the_chat() -> None:
    journal = Journal(KEY)

    assert journal.user(CHAT) == Journal(KEY).user(CHAT)
    assert journal.user(CHAT) != journal.user(CHAT + 1)
    # Другой ключ — другие псевдонимы: без ключа chat_id не перебрать.
    assert journal.user(CHAT) != Journal(SecretStr("other")).user(CHAT)
    assert len(journal.user(CHAT)) == 12 and str(CHAT) not in journal.user(CHAT)
    # Без ключа — случайный ключ процесса, а не постоянный общий.
    assert Journal().user(CHAT) != Journal().user(CHAT)


def test_answer_event_has_parse_found_and_shown_but_no_personal_data() -> None:
    lines: list[str] = []
    question = f"курица без лука, мой телефон {PHONE}, почта {EMAIL}"

    reply = service(Answerer(), lines).ask(CHAT, question)

    (line,) = lines
    assert PHONE not in line and "345-67-89" not in line and EMAIL not in line
    (record,) = records(lines)
    assert record["event"] == "answer" and record["status"] == "answer"
    assert record["user"] == Journal(KEY).user(CHAT)
    assert record["time"] == "2026-11-07T20:00:00+00:00"
    assert record["answer_id"] == reply.answer_id
    assert record["question"].startswith("курица без лука, мой телефон ")
    assert record["source"] == "all" and record["stock_items"] is None
    assert record["search"] == {
        "queries": ["курица с рисом"],
        "include": [["курица"]],
        "exclude": ["лук репчатый"],
        "other": False,
        "dropped": [],
    }
    assert record["found"] == [CHICKEN.recipe_id, COCKTAIL.recipe_id]
    assert record["shown"] == [CHICKEN.recipe_id]
    assert record["model"] == "zai-org/GLM-5.1" and record["prompt"] == "recipes_v5"
    assert len(record["request_id"]) == 12


class LoggingAnswerer(Answerer):
    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        logging.getLogger("assistant.http").warning("повтор запроса к API")
        return super().answer(question, history, progress, stock)


def test_log_lines_on_the_way_carry_the_journal_request_id() -> None:
    """По request_id из журнала находятся все записи лога этого ответа."""
    lines: list[str] = []
    seen: list[str | None] = []

    class Seen(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(REQUEST_ID.get())

    handler = Seen()
    logging.getLogger("assistant.http").addHandler(handler)
    try:
        service(LoggingAnswerer(), lines).ask(CHAT, "курица")
    finally:
        logging.getLogger("assistant.http").removeHandler(handler)

    assert seen == [records(lines)[0]["request_id"]]
    assert REQUEST_ID.get() is None  # после ответа метка снята


def test_long_question_is_cut() -> None:
    lines: list[str] = []

    service(Answerer(), lines).ask(CHAT, "курица " * 200)

    assert len(records(lines)[0]["question"]) == QUESTION_CHARS


def test_failure_is_journaled_with_the_error_type_only() -> None:
    lines: list[str] = []

    service(Answerer(LlmError("cloud.ru ответил HTTP 500")), lines).ask(CHAT, "блины")

    (record,) = records(lines)
    assert record["status"] == "error" and record["error"] == "LlmError"
    assert record["found"] == [] and record["search"] is None


def test_votes_and_grocy_switch_are_journaled() -> None:
    lines: list[str] = []
    bot = service(Answerer(), lines)
    answer_id = bot.ask(CHAT, "курица").answer_id
    assert answer_id is not None

    bot.vote(CHAT, answer_id, "down")
    bot.vote_reason(CHAT, answer_id, "wrong")

    vote, reason = records(lines)[1:]
    assert vote["event"] == "vote" and vote["vote"] == "down"
    assert reason["reason"] == "wrong" and reason["answer_id"] == answer_id


def test_ticket_open_and_close_are_journaled() -> None:
    lines: list[str] = []
    bot = service(Answerer(), lines)
    bot.operators = True

    handoff = bot.human(CHAT, "request").handoff
    assert handoff is not None
    ticket, _ = bot.open_ticket(CHAT, handoff)
    bot.back_to_bot(CHAT)

    opened, closed = records(lines)
    assert opened == {
        "time": "2026-11-07T20:00:00+00:00",
        "event": "ticket_opened",
        "user": Journal(KEY).user(CHAT),
        "ticket": ticket.id,
        "reason": "request",
    }
    assert closed["event"] == "ticket_closed" and closed["by"] == "user"


def test_journal_lines_are_plain_json_with_secrets_masked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = "42:TEST-secret-token"
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    try:
        setup_logging(
            Settings(telegram_bot_token=SecretStr(token), log_level="WARNING")
        )
        # Даже при LOG_LEVEL=WARNING журнал пишется; секрет — последней линией.
        Journal(KEY).event("answer", CHAT, question=f"мой токен {token}")
        err = capsys.readouterr().err
    finally:
        root.handlers, root.level = saved
        logging.getLogger(JOURNAL).handlers = []
        logging.getLogger(JOURNAL).propagate = True

    (line,) = err.splitlines()
    assert token not in line
    assert json.loads(line)["question"] == f"мой токен {MASK}"
