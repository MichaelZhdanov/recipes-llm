"""JSON-логи (этап 8, F8): одна строка на запись, request_id по пути ответа,
секреты вычищены — и у бота, и у HTTP API."""

import json
import logging
import sys
from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from assistant.api import create_app
from assistant.config import Settings
from assistant.generation.answer import AnswerResult, Progress, StockRequest
from assistant.generation.gigachat import LlmError, Message
from assistant.logs import MASK, REQUEST_ID, JsonFormatter, setup_logging

TOKEN = "42:TEST-secret-token"


class Lines(logging.Handler):
    """Записи, отформатированные как в проде."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter([TOKEN]))
        self.lines: list[dict[str, object]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def lines() -> Iterator[Lines]:
    handler = Lines()
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield handler
    root.removeHandler(handler)
    root.setLevel(level)


def record(message: str, *args: object, exc: bool = False) -> logging.LogRecord:
    exc_info = None
    if exc:
        try:
            raise ConnectionError(f"https://api.telegram.org/bot{TOKEN}/getMe")
        except ConnectionError:
            exc_info = sys.exc_info()
    return logging.LogRecord(
        "assistant.x", logging.ERROR, __file__, 1, message, args, exc_info
    )


def test_line_is_json_with_level_logger_and_masked_secrets() -> None:
    formatter = JsonFormatter([TOKEN])

    data = json.loads(formatter.format(record("токен %s", TOKEN, exc=True)))

    assert data["level"] == "ERROR" and data["logger"] == "assistant.x"
    assert data["message"] == f"токен {MASK}"
    assert TOKEN not in data["exception"] and f"bot{MASK}/getMe" in data["exception"]
    assert data["time"].endswith("+00:00") and "request_id" not in data


def test_request_id_is_added_only_inside_an_answer() -> None:
    formatter = JsonFormatter([])
    token = REQUEST_ID.set("abc123")
    try:
        inside = json.loads(formatter.format(record("внутри")))
    finally:
        REQUEST_ID.reset(token)
    outside = json.loads(formatter.format(record("снаружи")))

    assert inside["request_id"] == "abc123" and "request_id" not in outside


def test_setup_gives_root_and_uvicorn_the_json_formatter() -> None:
    root = logging.getLogger()
    saved = root.handlers[:], root.level
    uvicorn = {
        n: logging.getLogger(n).handlers[:] for n in ("uvicorn", "uvicorn.access")
    }
    try:
        setup_logging(Settings(telegram_bot_token=SecretStr(TOKEN)))
        (handler,) = root.handlers
        assert isinstance(handler.formatter, JsonFormatter)
        for name in uvicorn:
            assert logging.getLogger(name).handlers == [handler]
    finally:
        root.handlers, root.level = saved
        for name, handlers in uvicorn.items():
            logging.getLogger(name).handlers = handlers
        logging.getLogger("journal").handlers = []
        logging.getLogger("journal").propagate = True


class LoggingAnswerer:
    """Пишет в лог по пути ответа, как адаптеры при повторе запроса."""

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        logging.getLogger("assistant.http").warning("повтор запроса к API")
        raise LlmError("cloud.ru ответил HTTP 500")


def test_api_request_id_is_in_the_header_and_in_the_log(lines: Lines) -> None:
    client = TestClient(create_app(Settings(), answerer=LoggingAnswerer()))

    response = client.post("/ask", json={"question": "блины"})

    assert response.status_code == 503
    (logged,) = [line for line in lines.lines if line["logger"] == "assistant.http"]
    assert logged["request_id"] == response.headers["X-Request-ID"]
    assert len(response.headers["X-Request-ID"]) == 12
