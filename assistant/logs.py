"""Логи бота и HTTP API: секреты вычищаются из каждой записи.

Токен бота входит в адрес каждого запроса к Bot API
(`https://api.telegram.org/bot<токен>/…`): текст сетевой ошибки или
трассировка может его содержать. Форматтер заменяет значения всех секретов
из настроек на `***` во всей записи, вместе с трассировкой. Это последняя
линия: сам код бота секреты в лог не пишет.

Формат (этап 8, F8) — одна JSON-строка на запись: время, уровень, логгер,
текст и `request_id`, если запись сделана по пути ответа. `request_id`
живёт в `REQUEST_ID` (contextvars): бот ставит его на время ответа
(`BotService._answer`), API — на время HTTP-запроса. `asyncio.to_thread` и
пул потоков FastAPI копируют контекст, поэтому метку видят все записи по
пути ответа — повторы HTTP, ошибки адаптеров — и та же строка журнала.
"""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

from pydantic import SecretStr

from assistant.config import Settings

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Логгер журнала событий бота (bot/journal.py): свой обработчик без префиксов.
JOURNAL = "journal"
MASK = "***"
# Метка запроса для всех записей по пути одного ответа; None — вне ответа.
REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)


def secrets_of(settings: Settings) -> list[str]:
    """Непустые значения всех SecretStr настроек."""
    values = (getattr(settings, name) for name in type(settings).model_fields)
    return [
        value.get_secret_value()
        for value in values
        if isinstance(value, SecretStr) and value.get_secret_value()
    ]


class RedactingFormatter(logging.Formatter):
    def __init__(self, secrets: list[str], fmt: str = FORMAT) -> None:
        super().__init__(fmt)
        # Длинные — первыми: секрет может содержать другой как подстроку.
        self._secrets = sorted(set(secrets), key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, MASK)
        return text

    def format(self, record: logging.LogRecord) -> str:
        return self.redact(super().format(record))


class JsonFormatter(RedactingFormatter):
    """Одна JSON-строка на запись. Секреты вычищаются до сериализации: в JSON
    символы экранируются, и после неё значение могло бы не совпасть."""

    def format(self, record: logging.LogRecord) -> str:
        created = datetime.fromtimestamp(record.created, UTC)
        data: dict[str, object] = {
            "time": created.isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": self.redact(record.getMessage()),
        }
        request_id = REQUEST_ID.get()
        if request_id is not None:
            data["request_id"] = request_id
        if record.exc_info:
            data["exception"] = self.redact(self.formatException(record.exc_info))
        return json.dumps(data, ensure_ascii=False)


def setup_logging(settings: Settings) -> None:
    secrets = secrets_of(settings)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(secrets))
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
    # Uvicorn ставит своим логгерам свои обработчики (текст): тот же JSON.
    for name in ("uvicorn", "uvicorn.access"):
        logging.getLogger(name).handlers = [handler]
    # Журнал (journal.py) — чистые JSON-строки, без времени и уровня в начале,
    # и всегда, даже при LOG_LEVEL=WARNING; секреты вычищаются так же.
    journal = logging.StreamHandler()
    journal.setFormatter(RedactingFormatter(secrets, fmt="%(message)s"))
    journal_logger = logging.getLogger(JOURNAL)
    journal_logger.handlers = [journal]
    journal_logger.setLevel(logging.INFO)
    journal_logger.propagate = False
    # httpx2 пишет на INFO каждый запрос с адресом — лишний шум в логе.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
