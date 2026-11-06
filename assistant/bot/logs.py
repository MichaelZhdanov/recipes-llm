"""Логи бота: секреты вычищаются из каждой записи.

Токен бота входит в адрес каждого запроса к Bot API
(`https://api.telegram.org/bot<токен>/…`): текст сетевой ошибки или
трассировка может его содержать. Форматтер заменяет значения всех секретов
из настроек на `***` во всей записи, вместе с трассировкой. Это последняя
линия: сам код бота секреты в лог не пишет.
"""

import logging

from pydantic import SecretStr

from assistant.config import Settings

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MASK = "***"


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

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, MASK)
        return text


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter(secrets_of(settings)))
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
    # httpx2 пишет на INFO каждый запрос с адресом — лишний шум в логе.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
