"""Запуск бота: `python -m assistant.bot` (сервис `bot` в docker-compose)."""

import asyncio

from assistant.bot.app import run
from assistant.config import Settings
from assistant.logs import setup_logging


def main() -> None:
    settings = Settings()
    setup_logging(settings)
    asyncio.run(run(settings))


if __name__ == "__main__":
    main()
