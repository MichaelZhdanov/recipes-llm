"""Здоровье бота для Docker: `python -m assistant.bot.health`, код 0 — здоров.

HTTP-порта у бота нет (long polling), поэтому проверка — по файлу-
сердцебиению: раз в HEARTBEAT_EVERY бот спрашивает Telegram getMe и при
успехе обновляет файл (`app.heartbeat`). Файл старше MAX_AGE — значит,
бот завис или до Telegram не достучаться: контейнер помечается unhealthy.
"""

import sys
import time
from pathlib import Path

HEARTBEAT = Path("/tmp/recipes-bot-heartbeat")
HEARTBEAT_EVERY = 60.0
# Три пропущенных удара подряд.
MAX_AGE = 3 * HEARTBEAT_EVERY


def fresh(
    path: Path = HEARTBEAT, max_age: float = MAX_AGE, now: float | None = None
) -> bool:
    try:
        beat = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return (time.time() if now is None else now) - beat <= max_age


def main() -> int:
    return 0 if fresh() else 1


if __name__ == "__main__":
    sys.exit(main())
