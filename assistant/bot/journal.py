"""Журнал событий бота: одна JSON-строка на событие (этап 7, F7; решение 76).

Кто спросил, во что модель разобрала вопрос, что нашлось и что показали,
передача человеку, оценки, срабатывание лимита. Без секретов и без
немаскированных ПДн:
- пользователь — псевдоним: HMAC-SHA256 от chat_id на секрете JOURNAL_KEY,
  первые 12 hex-символов. Один человек — один и тот же псевдоним, по нему
  видно «этот же пользователь снова», но не «кто это»: chat_id из псевдонима
  без ключа не перебрать. Простой хэш без ключа перебирался бы — chat_id
  всего около 10 цифр;
- вопрос — после mask_pii и не длиннее QUESTION_CHARS;
- значения секретов вычищает тот же форматтер, что у обычного лога
  (`assistant/logs.py`): последняя линия, сам журнал секретов не пишет.

Строки уходят в stdout логгером `journal` без префиксов, их собирает Docker
(ротация — `x-logging` в docker-compose.yml). Обычный лог остаётся для
предупреждений и ошибок.
"""

import hashlib
import hmac
import json
import logging
import secrets
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import SecretStr

from assistant.logs import JOURNAL as LOGGER
from assistant.security.pii import mask_pii
from assistant.tools.search_recipes import SearchArgs

QUESTION_CHARS = 300
PSEUDONYM_CHARS = 12

logger = logging.getLogger(LOGGER)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Journal:
    def __init__(
        self,
        key: SecretStr | None = None,
        *,
        write: Callable[[str], None] | None = None,
        now: Callable[[], str] = _now,
    ) -> None:
        """key пуст — случайный ключ процесса: псевдонимы безопасны, но после
        перезапуска у того же человека будет другой."""
        value = key.get_secret_value() if key is not None else ""
        self._key = value.encode() or secrets.token_bytes(32)
        self._write = write or logger.info
        self._now = now

    def user(self, chat_id: int) -> str:
        digest = hmac.new(self._key, str(chat_id).encode(), hashlib.sha256)
        return digest.hexdigest()[:PSEUDONYM_CHARS]

    def event(self, name: str, chat_id: int, **fields: object) -> None:
        record = {"time": self._now(), "event": name, "user": self.user(chat_id)}
        self._write(json.dumps({**record, **fields}, ensure_ascii=False))


def question_text(question: str) -> str:
    """Вопрос для журнала: без ПДн и не длиннее QUESTION_CHARS."""
    return mask_pii(question).text[:QUESTION_CHARS]


def search_fields(args: SearchArgs | None) -> dict[str, object] | None:
    """Во что модель разобрала вопрос: запросы и условия поиска."""
    if args is None:
        return None
    return {
        # Запросы модель пишет по уже маскированному вопросу; маскер — ещё раз,
        # на случай если модель что-то восстановила.
        "queries": [question_text(query) for query in args.all_queries],
        "include": [list(group) for group in args.conditions.include],
        "exclude": list(args.conditions.exclude),
        "other": args.other,
        "dropped": list(args.dropped),
    }
