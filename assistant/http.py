"""Повторы HTTP-запросов к внешним сервисам (cloud.ru, Voyage, GigaChat, позже Telegram и Grocy).

Повторяются сетевые ошибки и ответы «попробуйте позже»: 429 и 5xx. Пауза
берётся из заголовка Retry-After, если сервер его прислал, иначе растёт
экспоненциально. Остальные коды (400, 401, 404) не повторяются: повтор
их не исправит.
"""

import time
from collections.abc import Callable

import httpx2

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _backoff(attempt: int, base_delay: float, max_delay: float) -> float:
    return float(min(base_delay * 2 ** (attempt - 1), max_delay))


def _retry_after(response: httpx2.Response, max_delay: float) -> float | None:
    value = response.headers.get("retry-after", "")
    try:
        return min(float(value), max_delay)
    except ValueError:
        # Retry-After бывает и датой; для простоты тогда используем свою паузу.
        return None


def request_with_retries(
    send: Callable[[], httpx2.Response],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx2.Response:
    """Вызывает send() до attempts раз. Возвращает последний ответ, статус не проверяет."""
    for attempt in range(1, attempts + 1):
        is_last = attempt == attempts
        try:
            response = send()
        except httpx2.TransportError:
            if is_last:
                raise
            sleep(_backoff(attempt, base_delay, max_delay))
            continue
        if response.status_code not in RETRY_STATUSES or is_last:
            return response
        delay = _retry_after(response, max_delay)
        sleep(delay if delay is not None else _backoff(attempt, base_delay, max_delay))
    raise AssertionError("цикл всегда возвращает ответ или пробрасывает ошибку")
