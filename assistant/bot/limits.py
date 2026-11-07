"""Ограничение частоты на пользователя (этап 7, F7; решение 75).

Каждый ответ — вызовы модели и ≈ 2–3 ₽. Без лимита один человек (или
скрипт) может за минуты потратить весь баланс и занять очередь остальным.
Считаются только действия, которые зовут модель: вопрос, выбор источника,
«🔎 Поискать ещё». Команды, оценки, кнопки шагов со страницы и переписка с
оператором — бесплатны и не считаются.

Окно скользящее: время каждого обращения лежит в SQLite (`Store.add_ask`),
поэтому лимит переживает перезапуск бота.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Limit:
    count: int
    window: float
    # «за 10 минут», «в сутки» — для ответа пользователю.
    label: str


LIMITS = (Limit(10, 10 * 60, "за 10 минут"), Limit(50, 24 * 60 * 60, "в сутки"))
# Старше самого длинного окна обращения не нужны.
KEEP = max(limit.window for limit in LIMITS)


@dataclass(frozen=True)
class Exceeded:
    limit: Limit
    # Через сколько секунд можно спросить снова.
    wait: float


def exceeded(
    times: Sequence[float], now: float, limits: Sequence[Limit] = LIMITS
) -> Exceeded | None:
    """Какой лимит исчерпан и сколько ждать; None — можно спрашивать.

    times — моменты прошлых обращений. Если исчерпаны оба лимита, важен тот,
    после которого ждать дольше."""
    found = []
    for limit in limits:
        recent = sorted(t for t in times if t > now - limit.window)
        if len(recent) >= limit.count:
            # Освободится место, когда из окна выйдет самое старое из последних count.
            freed = recent[-limit.count] + limit.window
            found.append(Exceeded(limit, max(freed - now, 0.0)))
    return max(found, key=lambda e: e.wait, default=None)


def wait_text(seconds: float) -> str:
    """«через 4 мин», «через 3 ч»: с округлением вверх, чтобы не обмануть."""
    minutes = max(1, math.ceil(seconds / 60))
    if minutes < 60:
        return f"через {minutes} мин"
    return f"через {math.ceil(minutes / 60)} ч"
