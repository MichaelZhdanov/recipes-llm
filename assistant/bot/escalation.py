"""Передача разговора человеку (этап 5): поводы, которые ловит код, и карточка.

Поводы приходят двумя путями (решения 65 и 67 в PLAN.md):
- код бота — явная просьба (`/human` или фраза из короткого списка ниже),
  сбой ассистента, второй «не нашёл» подряд, 👎 «не получилось»: для них
  бот показывает кнопку «Позвать повара», кроме явной просьбы — она сразу;
- модель — в том же вызове, что разбирает вопрос в поиск (поле escalation
  в search_recipes): не получилось, непонятен рецепт, здоровье, ошибка в
  данных, просьба другими словами.

Здесь — чистые функции без Telegram: распознать просьбу, подписать повод,
собрать имя темы и карточку для операторов. Карточка — без username и id
пользователя; тексты в ней маскируются ещё раз (`mask_pii`): карточка
остаётся в истории группы операторов.
"""

import html
import re
from collections.abc import Sequence

from assistant.generation.gigachat import Message
from assistant.security.pii import mask_pii

# Повод → подпись для операторов. health — 🔴: такие обращения смотрят первыми.
REASON_LABELS = {
    "request": "просит человека",
    "cooking_failed": "не получилось",
    "unclear_recipe": "непонятен рецепт",
    "health": "🔴 здоровье",
    "data_issue": "ошибка в данных",
    "not_found_twice": "дважды не нашёл",
    "failure": "сбой ассистента",
    "feedback": "👎 не получилось",
}
# Поводы, которые бот предлагает кнопкой (а не передаёт сразу).
OFFERED_REASONS = frozenset({"not_found_twice", "failure", "feedback"})
# Обращение без сообщений в любую сторону столько секунд закрывается само
# (решение автора 2026-11-07): брошенная тема не держит пользователя вдали от
# ассистента.
TICKET_IDLE = 12 * 60 * 60
# Имя темы в Telegram — 1–128 символов (Bot API, createForumTopic).
MAX_TOPIC_NAME = 128
# Сколько последних реплик диалога видит оператор в карточке.
CARD_TURNS = 6
# Длинная реплика в карточке обрезается: карточка — сводка, а не протокол.
# У ответов бота лимит больше: в них названия рецептов и ссылки. Шесть реплик
# с запасом помещаются в сообщение Telegram (4096 символов).
CARD_USER_CHARS = 400
CARD_BOT_CHARS = 700

# Явная просьба позвать человека — короткий список фраз (решение 67);
# остальные формулировки распознаёт модель (повод request).
_REQUEST = re.compile(
    r"\b(оператор\w*|жив\w* (человек\w*|повар\w*)|"
    r"позов\w* (человек\w*|повар\w*|кого-нибудь))\b",
    re.IGNORECASE,
)


def wants_human(text: str) -> bool:
    """Явная просьба позвать человека: «позовите человека», «оператор», «живой повар»."""
    return _REQUEST.search(text.replace("ё", "е").replace("Ё", "Е")) is not None


def reason_label(reason: str) -> str:
    return REASON_LABELS.get(reason, reason)


def topic_name(ticket_id: int, reason: str) -> str:
    """«#N · повод» — тема обращения в группе операторов."""
    return f"#{ticket_id} · {reason_label(reason)}"[:MAX_TOPIC_NAME]


def _clip(text: str, limit: int) -> str:
    text = mask_pii(text).text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def card_html(
    ticket_id: int, reason: str, summary: str, history: Sequence[Message]
) -> str:
    """Карточка обращения: повод, сводка и последние реплики диалога. Ответы
    бота в истории — текстом со ссылками на рецепты, так оператор их и видит."""
    lines = [f"<b>Обращение #{ticket_id}</b> · {html.escape(reason_label(reason))}"]
    if summary:
        lines.append(f"<b>Суть:</b> {html.escape(_clip(summary, CARD_USER_CHARS))}")
    turns = history[-CARD_TURNS:]
    if turns:
        lines += ["", "<b>Последние сообщения:</b>"]
        for turn in turns:
            user = turn["role"] == "user"
            text = _clip(turn["content"], CARD_USER_CHARS if user else CARD_BOT_CHARS)
            lines.append(f"{'👤' if user else '🤖'} {html.escape(text)}")
    lines += [
        "",
        "Пишите в этой теме — бот перешлёт пользователю. Закрыть обращение — /close.",
    ]
    return "\n".join(lines)
