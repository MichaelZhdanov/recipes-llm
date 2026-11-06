"""Тексты бота: ответ в HTML для Telegram, кнопки, статусы и служебные фразы.

Без aiogram: кнопки — пары (текст, callback_data), приложение превращает их
в клавиатуру. Так всё, что видит пользователь, проверяется обычными тестами.

- `parse_mode=HTML`: всё, что пришло от модели и с сайта, экранируется.
- Утверждения — как написала модель (эмодзи вместо маркеров списка), без
  номеров; источники — нумерованным списком: название и «способ
  приготовления» (ссылка, если страница жива).
- Telegram ограничивает сообщение 4096 символами после разбора разметки;
  длина HTML — оценка с запасом. Не помещается — убираются последние
  утверждения, источники остаются.
"""

import html
import random
from collections.abc import Mapping, Sequence

from assistant.generation.answer import REASON_TEXT, Answer, SourceLink, Stage

MAX_CHARS = 3800

Button = tuple[str, str]
Keyboard = list[list[Button]]

START_TEXT = (
    "👋 Здравствуйте! Я помогу подобрать рецепт с povarenok.ru.\n\n"
    "Напишите, что хочется или что есть дома: «что-нибудь с курицей, но без лука», "
    "«блины на молоке», «что приготовить на ужин из фарша и картошки». "
    "Можно уточнять: «а без сахара?», «сколько муки во втором?».\n\n"
    "/new — начать разговор заново."
)
HELP_TEXT = (
    "Я ищу рецепты на povarenok.ru и отвечаю только тем, что в них написано: "
    "состав и количества, а способ приготовления — по ссылке на странице рецепта.\n\n"
    "/new — забыть разговор и начать заново\n/help — эта подсказка"
)
NEW_TEXT = "🧹 Начинаем заново. Что приготовим?"
BUSY_TEXT = (
    "⏳ Секунду, ещё отвечаю на прошлый вопрос — пришлю ответ, и спросите дальше."
)
NOT_TEXT = "Пока понимаю только текст. Напишите, что хочется приготовить."
TOO_LONG_TEXT = "Слишком длинное сообщение. Опишите, что хочется приготовить, покороче."
ERROR_TEXT = "😔 Что-то пошло не так, и ответить сейчас не получилось. Попробуйте ещё раз чуть позже."
THANKS_TEXT = "Спасибо, учту!"
WHAT_WRONG_TEXT = "Что не так?"
REASON_BUTTONS: list[Button] = [
    ("Непонятно", "unclear"),
    ("Не то искал", "wrong"),
    ("Не получилось", "failed"),
]

# Статусы, пока бот думает: фраза по стадии ответа выбирается случайно.
STATUS: dict[Stage, list[str]] = {
    "search": [
        "🔎 Ищу рецепты…",
        "📖 Листаю кулинарную книгу…",
        "🧺 Заглядываю в кладовую…",
    ],
    "answer": [
        "🧑‍🍳 Подбираю лучшее…",
        "✍️ Записываю ингредиенты…",
    ],
    "review": [
        "🧂 Сверяю с рецептами…",
        "✅ Перепроверяю, чтобы ничего не напутать…",
    ],
}


def status_text(stage: Stage, rng: random.Random | None = None) -> str:
    return (rng or random).choice(STATUS[stage])


def _source_line(number: int, link: SourceLink, alive: bool) -> str:
    name = html.escape(link.name)
    if alive:
        url = html.escape(link.url, quote=True)
        return f'{number}. 📖 <b>{name}</b> — <a href="{url}">способ приготовления</a>'
    return f"{number}. 📖 <b>{name}</b> — страница рецепта на сайте недоступна"


def answer_html(
    answer: Answer, pages: Mapping[str, bool], *, gigachat_note: bool = False
) -> str:
    """Ответ для Telegram. pages — жива ли страница рецепта, по url."""
    if answer.status == "chat":
        return html.escape(answer.message)
    if answer.reason is not None:
        head = [html.escape(REASON_TEXT[answer.reason]) + "."]
        if answer.message:
            head.append(html.escape(answer.message))
        if answer.sources:
            head += ["", "Вот что может пригодиться:"]
        statements: list[str] = []
    else:
        head = [html.escape(answer.message)] if answer.message else []
        statements = [html.escape(s.text) for s in answer.statements]
    sources = [
        _source_line(link.number, link, pages.get(link.url, False))
        for link in answer.sources
    ]
    note = (
        ["", f"Ответ сгенерирован GigaChat ({html.escape(answer.model)})."]
        if gigachat_note
        else []
    )

    def build(kept: Sequence[str]) -> str:
        parts = [*head, *kept]
        if sources:
            parts += ["", *sources]
        return "\n".join([*parts, *note])

    text = build(statements)
    while len(text) > MAX_CHARS and statements:
        statements = statements[:-1]
        text = build(statements)
    return text[:MAX_CHARS]


def answer_buttons(
    answer_id: int, answer: Answer, pages: Mapping[str, bool]
) -> Keyboard:
    """👍/👎 под ответом и «🍳 Шаги» для рецептов без живой страницы."""
    if answer.status == "chat":
        return []
    rows: Keyboard = [
        [("👍", f"vote:{answer_id}:up"), ("👎", f"vote:{answer_id}:down")]
    ]
    for link in answer.sources:
        if not pages.get(link.url, False):
            label = link.name if len(link.name) <= 40 else link.name[:39] + "…"
            rows.append([(f"🍳 Шаги: {label}", f"steps:{answer_id}:{link.recipe_id}")])
    return rows


def reason_buttons(answer_id: int) -> Keyboard:
    return [[(text, f"why:{answer_id}:{code}")] for text, code in REASON_BUTTONS]
