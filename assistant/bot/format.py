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
    "/new — начать разговор заново, /human — позвать повара-консультанта."
)
HELP_TEXT = (
    "Я ищу рецепты на povarenok.ru и отвечаю только тем, что в них написано: "
    "состав и количества, а способ приготовления — по ссылке на странице рецепта.\n\n"
    "Если не получилось, непонятно или вопрос о самочувствии — передам разговор "
    "живому повару-консультанту, он ответит здесь же.\n\n"
    "/new — забыть разговор и начать заново\n/human — позвать повара\n"
    "/bot — вернуться ко мне от повара\n/help — эта подсказка"
)
# Склад Grocy в /start и /help (решения автора 70–71): предложение подключить —
# кнопкой или /grocy; в MVP подключается общий склад на сервере; отключить —
# /grocy_off; без склада — поиск по всем рецептам, как обычно. Только когда
# Grocy на этом сервере есть: иначе подключать нечего.
START_GROCY_OFFER_TEXT = (
    "🧺 Хотите, я буду подбирать рецепты из того, что есть у вас дома? Подключите "
    "склад продуктов Grocy — кнопкой ниже или командой /grocy. Пока это демо: "
    "подключится общий склад на сервере. Отключить — /grocy_off. Без склада я ищу "
    "по всем рецептам, как обычно."
)
START_GROCY_ON_TEXT = (
    "🧺 Склад Grocy подключён: на новый вопрос я спрошу, искать ли из того, что есть "
    "дома, или по всем рецептам. Отключить — /grocy_off."
)
HELP_GROCY_TEXT = (
    "Склад продуктов Grocy: /grocy подключает его (пока это демо — общий склад на "
    "сервере), /grocy_off отключает. С подключённым складом на новый вопрос я спрошу, "
    "искать ли из того, что есть дома, или по всем рецептам: сначала — рецепты "
    "целиком из ваших продуктов, а если не хватает одного-четырёх — покажу и их, "
    "с тем, что докупить. Без склада ищу по всем рецептам, как обычно."
)
GROCY_CONNECT_BUTTON: Button = ("🧺 Подключить Grocy", "grocy:on")


def start_text(grocy: bool = False, connected: bool = False) -> str:
    """/start; grocy — есть ли Grocy на сервере, connected — подключён ли у чата."""
    if not grocy:
        return START_TEXT
    head, _, commands = START_TEXT.rpartition("\n\n")
    about = START_GROCY_ON_TEXT if connected else START_GROCY_OFFER_TEXT
    return f"{head}\n\n{about}\n\n{commands}"


def start_buttons(grocy: bool = False, connected: bool = False) -> Keyboard:
    """Под /start — «Подключить Grocy», если есть что подключать."""
    return [[GROCY_CONNECT_BUTTON]] if grocy and not connected else []


def help_text(grocy: bool = False) -> str:
    """/help; grocy — есть ли Grocy на сервере."""
    if not grocy:
        return HELP_TEXT
    head, _, commands = HELP_TEXT.rpartition("\n\n")
    commands = commands.replace(
        "/bot — вернуться ко мне от повара\n",
        "/bot — вернуться ко мне от повара\n"
        "/grocy — подключить склад Grocy\n/grocy_off — отключить склад Grocy\n",
    )
    return f"{head}\n\n{HELP_GROCY_TEXT}\n\n{commands}"


NEW_TEXT = "🧹 Начинаем заново. Что приготовим?"
BUSY_TEXT = (
    "⏳ Секунду, ещё отвечаю на прошлый вопрос — пришлю ответ, и спросите дальше."
)
NOT_TEXT = "Пока понимаю только текст. Напишите, что хочется приготовить."
TOO_LONG_TEXT = "Слишком длинное сообщение. Опишите, что хочется приготовить, покороче."
ERROR_TEXT = "😔 Что-то пошло не так, и ответить сейчас не получилось. Попробуйте ещё раз чуть позже."
THANKS_TEXT = "Спасибо, учту!"
GONE_TEXT = "Этот ответ уже не найти."
WHAT_WRONG_TEXT = "Что не так?"
REASON_BUTTONS: list[Button] = [
    ("Непонятно", "unclear"),
    ("Не то искал", "wrong"),
    ("Не получилось", "failed"),
]

# --- Передача человеку (этап 5) ---
HUMAN_BUTTON = "🧑‍🍳 Позвать повара"
HANDOFF_TEXT = (
    "🧑‍🍳 Передаю разговор повару-консультанту — он ответит здесь же. "
    "Пишите ему прямо в этот чат. Вернуться ко мне — /bot."
)
HEALTH_TEXT = (
    "🔴 Советов о здоровье я не даю. При плохом самочувствии обратитесь к врачу, "
    "экстренно — 103 или 112."
)
NO_OPERATORS_TEXT = (
    "😔 Передать разговор человеку сейчас не получается. "
    "Попробуйте чуть позже или спросите меня о рецепте."
)
ALREADY_WITH_HUMAN_TEXT = (
    "Повар уже на связи — пишите ему здесь. Вернуться ко мне — /bot."
)
NOT_WITH_HUMAN_TEXT = "Я и так на связи 🙂 Что приготовим?"
BACK_TO_BOT_TEXT = "🤖 Снова я. Как прошёл разговор с поваром?"
CLOSED_BY_OPERATOR_TEXT = (
    "🧑‍🍳 Повар закрыл обращение, я снова на связи. Как прошёл разговор?"
)
NOT_FOUND_TWICE_TEXT = "Если хотите, позову живого повара — он поможет подобрать."
FAILED_TEXT = (
    "Жаль, что не получилось. Могу позвать повара-консультанта — он разберётся."
)
ALLERGY_NOTE = (
    "⚠️ В покупных продуктах бывают скрытые аллергены — проверьте состав на упаковке."
)
# Тексты для темы в группе операторов.
TOPIC_CLOSED_TEXT = "✅ Обращение закрыто, пользователь снова с ботом."
USER_LEFT_TEXT = "↩️ Пользователь вернулся к боту — обращение закрыто."
OPERATOR_ONLY_TEXT = "Здесь пишут операторы: тема обращения закрыта или не найдена."
EXPIRED_TEXT = (
    "⏰ Обращение к повару закрыто: 12 часов не было сообщений. Я снова на связи — "
    "как прошёл разговор?"
)
TOPIC_EXPIRED_TEXT = "⏰ Обращение закрыто автоматически: 12 часов без сообщений."

# --- Склад Grocy (этап 6) ---
GROCY_ON_TEXT = (
    "🧺 Склад Grocy подключён — пока это демо, общий склад на сервере. На новый "
    "вопрос я спрошу, искать ли из того, что есть дома, или по всем рецептам. "
    "Отключить — /grocy_off."
)
GROCY_ALREADY_ON_TEXT = "Склад Grocy уже подключён. Отключить — /grocy_off."
GROCY_OFF_TEXT = "Склад Grocy отключён — ищу по всем рецептам, как обычно. Подключить снова — /grocy."
GROCY_ALREADY_OFF_TEXT = (
    "Склад Grocy и так не подключён — ищу по всем рецептам. Подключить — /grocy."
)
GROCY_UNAVAILABLE_TEXT = "Склада Grocy на этом сервере нет — ищу по всем рецептам."
SOURCE_QUESTION_TEXT = "Искать из того, что есть дома, или по всем рецептам?"
SOURCE_BUTTONS: list[Button] = [
    ("🧺 Из того, что есть", "source:stock"),
    ("📖 По всем рецептам", "source:all"),
]
ASK_AGAIN_TEXT = "Напишите, пожалуйста, вопрос ещё раз — я его потерял."
STOCK_DOWN_TEXT = "⚠️ Склад Grocy сейчас недоступен — ищу по всем рецептам."
# После строгого ответа — строка и кнопка «поискать ещё»: рецепты, где не хватает
# 1–4 продуктов (формулировка автора).
NEAR_OFFER_TEXT = "Есть рецепты, в которых не хватает пары продуктов."
NEAR_BUTTON = "🔎 Поискать ещё"

STEPS_STATUS_TEXT = (
    "🍳 Расписываю шаги и сверяю их с рецептом — это займёт около 20 секунд…"
)
STEPS_NOTE = (
    "⚠️ Шаги написал ИИ: страницы рецепта на сайте нет. "
    "Ориентируйтесь на вид и готовность продуктов."
)

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
    # Проверка сказала «рецепты не по вопросу» — ищем заново.
    "research": [
        "🔄 Нашлось не совсем то — ищу ещё…",
        "🧭 Уточняю поиск, чтобы было точно по вопросу…",
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


def _missing_line(missing: Sequence[str]) -> str:
    """Чего не хватает рецепту из склада — считает код по строкам рецепта."""
    if not missing:
        return ""
    return "\n    🛒 Не хватает: " + html.escape(", ".join(missing))


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
        + _missing_line(answer.missing.get(link.recipe_id, []))
        for link in answer.sources
    ]
    note = (
        ["", f"Ответ сгенерирован GigaChat ({html.escape(answer.model)})."]
        if gigachat_note
        else []
    )

    # Аллергия как условие (решение 66) — предупреждение от кода, не от модели.
    allergy = ["", html.escape(ALLERGY_NOTE)] if answer.allergy and statements else []

    def build(kept: Sequence[str]) -> str:
        parts = [*head, *kept]
        if sources:
            parts += ["", *sources]
        return "\n".join([*parts, *allergy, *note])

    text = build(statements)
    while len(text) > MAX_CHARS and statements:
        statements = statements[:-1]
        text = build(statements)
    return text[:MAX_CHARS]


def answer_buttons(
    answer_id: int,
    answer: Answer,
    pages: Mapping[str, bool],
    *,
    ai_steps: bool = True,
) -> Keyboard:
    """👍/👎 под ответом; у рецепта с живой страницей — «📖 Шаги с фото»
    (фото блюда и число шагов, сами шаги по ссылке), без неё — «🍳 Шаги» от ИИ.
    ai_steps=False — шаги от ИИ некому проверить: у такого рецепта кнопки нет."""
    if answer.status == "chat":
        return []
    rows: Keyboard = [
        [("👍", f"vote:{answer_id}:up"), ("👎", f"vote:{answer_id}:down")]
    ]
    for link in answer.sources:
        label = link.name if len(link.name) <= 40 else link.name[:39] + "…"
        if pages.get(link.url, False):
            rows.append(
                [(f"📖 Шаги с фото: {label}", f"card:{answer_id}:{link.recipe_id}")]
            )
        elif ai_steps:
            rows.append([(f"🍳 Шаги: {label}", f"steps:{answer_id}:{link.recipe_id}")])
    return rows


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


# Подпись к фото в Telegram — до 1024 символов.
MAX_CAPTION = 1000


def page_steps_header(
    name: str, url: str, steps: int, missing: Sequence[str] = ()
) -> str:
    """Первое сообщение шагов со страницы: что за рецепт, откуда шаги и — для
    рецепта из расширенного поиска по складу — чего не хватает."""
    link = f'<a href="{html.escape(url, quote=True)}">povarenok.ru</a>'
    head = f"📖 <b>{html.escape(name)}</b>"
    if not steps:
        text = f"{head}\nШагов на странице не нашлось — рецепт на {link}"
    else:
        count = f"{steps} {_plural(steps, 'шаг', 'шага', 'шагов')}"
        text = f"{head} — {count} со страницы рецепта.\nИсточник: {link}"
    if missing:
        text += "\n🛒 Не хватает: " + html.escape(", ".join(missing))
    return text


def step_caption(number: int, text: str) -> str:
    """Текст шага под его фото; длинный — обрезается до лимита подписи."""
    caption = f"{number}. {html.escape(text)}"
    if len(caption) <= MAX_CAPTION:
        return caption
    return caption[: MAX_CAPTION - 1] + "…"


def reason_buttons(answer_id: int) -> Keyboard:
    return [[(text, f"why:{answer_id}:{code}")] for text, code in REASON_BUTTONS]


def source_buttons() -> Keyboard:
    """Откуда искать при включённом Grocy: из того, что есть, или по всем."""
    return [SOURCE_BUTTONS]


def near_buttons(answer_id: int) -> Keyboard:
    """Расширенный поиск по складу для вопроса этого ответа."""
    return [[(NEAR_BUTTON, f"near:{answer_id}")]]


def human_buttons(reason: str) -> Keyboard:
    """Кнопка «Позвать повара» с поводом: сбой, два «не нашёл», 👎 «не получилось»."""
    return [[(HUMAN_BUTTON, f"human:{reason}")]]


def rate_buttons(ticket_id: int) -> Keyboard:
    """👍/👎 разговору с поваром после закрытия обращения."""
    return [[("👍", f"rate:{ticket_id}:up"), ("👎", f"rate:{ticket_id}:down")]]


def steps_html(name: str, steps: Sequence[str]) -> str:
    """Шаги для рецепта без живой страницы — с пометкой, что их написал ИИ."""
    lines = [f"🍳 <b>{html.escape(name)}</b> — как приготовить", ""]
    lines += [f"{n}. {html.escape(step)}" for n, step in enumerate(steps, start=1)]
    return "\n".join([*lines, "", html.escape(STEPS_NOTE)])[:MAX_CHARS]


def steps_refused_html(name: str) -> str:
    return (
        f"😔 Не смог надёжно расписать шаги для «{html.escape(name)}» — "
        "не хочу советовать наугад. Напишите, что хочется приготовить, "
        "и я поищу похожий рецепт."
    )
