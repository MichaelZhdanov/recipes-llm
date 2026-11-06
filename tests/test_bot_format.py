"""Сообщение бота в HTML, кнопки и проверка страницы рецепта."""

import random

import httpx2

from assistant.bot.format import (
    MAX_CHARS,
    STATUS,
    answer_buttons,
    answer_html,
    reason_buttons,
    status_text,
)
from assistant.bot.pages import PageChecker
from assistant.bot.store import Store
from assistant.generation.answer import Answer, AnswerStatement, SourceLink

ALIVE = "https://www.povarenok.ru/recipes/show/1/"
DEAD = "https://www.povarenok.ru/recipes/show/2/"


def answer(statements: list[str], message: str = "🍳 Вот что нашлось:") -> Answer:
    return Answer(
        status="answer",
        reason=None,
        message=message,
        statements=[AnswerStatement(text=t, sources=[1]) for t in statements],
        sources=[
            SourceLink(
                number=1, recipe_id=1, name="Блины <тонкие> & нежные", url=ALIVE
            ),
            SourceLink(number=2, recipe_id=2, name="Оладьи", url=DEAD),
        ],
        model="claude-haiku-5-5",
        prompt_version="recipes_v3",
        dataset_revision="ffaf49ee",
    )


def test_answer_is_escaped_html_with_recipe_links_and_dead_pages_marked() -> None:
    text = answer_html(answer(["🥞 Молока 500 мл <много>"]), {ALIVE: True, DEAD: False})

    assert text.startswith("🍳 Вот что нашлось:\n🥞 Молока 500 мл &lt;много&gt;")
    assert '1. 📖 <b>Блины &lt;тонкие&gt; &amp; нежные</b> — <a href="' + ALIVE in text
    assert "2. 📖 <b>Оладьи</b> — страница рецепта на сайте недоступна" in text
    assert "GigaChat" not in text


def test_long_answer_drops_last_statements_but_keeps_sources() -> None:
    long = ["🥞 " + "очень длинное утверждение " * 30] * 12

    text = answer_html(answer(long), {ALIVE: True})

    assert len(text) <= MAX_CHARS
    assert "1. 📖" in text and "2. 📖" in text


def test_chat_and_not_found_texts() -> None:
    chat = answer([], message="Привет! Подберу рецепт.").model_copy(
        update={"status": "chat", "sources": []}
    )
    refused = answer([], message="Могу подобрать рецепт по ингредиентам.").model_copy(
        update={"status": "not_found", "reason": "model_not_found", "sources": []}
    )

    assert answer_html(chat, {}) == "Привет! Подберу рецепт."
    assert answer_buttons(1, chat, {}) == []
    assert answer_html(refused, {}).startswith(
        "Не нашёл ответа в найденных источниках."
    )


def test_gigachat_note_only_when_asked() -> None:
    assert "сгенерирован GigaChat" in answer_html(answer(["…"]), {}, gigachat_note=True)


def test_buttons_vote_and_steps_only_for_dead_pages() -> None:
    rows = answer_buttons(5, answer(["…"]), {ALIVE: True, DEAD: False})

    assert rows[0] == [("👍", "vote:5:up"), ("👎", "vote:5:down")]
    assert rows[1:] == [[("🍳 Шаги: Оладьи", "steps:5:2")]]
    assert [row[0][1] for row in reason_buttons(5)] == [
        "why:5:unclear",
        "why:5:wrong",
        "why:5:failed",
    ]
    assert all(len(data.encode()) <= 64 for row in rows for _, data in row)


def test_status_phrase_comes_from_the_stage_pool() -> None:
    assert status_text("review", random.Random(1)) in STATUS["review"]


def checker(
    statuses: dict[str, int | Exception], store: Store
) -> tuple[PageChecker, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        result = statuses[str(request.url)]
        if isinstance(result, Exception):
            raise result
        return httpx2.Response(result)

    return PageChecker(
        httpx2.Client(transport=httpx2.MockTransport(handle)), store
    ), seen


def test_page_check_caches_only_definite_answers() -> None:
    store = Store(":memory:")
    broken = "https://www.povarenok.ru/recipes/show/3/"
    offline = "https://www.povarenok.ru/recipes/show/4/"
    pages, seen = checker(
        {ALIVE: 200, DEAD: 404, broken: 503, offline: httpx2.ConnectError("нет сети")},
        store,
    )

    first = pages.alive_many([ALIVE, DEAD, broken, offline, ALIVE])
    again = pages.alive_many([ALIVE, DEAD, broken, offline])

    assert first == {ALIVE: True, DEAD: False, broken: False, offline: False}
    assert again == first
    # 200 и 404 — из кэша, 503 и сбой сети спрашиваются снова.
    assert sorted(str(r.url) for r in seen).count(ALIVE) == 1
    assert [str(r.url) for r in seen].count(broken) == 2
    assert all(r.headers["User-Agent"].startswith("recipes-llm/") for r in seen)


def test_redirect_is_not_followed_and_means_no_link() -> None:
    store = Store(":memory:")
    pages, seen = checker({ALIVE: 301}, store)

    assert pages.alive(ALIVE) is False
    assert len(seen) == 1
