"""Карточка страницы рецепта: главное фото и шаги по разметке schema.org."""

import httpx2

from assistant.bot.recipe_page import MAX_BYTES, RecipeCard, fetch_card, parse_card

URL = "https://www.povarenok.ru/recipes/show/1/"
# Синтетическая страница с той же разметкой, что у povarenok.ru.
PAGE = """
<html><body>
<h1 itemprop="name">Блины</h1>
<div class="m-img"><img itemprop="image" src="https://www.povarenok.ru/data/main.jpg"></div>
<img itemprop="image" src="https://www.povarenok.ru/data/other.jpg">
<h2>Рецепт «Блины»:</h2>
<ul itemprop="recipeInstructions">
  <li class="cooking-bl">
    <span class="cook-img"><img itemprop="image" src="https://www.povarenok.ru/data/1.jpg"></span>
    <div><p>Смешайте муку с молоком.</p></div>
  </li>
  <li class="cooking-bl"><div><p>Дайте тесту постоять.</p></div></li>
  <li class="cooking-bl">
    <span class="cook-img"><img itemprop="image" src="https://www.povarenok.ru/data/3.jpg"></span>
    <div><p>Жарьте на сковороде.</p></div>
  </li>
</ul>
</body></html>
"""


def test_main_photo_and_steps_with_their_photos_are_counted() -> None:
    assert parse_card(PAGE) == RecipeCard(
        photo="https://www.povarenok.ru/data/main.jpg", steps=3, step_photos=2
    )


def test_photo_from_another_site_is_not_taken_and_no_markup_means_no_steps() -> None:
    foreign = PAGE.replace(
        "https://www.povarenok.ru/data/main.jpg", "https://evil.example/x.jpg"
    )

    assert parse_card(foreign).photo == "https://www.povarenok.ru/data/other.jpg"
    assert parse_card("<html><p>Страница без разметки</p></html>") == RecipeCard(
        None, 0, 0
    )


def client(status: int, body: bytes, charset: str = "windows-1251") -> httpx2.Client:
    return httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                status,
                content=body,
                headers={"Content-Type": f"text/html; charset={charset}"},
            )
        )
    )


def test_page_in_windows_1251_is_decoded_and_bad_answers_give_no_card() -> None:
    page = client(200, PAGE.encode("cp1251"))

    assert fetch_card(page, URL) == parse_card(PAGE)
    assert fetch_card(client(404, b""), URL) is None
    assert fetch_card(client(200, b"x" * (MAX_BYTES + 1)), URL) is None
