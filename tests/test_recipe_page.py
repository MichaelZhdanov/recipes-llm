"""Шаги со страницы рецепта: главное фото и шаги по разметке schema.org."""

import httpx2

from assistant.bot.recipe_page import (
    MAX_BYTES,
    MAX_PHOTO_BYTES,
    RecipePage,
    Step,
    fetch_page,
    fetch_photo,
    parse_page,
)

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
    <span class="cook-img"><a href="https://www.povarenok.ru/data/1-640x480.jpg"><img itemprop="image" src="https://www.povarenok.ru/data/1-300x0.jpg"></a></span>
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


def test_steps_come_in_order_with_their_largest_photo() -> None:
    page = parse_page(PAGE)

    assert page.photo == "https://www.povarenok.ru/data/main.jpg"
    assert page.steps == [
        Step("Смешайте муку с молоком.", "https://www.povarenok.ru/data/1-640x480.jpg"),
        Step("Дайте тесту постоять.", None),
        Step("Жарьте на сковороде.", "https://www.povarenok.ru/data/3.jpg"),
    ]


def test_photo_from_another_site_is_not_taken_and_no_markup_means_no_steps() -> None:
    foreign = PAGE.replace(
        "https://www.povarenok.ru/data/3.jpg", "https://evil.example/3.jpg"
    )

    assert parse_page(foreign).steps[2] == Step("Жарьте на сковороде.", None)
    assert parse_page("<html><p>Страница без разметки</p></html>") == RecipePage(None)


# Старый формат povarenok (так устроены, например, 22811 и 5841): пошаговой
# разметки нет, шаги — абзацы одного <div> после заголовка «Рецепт …».
OLD_PAGE = """
<html><body>
<div class="article-text" itemprop="description"><div>Суп на скорую руку.</div></div>
<h2>Ингредиенты для «Суп»:</h2>
<div class="ingredients-bl"><ul><li itemprop="recipeIngredient">Грибы</li></ul></div>
<h2>Рецепт &laquo;Суп&raquo;:</h2>

  <div>Грибы промойте и залейте водой.
<br /> <br />
Сварите картофель, добавьте <b>обжаренный</b> лук.
<br /> <br />
Приятного аппетита!</div>
<div class="article-tags"><a href="https://www.povarenok.ru/recipes/destiny/2/">На обед</a></div>
</body></html>
"""


def test_old_recipes_without_step_markup_give_paragraphs_as_steps() -> None:
    """В логе бота: «шаги со страницы рецепта 22811: 0» — страница есть, а
    разметки шагов у старых рецептов нет."""
    assert parse_page(OLD_PAGE).steps == [
        Step("Грибы промойте и залейте водой.", None),
        Step("Сварите картофель, добавьте обжаренный лук.", None),
        Step("Приятного аппетита!", None),
    ]


def test_only_a_div_right_after_the_recipe_heading_is_taken_as_steps() -> None:
    # Другой заголовок перед <div> — не шаги (описание, ингредиенты, теги).
    other = OLD_PAGE.replace("<h2>Рецепт &laquo;Суп&raquo;:</h2>", "<h2>Советы:</h2>")
    # После заголовка «Рецепт» — новый формат: шаги только из разметки.
    assert parse_page(other).steps == []
    assert [s.text for s in parse_page(PAGE).steps] == [
        "Смешайте муку с молоком.",
        "Дайте тесту постоять.",
        "Жарьте на сковороде.",
    ]


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

    assert fetch_page(page, URL) == parse_page(PAGE)
    assert fetch_page(client(404, b""), URL) is None
    assert fetch_page(client(200, b"x" * (MAX_BYTES + 1)), URL) is None


def image(status: int, body: bytes, content_type: str = "image/jpeg") -> httpx2.Client:
    return httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                status, content=body, headers={"Content-Type": content_type}
            )
        )
    )


def test_photo_is_downloaded_only_from_the_recipe_site_and_only_if_an_image() -> None:
    photo = "https://www.povarenok.ru/data/main.jpg"

    assert fetch_photo(image(200, b"jpeg"), photo) == b"jpeg"
    assert fetch_photo(image(200, b"jpeg"), "https://evil.example/x.jpg") is None
    assert fetch_photo(image(200, b"<html>", "text/html"), photo) is None
    assert fetch_photo(image(404, b""), photo) is None
    assert fetch_photo(image(200, b"x" * (MAX_PHOTO_BYTES + 1)), photo) is None
