from chunk_helpers import make_chunk


def test_recipe_text_has_name_then_one_ingredient_per_line() -> None:
    chunk = make_chunk(
        "Блины на молоке",
        {"Молоко": "500 мл", "Яйцо куриное": "2 шт", "Соль": None},
        recipe_id=42,
    )

    assert chunk.text == "Блины на молоке\nМолоко — 500 мл\nЯйцо куриное — 2 шт\nСоль"
    assert chunk.source_id == "recipe/42"
    assert chunk.url == "https://www.povarenok.ru/recipes/show/42/"


def test_ingredients_for_filters_are_lowercase_without_repeats() -> None:
    chunk = make_chunk("Суп", {"Картофель": "3 шт", "картофель": "1 шт", "Вода": "1 л"})

    assert chunk.ingredients == ["картофель", "вода"]


def test_embedding_text_has_no_amounts() -> None:
    chunk = make_chunk("Блины на молоке", {"Молоко": "500 мл", "Яйцо куриное": "2 шт"})

    assert chunk.embedding_text() == "Блины на молоке\nмолоко, яйцо куриное"
