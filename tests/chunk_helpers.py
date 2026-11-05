"""Рецепты в индексе для тестов.

`make_chunk` идёт через ту же `recipe_chunk`, что и сборка индекса, поэтому
тесты видят ровно такой текст и такие имена ингредиентов, как в проде.
`text_chunk` — точка с произвольным текстом: проверке цитат всё равно,
рецепт это или нет, ей нужен текст со строками.
"""

from assistant.indexing.chunks import Chunk, recipe_chunk
from assistant.indexing.dataset import Ingredient, Recipe


def make_recipe(
    name: str, ingredients: dict[str, str | None] | None = None, *, recipe_id: int = 1
) -> Recipe:
    items = ingredients if ingredients is not None else {"Соль": "по вкусу"}
    return Recipe(
        id=recipe_id,
        url=f"https://www.povarenok.ru/recipes/show/{recipe_id}/",
        name=name,
        ingredients=tuple(Ingredient(k, v) for k, v in items.items()),
    )


def make_chunk(
    name: str,
    ingredients: dict[str, str | None] | None = None,
    *,
    recipe_id: int = 1,
    ref: str = "v3.4.2",
) -> Chunk:
    return recipe_chunk(make_recipe(name, ingredients, recipe_id=recipe_id), ref)


def text_chunk(text: str, *, recipe_id: int = 1, ref: str = "v3.4.2") -> Chunk:
    return Chunk(
        recipe_id=recipe_id,
        url=f"https://www.povarenok.ru/recipes/show/{recipe_id}/",
        name=text.split("\n", 1)[0] or "рецепт",
        ingredients=[],
        text=text,
        ref=ref,
    )


# Небольшой набор рецептов для тестов поиска, сборки индекса и ответов.
ROWS = [
    make_recipe(
        "Густой молочно-клубничный коктейль",
        {"Молоко": "250 мл", "Клубника": "200 г", "Сахар": "15 г"},
        recipe_id=164365,
    ),
    make_recipe(
        "Курица с рисом в духовке",
        {"Курица": "500 г", "Рис": "200 г", "Лук репчатый": "1 шт", "Соль": "по вкусу"},
        recipe_id=2001,
    ),
    make_recipe(
        "Овсяное печенье",
        {"Хлопья овсяные": "200 г", "Яйцо куриное": "2 шт", "Сахар": "100 г"},
        recipe_id=3001,
    ),
]
RECIPES = [recipe_chunk(row, "v3.4.2") for row in ROWS]
COCKTAIL, CHICKEN, COOKIES = RECIPES
