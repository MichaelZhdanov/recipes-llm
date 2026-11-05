"""Точка индекса — один рецепт.

Из строки датасета (`dataset.Recipe`) получается `Chunk`: его payload лежит
в Qdrant и возвращается поиском целиком.

- `text` — то, что видит модель: название первой строкой, дальше по строке
  на ингредиент («Молоко — 250 мл»). Цитата из ответа сверяется с этим
  текстом, а номер строки показывает, на какой ингредиент она опирается.
- `embedding_text()` — название и имена ингредиентов без количеств:
  для похожести по смыслу «250 мл» только шум.
- `ingredients` — имена в нижнем регистре: по ним фильтр Qdrant
  «с чем и без чего» (этап 2). Регистр на сайте не всегда один и тот же.
"""

from pydantic import BaseModel, ConfigDict, Field

from assistant.indexing.dataset import Recipe


class Chunk(BaseModel):
    """Рецепт в индексе. Все поля обязательны (приёмка F1)."""

    model_config = ConfigDict(frozen=True)

    recipe_id: int = Field(ge=1)
    url: str = Field(min_length=1)
    name: str = Field(min_length=1)
    ingredients: list[str]
    text: str = Field(min_length=1)
    # Ревизия датасета, из которой взят рецепт.
    ref: str = Field(min_length=1)

    @property
    def source_id(self) -> str:
        """Идентификатор источника для ссылок и отчётов eval."""
        return f"recipe/{self.recipe_id}"

    def embedding_text(self) -> str:
        return f"{self.name}\n{', '.join(self.ingredients)}"


def recipe_text(recipe: Recipe) -> str:
    lines = [recipe.name]
    lines += [
        f"{item.name} — {item.amount}" if item.amount else item.name
        for item in recipe.ingredients
    ]
    return "\n".join(lines)


def recipe_chunk(recipe: Recipe, ref: str) -> Chunk:
    return Chunk(
        recipe_id=recipe.id,
        url=recipe.url,
        name=recipe.name,
        # Порядок как на сайте, повторы после нижнего регистра убираются.
        ingredients=list(
            dict.fromkeys(item.name.lower() for item in recipe.ingredients)
        ),
        text=recipe_text(recipe),
        ref=ref,
    )
