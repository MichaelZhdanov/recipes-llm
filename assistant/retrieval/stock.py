"""Подбор рецептов по складу «что есть дома» (этап 6, решение автора 72).

Сначала — рецепты целиком из того, что есть: каждый ингредиент рецепта — со
склада или из базовых (STAPLES — соль, вода, чёрный перец, растительное
масло: они есть почти всегда, и без них «из того, что есть» не находилось бы
ничего). Это фильтр Qdrant (`IngredientFilter.only`), до векторов.

Если хорошо подходящему по вопросу рецепту не хватает от 1 до MAX_MISSING
продуктов, бот предлагает расширенный поиск: кандидаты — рецепты хотя бы с
одним продуктом со склада, лучшие по похожести на вопрос; чего не хватает,
считает код по строкам рецепта, а не модель.

Склад приходит из Grocy названиями продуктов; до поиска доходят только имена
из словаря индекса — чужое название (в том числе «инструкция» в нём)
отбрасывается и считается.
"""

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.store import Hit

STAPLES = (
    "соль",
    "вода",
    "перец черный",
    "перец черный молотый",
    "масло растительное",
)
# Сколько продуктов может не хватать в расширенном поиске (решение автора).
MAX_MISSING = 4
# Сколько лучших по вопросу рецептов смотреть, чтобы найти «почти готовые».
NEAR_CANDIDATES = 40


@dataclass(frozen=True)
class Stock:
    """Склад, сопоставленный со словарём индекса."""

    names: tuple[str, ...]
    # Названия со склада, которых нет в словаре: в поиск не попали.
    dropped: tuple[str, ...] = ()


def known_stock(names: Iterable[str], vocabulary: Collection[str]) -> Stock:
    known: list[str] = []
    dropped: list[str] = []
    for raw in names:
        name = raw.strip().lower()
        (known if name in vocabulary else dropped).append(name)
    return Stock(tuple(dict.fromkeys(known)), tuple(dict.fromkeys(dropped)))


def strict_conditions(base: IngredientFilter, stock: Stock) -> IngredientFilter:
    """Условия вопроса и «всё со склада или из базовых», хотя бы один — со склада."""
    return IngredientFilter(
        include=(*base.include, stock.names),
        exclude=base.exclude,
        only=(*stock.names, *STAPLES),
    )


def near_conditions(base: IngredientFilter, stock: Stock) -> IngredientFilter:
    """Кандидаты расширенного поиска: условия вопроса и хотя бы один продукт со склада."""
    return IngredientFilter(include=(*base.include, stock.names), exclude=base.exclude)


def missing(ingredients: Sequence[str], stock: Stock) -> tuple[str, ...]:
    """Чего не хватает рецепту — в порядке строк рецепта, без базовых."""
    have = {*stock.names, *STAPLES}
    return tuple(dict.fromkeys(name for name in ingredients if name not in have))


def near_hits(
    hits: Sequence[Hit], stock: Stock, limit: int
) -> list[tuple[Hit, tuple[str, ...]]]:
    """Лучшие по вопросу рецепты, которым не хватает от 1 до MAX_MISSING продуктов."""
    found = []
    for hit in hits:
        lacking = missing(hit.chunk.ingredients, stock)
        if 1 <= len(lacking) <= MAX_MISSING:
            found.append((hit, lacking))
    return found[:limit]
