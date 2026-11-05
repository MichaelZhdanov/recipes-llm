"""Условия поиска «с чем и без чего» и их перевод в фильтр Qdrant.

Векторы не понимают «без»: на «курица, но без лука» у векторного поиска
больше половины топа с луком (evals/reports/2026-11-05-models.md).
Поэтому условия проверяет Qdrant по payload до ранжирования:

- `include` — группы альтернатив: в рецепте хотя бы одно имя из **каждой**
  группы. «Курица и рис» — `[[курица, филе куриное, …], [рис]]`;
- `exclude` — ни одного из этих имён.

Имена — как в поле `ingredients` точки: в нижнем регистре, как их привёл
сайт («лук репчатый», «филе куриное»). Сопоставлением слов пользователя с
этими именами занимается разбор вопроса, здесь — только фильтр.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import models

FIELD = "ingredients"


def _names(items: Iterable[str]) -> tuple[str, ...]:
    """Имена в нижнем регистре, без пустых и повторов, в исходном порядке."""
    cleaned = (item.strip().lower() for item in items)
    return tuple(dict.fromkeys(name for name in cleaned if name))


@dataclass(frozen=True)
class IngredientFilter:
    include: tuple[tuple[str, ...], ...] = ()
    exclude: tuple[str, ...] = ()

    @classmethod
    def of(
        cls, include: Iterable[Iterable[str]] = (), exclude: Iterable[str] = ()
    ) -> IngredientFilter:
        """Условия из сырых списков. Пустая группа — не условие, а не «ничего
        не подходит»: модель иногда возвращает `[[]]`, когда ингредиентов нет."""
        groups = (_names(group) for group in include)
        return cls(
            include=tuple(dict.fromkeys(g for g in groups if g)),
            exclude=_names(exclude),
        )

    def __bool__(self) -> bool:
        return bool(self.include or self.exclude)

    def to_qdrant(self) -> models.Filter | None:
        if not self:
            return None
        must: list[models.Condition] = [
            models.FieldCondition(key=FIELD, match=models.MatchAny(any=list(group)))
            for group in self.include
        ]
        must_not: list[models.Condition] = (
            [
                models.FieldCondition(
                    key=FIELD, match=models.MatchAny(any=list(self.exclude))
                )
            ]
            if self.exclude
            else []
        )
        return models.Filter(must=must or None, must_not=must_not or None)
