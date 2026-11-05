"""Проверка кейсов по датасету: `python -m evals.check_cases check|vocabulary …`.

- `vocabulary <csv>` печатает словарь ингредиентов датасета с частотами
  (TSV): по нему пишутся `expected_include` и `expected_exclude`.
- `check <csv> <yaml> […]` проверяет схему кейсов, имена ингредиентов и
  для кейса с условиями печатает, сколько рецептов датасета подходит под
  все условия сразу. Правила те же, что у метрик поиска
  (`evals/retrieval_metrics.py`): имена ингредиентов целиком в нижнем
  регистре, слово названия — подстрокой без учёта регистра и «ё/е».

Скриптом пользуется субагент, который пишет набор (`evals/dataset/README.md`):
кейс, под который в датасете ничего не подходит, не засчитается никогда.
"""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from assistant.indexing.dataset import Recipe, read_dataset
from evals.dataset import Case
from evals.retrieval_metrics import fits, unknown_ingredients


def vocabulary(recipes: Sequence[Recipe]) -> Counter[str]:
    return Counter(i.name.lower() for r in recipes for i in r.ingredients)


def check(recipes: Sequence[Recipe], items: Sequence[object]) -> tuple[list[str], int]:
    """Строки отчёта и число проблем (ошибки схемы, имена не из словаря)."""
    names = {i.name for r in recipes for i in r.ingredients}
    rows = [(r.name, [i.name.lower() for i in r.ingredients]) for r in recipes]
    lines: list[str] = []
    problems = 0
    for item in items:
        try:
            case = Case.model_validate(item)
        except ValidationError as error:
            problems += 1
            label = item.get("id", "?") if isinstance(item, dict) else "?"
            lines.append(f"СХЕМА {label}: {error.errors()[0]['msg']}")
            continue
        for unknown in unknown_ingredients([case], names):
            problems += 1
            lines.append(f"НЕТ В СЛОВАРЕ {unknown}")
        if case.expected_include or case.expected_exclude or case.expected_name:
            count = sum(fits(case, name, ingredients) for name, ingredients in rows)
            lines.append(f"{case.id}\t{case.category}\tподходит рецептов: {count}")
    return lines, problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.check_cases")
    commands = parser.add_subparsers(dest="command", required=True)
    vocab = commands.add_parser("vocabulary", help="словарь ингредиентов (TSV)")
    vocab.add_argument("csv", type=Path)
    checking = commands.add_parser("check", help="проверить кейсы по датасету")
    checking.add_argument("csv", type=Path)
    checking.add_argument("yaml", type=Path, nargs="+")
    args = parser.parse_args(argv)

    recipes = read_dataset(args.csv).recipes
    if args.command == "vocabulary":
        print("имя\tрецептов")
        for name, count in vocabulary(recipes).most_common():
            print(f"{name}\t{count}")
        return 0
    items: list[object] = []
    for file in args.yaml:
        items += yaml.safe_load(file.read_text(encoding="utf-8")) or []
    lines, problems = check(recipes, items)
    print(*lines, f"проблем: {problems}", sep="\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
