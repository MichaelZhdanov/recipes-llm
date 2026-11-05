"""Проверка кейсов по датасету: `python -m evals.check_cases check|vocabulary|canaries …`.

- `vocabulary <csv>` печатает словарь ингредиентов датасета с частотами
  (TSV): по нему пишутся `expected_include` и `expected_exclude`.
- `check <csv> <yaml> […]` проверяет схему кейсов, имена ингредиентов и
  для кейса с условиями печатает, сколько рецептов датасета подходит под
  все условия сразу. Правила те же, что у метрик поиска
  (`evals/retrieval_metrics.py`): имена ингредиентов целиком в нижнем
  регистре, слово названия — подстрокой без учёта регистра и «ё/е».

- `canaries <csv> <каталог>` проверяет канарейки судьи (`judge/`): схему,
  уникальность id, что `quote` — строки одного настоящего рецепта (так их
  увидит судья), что рецепт канарейки релевантности есть в датасете, и
  печатает баланс вердиктов.

Скриптом пользуется субагент, который пишет набор (`evals/dataset/README.md`):
кейс, под который в датасете ничего не подходит, не засчитается никогда.
"""

import argparse
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from assistant.generation.citations import line_key
from assistant.indexing.chunks import recipe_text
from assistant.indexing.dataset import Recipe, read_dataset
from evals.dataset import Canary, Case, ClarityCanary, RelevanceCanary
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


def _lines_index(recipes: Sequence[Recipe]) -> dict[str, set[int]]:
    """Строка рецепта (как её сравнивает проверка цитат) → id рецептов с ней."""
    index: dict[str, set[int]] = {}
    for recipe in recipes:
        for line in recipe_text(recipe).split("\n"):
            index.setdefault(line_key(line), set()).add(recipe.id)
    return index


def check_canaries(
    recipes: Sequence[Recipe], files: Mapping[str, Sequence[object]]
) -> tuple[list[str], int]:
    """Канарейки судьи: files — содержимое canaries, clarity_canaries и
    relevance_canaries (по имени файла без .yaml). Строки отчёта и число проблем."""
    lines: list[str] = []
    problems = 0
    index = _lines_index(recipes)
    by_name: dict[str, list[set[str]]] = {}
    for recipe in recipes:
        by_name.setdefault(line_key(recipe.name), []).append(
            {i.name.lower() for i in recipe.ingredients}
        )
    models: dict[str, type[Canary] | type[ClarityCanary] | type[RelevanceCanary]] = {
        "canaries": Canary,
        "clarity_canaries": ClarityCanary,
        "relevance_canaries": RelevanceCanary,
    }
    for file, items in files.items():
        verdicts: Counter[bool] = Counter()
        seen: set[str] = set()
        for item in items:
            label = item.get("id", "?") if isinstance(item, dict) else "?"
            try:
                canary = models[file].model_validate(item)
            except ValidationError as error:
                problems += 1
                lines.append(f"СХЕМА {file}/{label}: {error.errors()[0]['msg']}")
                continue
            if canary.id in seen:
                problems += 1
                lines.append(f"ПОВТОР {file}/{canary.id}")
            seen.add(canary.id)
            if isinstance(canary, RelevanceCanary):
                verdicts[canary.relevant] += 1
                wanted = set(canary.ingredients)
                if not any(
                    wanted <= have for have in by_name.get(line_key(canary.name), [])
                ):
                    problems += 1
                    lines.append(
                        f"НЕТ РЕЦЕПТА {canary.id}: «{canary.name}» с такими ингредиентами"
                    )
                continue
            verdicts[
                canary.supported if isinstance(canary, Canary) else canary.clear
            ] += 1
            owners = [
                index.get(line_key(line), set()) for line in canary.quote.split("\n")
            ]
            if not owners or not set.intersection(*owners):
                problems += 1
                lines.append(
                    f"НЕ СТРОКИ РЕЦЕПТА {canary.id}: quote не из строк одного рецепта"
                )
        lines.append(
            f"{file}: {len(seen)} шт., да {verdicts[True]}, нет {verdicts[False]}"
        )
    return lines, problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.check_cases")
    commands = parser.add_subparsers(dest="command", required=True)
    vocab = commands.add_parser("vocabulary", help="словарь ингредиентов (TSV)")
    vocab.add_argument("csv", type=Path)
    checking = commands.add_parser("check", help="проверить кейсы по датасету")
    checking.add_argument("csv", type=Path)
    checking.add_argument("yaml", type=Path, nargs="+")
    canaries = commands.add_parser("canaries", help="проверить канарейки судьи")
    canaries.add_argument("csv", type=Path)
    canaries.add_argument("directory", type=Path, help="каталог judge/ с YAML")
    args = parser.parse_args(argv)

    recipes = read_dataset(args.csv).recipes
    if args.command == "vocabulary":
        print("имя\tрецептов")
        for name, count in vocabulary(recipes).most_common():
            print(f"{name}\t{count}")
        return 0
    if args.command == "canaries":
        files = {
            path.stem: yaml.safe_load(path.read_text(encoding="utf-8")) or []
            for path in sorted(args.directory.glob("*canaries.yaml"))
        }
        lines, problems = check_canaries(recipes, files)
        print(*lines, f"проблем: {problems}", sep="\n")
        return 1 if problems else 0
    items: list[object] = []
    for file in args.yaml:
        items += yaml.safe_load(file.read_text(encoding="utf-8")) or []
    lines, problems = check(recipes, items)
    print(*lines, f"проблем: {problems}", sep="\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
