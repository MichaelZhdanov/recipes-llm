"""Датасет рецептов povarenok: скачивание на зафиксированной ревизии и разбор.

Файл один — CSV из репозитория `DATASET_REPO` на HuggingFace, поля `url`,
`name`, `ingredients`. Ревизия (коммит репозитория) берётся из настроек
`DATASET_REVISION`, а sha256 файла на этой ревизии записан здесь: так видно,
если файл скачался не целиком или на ревизии лежит не тот файл.

`ingredients` — словарь Python строкой (`{'Молоко': '250 мл', 'Соль': None}`),
он разбирается `ast.literal_eval`: тот принимает только литералы и код
не выполняет. Плохие строки пропускаются, каждая причина считается
отдельно: эти числа попадают в паспорт индекса.

Рецепт здесь — лёгкий dataclass, а не модель pydantic: в памяти лежат все
146 тыс. рецептов сразу (около 120 МБ), а на сервере 2 ГБ RAM.
"""

import ast
import csv
import hashlib
import re
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx2

from assistant.http import request_with_retries

DATASET_REPO = "rogozinushka/povarenok-recipes"
DATASET_FILE = "povarenok_recipes_2021_06_16.csv"
# sha256 файла на известных ревизиях. Новая ревизия — новая строка здесь,
# вместе с прогоном eval на ней.
DATASET_SHA256 = {
    "ffaf49ee7439afe6dbf0653714ceacb9b57ff830": (
        "48b14908deae57bd4e3ff2ed7603216963d8376569c14c0a9270be5572b414ab"
    ),
}
TIMEOUT = 120.0

# Причины, по которым строка не становится рецептом.
UNPARSED = "ingredients не разбираются"
NO_INGREDIENTS = "нет ингредиентов"
JUNK_NAME = "вместо названия навигация сайта"
BAD_URL = "url не страница рецепта"

_RECIPE_URL = re.compile(r"^https://www\.povarenok\.ru/recipes/show/(\d+)/?$")
# Артефакт сбора с сайта: у 610 рецептов вместо названия хлебные крошки
# «Главная > Рецепты > версия для печати».
_JUNK_NAME = re.compile(r"^Главная\s*>\s*Рецепты", re.IGNORECASE)


class DatasetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Ingredient:
    name: str
    # «250 мл», «по вкусу»; None — количество на сайте не указано.
    amount: str | None


@dataclass(frozen=True, slots=True)
class Recipe:
    id: int
    url: str
    name: str
    ingredients: tuple[Ingredient, ...]


@dataclass(frozen=True)
class Dataset:
    recipes: list[Recipe]
    # Сколько строк пропущено и почему.
    skipped: Counter[str]
    sha256: str


def known_sha256(revision: str) -> str:
    if revision not in DATASET_SHA256:
        raise DatasetError(
            f"для ревизии {revision} нет sha256 файла: добавьте его в DATASET_SHA256 "
            "(assistant/indexing/dataset.py)"
        )
    return DATASET_SHA256[revision]


def dataset_url(revision: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{revision}/{DATASET_FILE}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dataset(
    revision: str, data_path: Path, http: httpx2.Client, *, sha256: str
) -> Path:
    """Путь к CSV ревизии. Скачивает, только если файла нет или он не тот.

    Файл сначала пишется рядом с суффиксом `.part` и переименовывается,
    только когда sha256 совпал: оборванная загрузка не выглядит готовым файлом.
    """
    target = data_path / revision / DATASET_FILE
    if target.exists() and file_sha256(target) == sha256:
        return target
    response = request_with_retries(
        lambda: http.get(dataset_url(revision), follow_redirects=True, timeout=TIMEOUT)
    )
    if response.status_code != 200:
        raise DatasetError(
            f"HuggingFace ответил HTTP {response.status_code} на {dataset_url(revision)}"
        )
    got = hashlib.sha256(response.content).hexdigest()
    if got != sha256:
        raise DatasetError(
            f"sha256 скачанного файла {got[:12]}… не совпал с ожидаемым {sha256[:12]}…: "
            "файл скачался не целиком или ревизия не та"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    partial.write_bytes(response.content)
    partial.replace(target)
    return target


def _ingredients(raw: str) -> tuple[Ingredient, ...] | None:
    """Ингредиенты из строки-словаря или None, если строка не разбирается."""
    try:
        parsed = ast.literal_eval(raw)
    except ValueError, SyntaxError, MemoryError, RecursionError:
        return None
    if not isinstance(parsed, dict):
        return None
    result = []
    for name, amount in parsed.items():
        clean = " ".join(str(name).split())
        if not clean:
            continue
        value = " ".join(str(amount).split()) if amount is not None else ""
        # Имена и количества сильно повторяются («Соль» — 78 тыс. раз):
        # intern хранит одинаковые строки один раз.
        result.append(
            Ingredient(sys.intern(clean), sys.intern(value) if value else None)
        )
    return tuple(result)


def parse_row(row: dict[str, str]) -> Recipe | str:
    """Рецепт из строки CSV или причина, по которой строка пропущена."""
    match = _RECIPE_URL.match(row.get("url", "").strip())
    if match is None:
        return BAD_URL
    # Пробелы, неразрывные пробелы и переводы строк внутри названия — один пробел.
    name = " ".join((row.get("name") or "").split())
    if not name or _JUNK_NAME.match(name):
        return JUNK_NAME
    ingredients = _ingredients(row.get("ingredients") or "")
    if ingredients is None:
        return UNPARSED
    if not ingredients:
        return NO_INGREDIENTS
    return Recipe(id=int(match[1]), url=match[0], name=name, ingredients=ingredients)


def iter_rows(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def read_dataset(path: Path) -> Dataset:
    recipes: list[Recipe] = []
    skipped: Counter[str] = Counter()
    for row in iter_rows(path):
        parsed = parse_row(row)
        if isinstance(parsed, Recipe):
            recipes.append(parsed)
        else:
            skipped[parsed] += 1
    return Dataset(recipes=recipes, skipped=skipped, sha256=file_sha256(path))
