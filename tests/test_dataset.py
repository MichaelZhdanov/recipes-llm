import csv
import hashlib
import io
from pathlib import Path

import httpx2
import pytest

from assistant.indexing.dataset import (
    BAD_URL,
    DATASET_FILE,
    JUNK_NAME,
    NO_INGREDIENTS,
    UNPARSED,
    DatasetError,
    Ingredient,
    Recipe,
    dataset_url,
    ensure_dataset,
    known_sha256,
    parse_row,
    read_dataset,
)

REVISION = "abc123"
ROWS = [
    # обычный рецепт
    (
        "https://www.povarenok.ru/recipes/show/164365/",
        "Густой молочно-клубничный коктейль",
        "{'Молоко': '250 мл', 'Клубника': '200 г', 'Сахар': None}",
    ),
    # неразрывные пробелы и перевод строки в названии
    (
        "https://www.povarenok.ru/recipes/show/10625/",
        'Салат\xa0"Баклажанчик"\n',
        "{'Баклажан': '3 шт', ' Чеснок ': '2  зуб.'}",
    ),
    # артефакт сбора: навигация вместо названия
    (
        "https://www.povarenok.ru/recipes/show/74610/",
        "Главная\xa0\xa0>\xa0\n  Рецепты\n  >\xa0версия для печати",
        "{'Чернослив': '150 г'}",
    ),
    ("https://www.povarenok.ru/recipes/show/7/", "Пустой", "{}"),
    ("https://www.povarenok.ru/recipes/show/8/", "Сломанный", ""),
    ("https://www.povarenok.ru/recipes/show/9/", "Не словарь", "['Соль']"),
    ("https://example.com/recipe/1", "Чужой сайт", "{'Соль': None}"),
]


def write_csv(path: Path, rows: list[tuple[str, str, str]] = ROWS) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["url", "name", "ingredients"])
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    path.write_bytes(data)
    return data


def test_rows_become_recipes_and_bad_rows_are_counted(tmp_path: Path) -> None:
    path = tmp_path / DATASET_FILE
    data = write_csv(path)

    dataset = read_dataset(path)

    assert dataset.recipes == [
        Recipe(
            id=164365,
            url="https://www.povarenok.ru/recipes/show/164365/",
            name="Густой молочно-клубничный коктейль",
            ingredients=(
                Ingredient("Молоко", "250 мл"),
                Ingredient("Клубника", "200 г"),
                Ingredient("Сахар", None),
            ),
        ),
        Recipe(
            id=10625,
            url="https://www.povarenok.ru/recipes/show/10625/",
            name='Салат "Баклажанчик"',
            ingredients=(
                Ingredient("Баклажан", "3 шт"),
                Ingredient("Чеснок", "2 зуб."),
            ),
        ),
    ]
    assert dataset.skipped == {JUNK_NAME: 1, NO_INGREDIENTS: 1, UNPARSED: 2, BAD_URL: 1}
    assert dataset.sha256 == hashlib.sha256(data).hexdigest()


def test_ingredients_are_parsed_as_literals_not_code() -> None:
    """literal_eval не выполняет код: вызов функции в строке — просто плохая строка."""
    row = {
        "url": "https://www.povarenok.ru/recipes/show/1/",
        "name": "Ловушка",
        "ingredients": "__import__('os').system('echo hacked')",
    }

    assert parse_row(row) == UNPARSED


def test_known_revision_has_sha256_and_unknown_is_explained() -> None:
    assert len(known_sha256("ffaf49ee7439afe6dbf0653714ceacb9b57ff830")) == 64
    with pytest.raises(DatasetError, match="DATASET_SHA256"):
        known_sha256("deadbeef")


def serving(data: bytes, requests: list[str]) -> httpx2.Client:
    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(str(request.url))
        return httpx2.Response(200, content=data)

    return httpx2.Client(transport=httpx2.MockTransport(handle))


def test_download_checks_sha256_then_keeps_the_file(tmp_path: Path) -> None:
    data = write_csv(tmp_path / "source.csv")
    sha = hashlib.sha256(data).hexdigest()
    requests: list[str] = []

    path = ensure_dataset(
        REVISION, tmp_path / "data", serving(data, requests), sha256=sha
    )
    again = ensure_dataset(
        REVISION, tmp_path / "data", serving(data, requests), sha256=sha
    )

    assert path == again == tmp_path / "data" / REVISION / DATASET_FILE
    assert path.read_bytes() == data
    # Второй раз файл уже на месте и sha256 совпал: в сеть не ходим.
    assert requests == [dataset_url(REVISION)]
    assert "/resolve/abc123/" in requests[0]


def test_wrong_sha256_leaves_no_file(tmp_path: Path) -> None:
    data = write_csv(tmp_path / "source.csv")

    with pytest.raises(DatasetError, match="sha256"):
        ensure_dataset(REVISION, tmp_path / "data", serving(data, []), sha256="0" * 64)

    assert not (tmp_path / "data" / REVISION / DATASET_FILE).exists()
    assert not list((tmp_path / "data").rglob("*.part"))


def test_http_error_is_clear(tmp_path: Path) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404)

    client = httpx2.Client(transport=httpx2.MockTransport(handle))

    with pytest.raises(DatasetError, match="HTTP 404"):
        ensure_dataset(REVISION, tmp_path, client, sha256="0" * 64)
