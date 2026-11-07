"""Демо-склад Grocy: `make grocy-seed` (сам запускается в `make up` и деплое).

Идемпотентно — повторный запуск досоздаёт только то, чего нет:
1. ждёт Grocy; первый запрос к `/` создаёт его базу (миграции), без него
   вход отвечает 500;
2. входит как admin; если пароль ещё стандартный `admin`, меняет его на
   GROCY_ADMIN_PASSWORD;
3. заводит пользователя `assistant` и очищает его права: новым
   пользователям Grocy даёт ADMIN (DEFAULT_PERMISSIONS);
4. его ключ — значение GROCY_API_KEY. Через API ключ с заданным значением
   не создать (`api_keys` в Grocy только для чтения, ExposedEntityNoEdit),
   поэтому ключ — строкой в таблицу `api_keys` базы Grocy (том `grocy-data`
   смонтирован в контейнер сида);
5. продукты демо-склада и их остатки — через API: имена — из словаря индекса,
   чтобы склад сопоставлялся с рецептами без модели;
6. самопроверка: ключ читает склад, а запись им получает отказ — иначе сид
   падает, и деплой вместе с ним.

Секреты — из окружения (их генерирует `scripts/init_env.py`); пароль
администратора в настройки ассистента не попадает.
"""

import os
import sqlite3
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2

from assistant.grocy.client import KEY_HEADER

DEFAULT_PASSWORD = "admin"
ASSISTANT_USER = "assistant"
KEY_DESCRIPTION = "recipes-llm: только чтение"
KEY_EXPIRES = "2999-12-31 23:59:59"
WAIT_SECONDS = 120

# Демо-«домохозяйство» — список продуктов на неделю от автора (2026-11-07).
# Название продукта в Grocy — имя из словаря индекса рецептов: так склад
# сопоставляется с рецептами без модели. Формулировка автора — в описании.
# Нет в словаре (ни в одном рецепте датасета) и потому не заведены: прошутто,
# салат корн, «гавайская» овощная смесь, чиабатта.


@dataclass(frozen=True)
class Item:
    name: str
    amount: float
    unit: str
    location: str
    note: str = ""


FRIDGE, FRESH, FREEZER = "Холодильник", "Зона свежести", "Морозилка"
PANTRY, BREAD = "Кухонные шкафы", "Хлебный ящик"
# Место хранения → морозилка ли (поле is_freezer в Grocy).
LOCATIONS = {FRIDGE: False, FRESH: False, FREEZER: True, PANTRY: False, BREAD: False}
UNITS = ("шт", "г", "кг", "л", "упаковка", "бутылка", "банка", "буханка")

DEMO_STOCK: tuple[Item, ...] = (
    # Свежие продукты
    Item("яйцо куриное", 20, "шт", FRIDGE, "Яйца куриные С0"),
    Item("яйцо перепелиное", 10, "шт", FRIDGE, "Яйца перепелиные"),
    Item("молоко", 3, "л", FRIDGE, "Молоко цельное пастеризованное"),
    Item("молоко миндальное", 1.5, "л", FRIDGE, "Растительное молоко"),
    Item(
        "масло сливочное", 360, "г", FRIDGE, "Сливочное масло 82,5%, 2 пачки по 180 г"
    ),
    Item("пармезан", 300, "г", FRIDGE, "Твёрдый сыр для пасты"),
    Item("сыр полутвердый", 400, "г", FRIDGE, "Полутвёрдый сыр на бутерброды"),
    Item("сыр творожный", 200, "г", FRIDGE, "Творожный сыр"),
    Item("йогурт", 1000, "г", FRIDGE, "Греческий йогурт"),
    Item("творог", 800, "г", FRIDGE, "Творог"),
    Item("сметана", 400, "г", FRIDGE, "Сметана"),
    Item("семга", 300, "г", FRIDGE, "Слабосолёная сёмга, нарезка"),
    Item("форель", 200, "г", FRIDGE, "Слабосолёная форель, нарезка"),
    Item("индейка", 300, "г", FRIDGE, "Нарезка из индейки"),
    # Овощи, фрукты, зелень
    Item("шпинат", 2, "упаковка", FRESH, "Готовая мытая зелень"),
    Item("руккола", 2, "упаковка", FRESH, "Готовая мытая зелень"),
    Item("помидоры черри", 1, "кг", FRESH, "Томаты черри"),
    Item("огурец", 1.5, "кг", FRESH, "Огурцы короткоплодные"),
    Item("авокадо", 4, "шт", FRESH, "Авокадо"),
    Item("перец болгарский", 1, "кг", FRESH, "Перец"),
    Item("картофель", 2, "кг", FRESH, "Картофель"),
    Item("лук репчатый", 1, "кг", FRESH, "Лук"),
    Item("морковь", 0.7, "кг", FRESH, "Морковь"),
    Item("чеснок", 0.2, "кг", FRESH, "Чеснок"),
    Item("голубика", 2, "упаковка", FRESH, "Свежие ягоды"),
    Item("малина", 1, "упаковка", FRESH, "Свежие ягоды"),
    Item("лимон", 0.5, "кг", FRESH, "Лимоны"),
    Item("банан", 1.5, "кг", FRESH, "Бананы"),
    Item("яблоко", 1.5, "кг", FRESH, "Яблоки"),
    Item("апельсин", 1, "кг", FRESH, "Сезонные цитрусовые"),
    Item("манго", 0.8, "кг", FRESH, "Манго"),
    # Мясо и рыба — охлаждённое
    Item("грудка индюшки", 1.5, "кг", FRIDGE, "Филе грудки индейки"),
    Item("бедро куриное", 1.5, "кг", FRIDGE, "Бёдра куриные"),
    Item("говядина", 1, "кг", FRIDGE, "Говяжьи стейки"),
    Item("фарш мясной", 0.8, "кг", FRIDGE, "Нежирный фарш"),
    Item("лосось", 0.8, "кг", FRIDGE, "Стейки лосося"),
    Item("треска", 0.6, "кг", FRIDGE, "Стейки трески"),
    Item("креветки", 0.5, "кг", FRIDGE, "Очищенные креветки"),
    # Заморозка
    Item("брокколи", 2, "упаковка", FREEZER, "Замороженная брокколи, 400 г"),
    Item(
        "фасоль стручковая",
        2,
        "упаковка",
        FREEZER,
        "Замороженная стручковая фасоль, 400 г",
    ),
    # Бакалея
    Item("киноа", 1, "упаковка", PANTRY, "Киноа"),
    Item("рис коричневый", 1, "упаковка", PANTRY, "Бурый рис"),
    Item("хлопья овсяные", 1, "упаковка", PANTRY, "Овсяные хлопья долгой варки"),
    Item("крупа гречневая", 1, "упаковка", PANTRY, "Гречка"),
    Item("спагетти", 2, "упаковка", PANTRY, "Паста из твёрдых сортов"),
    Item("макаронные изделия", 1, "упаковка", PANTRY, "Безглютеновая паста"),
    Item("масло оливковое", 1, "бутылка", PANTRY, "Оливковое холодного отжима"),
    Item("масло кокосовое", 1, "банка", PANTRY, "Кокосовое масло"),
    Item("масло подсолнечное", 1, "бутылка", PANTRY, "Подсолнечное масло"),
    Item("песто", 1, "банка", PANTRY, "Соус песто"),
    Item("соевый соус", 1, "бутылка", PANTRY, "Соевый соус"),
    Item("бальзамик", 1, "бутылка", PANTRY, "Бальзамический крем"),
    Item("томаты в собственном соку", 2, "банка", PANTRY, "Томаты в собственном соку"),
    Item("миндаль", 300, "г", PANTRY, "Миндаль"),
    Item("орехи грецкие", 300, "г", PANTRY, "Грецкие орехи"),
    Item("орехи кешью", 200, "г", PANTRY, "Кешью"),
    Item("финик", 200, "г", PANTRY, "Финики"),
    Item("тунец", 3, "банка", PANTRY, "Тунец для салатов"),
    Item("нут", 2, "банка", PANTRY, "Нут"),
    Item("фасоль консервированная", 2, "банка", PANTRY, "Белая фасоль"),
    # Чай, кофе, сладости
    Item("кофе натуральный", 1, "кг", PANTRY, "Кофе в зёрнах"),
    Item("чай черный", 1, "упаковка", PANTRY, "Листовой чай"),
    Item("чай зеленый", 1, "упаковка", PANTRY, "Травяные сборы и зелёный чай"),
    Item("матча", 1, "упаковка", PANTRY, "Матча"),
    Item("шоколад темный", 300, "г", PANTRY, "Горький шоколад 75%+"),
    Item("мед", 300, "г", PANTRY, "Мёд"),
    Item("пастила", 200, "г", PANTRY, "Натуральная пастила"),
    # Хлебный ящик
    Item("хлеб", 2, "буханка", BREAD, "Бездрожжевой на закваске, цельнозерновой"),
    Item("багет", 1, "шт", BREAD, "Багет"),
)


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"не задан {name}: его создаёт scripts/init_env.py (make up)")
    return value


def wait_for_grocy(client: httpx2.Client, seconds: float = WAIT_SECONDS) -> None:
    """Первый запрос к / создаёт базу Grocy; ждём, пока он пройдёт."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            if client.get("/").status_code < 400:
                return
        except httpx2.TransportError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit("Grocy не ответил: docker compose logs grocy")
        time.sleep(2)


def login(client: httpx2.Client, password: str) -> bool:
    """Вход в сессию; при неудаче Grocy уводит обратно на /login."""
    client.get("/login")
    response = client.post("/login", data={"username": "admin", "password": password})
    return response.status_code == 302 and "login" not in response.headers.get(
        "location", ""
    )


def admin_session(client: httpx2.Client, password: str) -> None:
    if login(client, password):
        return
    if not login(client, DEFAULT_PASSWORD):
        raise SystemExit("не вышло войти в Grocy как admin: пароль не тот")
    admin = next(u for u in _get(client, "/api/users") if u["username"] == "admin")
    _check(
        client.put(
            f"/api/users/{admin['id']}",
            # last_name обязателен: без него Grocy 4.7 отвечает 500 (EditUser).
            json={
                "username": "admin",
                "first_name": "admin",
                "last_name": "",
                "password": password,
            },
        ),
        "смена пароля admin",
    )
    print("пароль admin заменён на GROCY_ADMIN_PASSWORD")


def assistant_user(client: httpx2.Client) -> int:
    """Пользователь без прав: его ключ только читает."""
    users = _get(client, "/api/users")
    found = next((u for u in users if u["username"] == ASSISTANT_USER), None)
    if found is None:
        _check(
            client.post(
                "/api/users",
                json={
                    "username": ASSISTANT_USER,
                    "first_name": "Ассистент рецептов",
                    "last_name": "",
                    # Входить им не нужно: пароль случайный и нигде не хранится.
                    "password": os.urandom(24).hex(),
                },
            ),
            "создание пользователя",
        )
        found = next(
            u for u in _get(client, "/api/users") if u["username"] == ASSISTANT_USER
        )
    user_id = int(found["id"])
    _check(
        client.put(f"/api/users/{user_id}/permissions", json={"permissions": []}),
        "очистка прав",
    )
    return user_id


def install_key(database: Path, user_id: int, key: str) -> None:
    """Ключ assistant = GROCY_API_KEY: прежний ключ сида заменяется (ротация)."""
    with sqlite3.connect(database, timeout=10) as db:
        db.execute(
            "DELETE FROM api_keys WHERE user_id = ? AND description = ?",
            (user_id, KEY_DESCRIPTION),
        )
        db.execute(
            "INSERT INTO api_keys (api_key, user_id, expires, key_type, description) "
            "VALUES (?, ?, ?, 'default', ?)",
            (key, user_id, KEY_EXPIRES, KEY_DESCRIPTION),
        )


def ensure_named(
    client: httpx2.Client, entity: str, fields: Mapping[str, Mapping[str, object]]
) -> dict[str, int]:
    """Места хранения и единицы по имени: нет — создать; имя → id."""
    existing = {
        str(o["name"]): int(o["id"]) for o in _get(client, f"/api/objects/{entity}")
    }
    for name, extra in fields.items():
        if name not in existing:
            created = _check(
                client.post(f"/api/objects/{entity}", json={"name": name, **extra}),
                f"{entity}: {name}",
            ).json()["created_object_id"]
            existing[name] = int(created)
    return existing


def stock_products(client: httpx2.Client, stock: Sequence[Item]) -> int:
    """Продукты, которых ещё нет, и их остатки; уже заведённые не трогаются."""
    locations = ensure_named(
        client,
        "locations",
        {name: {"is_freezer": int(freezer)} for name, freezer in LOCATIONS.items()},
    )
    units = ensure_named(
        client, "quantity_units", {unit: {"name_plural": unit} for unit in UNITS}
    )
    existing = {p["name"] for p in _get(client, "/api/objects/products")}
    added = 0
    for item in stock:
        if item.name in existing:
            continue
        created = _check(
            client.post(
                "/api/objects/products",
                json={
                    "name": item.name,
                    "description": item.note,
                    "location_id": locations[item.location],
                    "qu_id_purchase": units[item.unit],
                    "qu_id_stock": units[item.unit],
                },
            ),
            f"продукт {item.name}",
        ).json()["created_object_id"]
        _check(
            client.post(
                f"/api/stock/products/{created}/add", json={"amount": item.amount}
            ),
            f"остаток {item.name}",
        )
        added += 1
    return added


def self_check(client: httpx2.Client) -> None:
    """Ключ (в заголовках client) читает склад, а запись им — отказ:
    «пишущих вызовов ноль» и со стороны Grocy."""
    read = client.get("/api/stock")
    if read.status_code != 200:
        raise SystemExit(f"ключ ассистента не читает склад: HTTP {read.status_code}")
    products = client.get("/api/objects/products").json()
    write = client.post(
        f"/api/stock/products/{products[0]['id']}/add", json={"amount": 1}
    )
    if write.status_code != 403:
        raise SystemExit(
            f"ключ ассистента может писать: HTTP {write.status_code} — сид остановлен"
        )
    print(f"ключ ассистента: склад читает ({len(read.json())} продуктов), запись — 403")


def _get(client: httpx2.Client, path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = _check(client.get(path), path).json()
    return rows


def _check(response: httpx2.Response, what: str) -> httpx2.Response:
    if response.status_code >= 400:
        raise SystemExit(f"{what}: HTTP {response.status_code} {response.text[:200]}")
    return response


def main() -> int:
    base_url = os.environ.get("GROCY_URL", "http://grocy")
    database = Path(os.environ.get("GROCY_DB", "/grocy/data/grocy.db"))
    key, password = _env("GROCY_API_KEY"), _env("GROCY_ADMIN_PASSWORD")
    with httpx2.Client(base_url=base_url, timeout=30) as client:
        wait_for_grocy(client)
        admin_session(client, password)
        user_id = assistant_user(client)
        install_key(database, user_id, key)
        added = stock_products(client, DEMO_STOCK)
    print(f"демо-склад: добавлено продуктов {added}, всего в списке {len(DEMO_STOCK)}")
    with httpx2.Client(base_url=base_url, headers={KEY_HEADER: key}, timeout=20) as k:
        self_check(k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
