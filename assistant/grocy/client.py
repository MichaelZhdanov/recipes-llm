"""Grocy — внешняя система учёта продуктов, только на чтение (этап 6, F4).

Адаптер умеет ровно один запрос — `GET /stock` (что сейчас есть на складе).
Других методов у класса нет, поэтому пишущих вызовов к Grocy не бывает по
построению; тест это проверяет. Вторая защита — ключ: он принадлежит
пользователю Grocy без прав (`seed.py`), и любая запись им получает 403
«Permission missing» — проверено вживую (CLAUDE.md, «Проверено: Grocy»).

Названия продуктов — данные, а не инструкции. Они маскируются (`mask_pii`),
как всё, что приходит из внешних систем, а до поиска доходят только имена
из словаря индекса (`stock.py`): «игнорируй правила» в названии продукта
модель не увидит.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx2
from pydantic import SecretStr

from assistant.http import request_with_retries
from assistant.security.pii import mask_pii

TIMEOUT = 10.0
# Заголовок ключа API Grocy (app.php, OpenAPI securitySchemes).
KEY_HEADER = "GROCY-API-KEY"


class GrocyError(Exception):
    """Grocy не ответил или ответил не так: склад сейчас не посмотреть."""


@dataclass(frozen=True)
class StockItem:
    name: str
    amount: float


class Grocy:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: SecretStr,
        client: httpx2.Client,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise GrocyError("не задан GROCY_API_KEY: его создаёт make grocy-seed")
        self._url = api_url.rstrip("/")
        self._key = api_key
        self._client = client
        self._sleep = sleep

    def stock(self) -> list[StockItem]:
        """Продукты, которые сейчас есть на складе (количество больше нуля)."""
        try:
            response = request_with_retries(
                lambda: self._client.get(
                    f"{self._url}/stock",
                    headers={KEY_HEADER: self._key.get_secret_value()},
                    timeout=TIMEOUT,
                ),
                sleep=self._sleep,
            )
        except httpx2.TransportError as error:
            raise GrocyError(f"Grocy недоступен: {type(error).__name__}") from error
        if response.status_code != 200:
            raise GrocyError(f"Grocy ответил HTTP {response.status_code}")
        try:
            rows = response.json()
            items = [
                StockItem(
                    mask_pii(str(row["product"]["name"])).text.strip(),
                    float(row["amount"]),
                )
                for row in rows
            ]
        except (ValueError, KeyError, TypeError) as error:
            raise GrocyError("Grocy вернул остатки не в том формате") from error
        return [item for item in items if item.name and item.amount > 0]


def make_http_client() -> httpx2.Client:
    return httpx2.Client(timeout=TIMEOUT)
