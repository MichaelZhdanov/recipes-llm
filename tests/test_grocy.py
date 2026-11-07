"""Grocy: адаптер только на чтение (без сети — MockTransport)."""

import inspect
import json

import httpx2
import pytest
from pydantic import SecretStr

from assistant.grocy.client import KEY_HEADER, Grocy, GrocyError, StockItem

URL = "http://grocy.test/api"


def grocy(handler: httpx2.MockTransport) -> tuple[Grocy, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler.handle_request(request)

    client = httpx2.Client(transport=httpx2.MockTransport(record))
    return Grocy(
        api_url=URL, api_key=SecretStr("k-1"), client=client, sleep=lambda s: None
    ), seen


def respond(status: int, body: object) -> httpx2.MockTransport:
    return httpx2.MockTransport(
        lambda request: httpx2.Response(status, content=json.dumps(body))
    )


def test_stock_reads_names_and_amounts_with_the_key_by_get() -> None:
    rows = [
        {"product_id": 1, "amount": 3, "product": {"name": "Молоко "}},
        {"product_id": 2, "amount": 0, "product": {"name": "сметана"}},
        {"product_id": 3, "amount": 1, "product": {"name": "звоните +7 916 123-45-67"}},
    ]
    adapter, seen = grocy(respond(200, rows))

    items = adapter.stock()

    assert items[0] == StockItem("Молоко", 3.0)
    # Нулевой остаток — продукта нет; ПДн из названия маскируются.
    assert [i.name for i in items][1:] and "916" not in items[1].name
    assert len(items) == 2
    (request,) = seen
    assert request.method == "GET" and str(request.url) == f"{URL}/stock"
    assert request.headers[KEY_HEADER] == "k-1"


def test_adapter_can_only_read() -> None:
    """Пишущих вызовов к Grocy нет по построению: у адаптера один метод — stock."""
    public = {
        name
        for name, _ in inspect.getmembers(Grocy, inspect.isfunction)
        if not name.startswith("_")
    }

    assert public == {"stock"}


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (respond(403, {"error_message": "no"}), "HTTP 403"),
        (respond(200, {"not": "a list"}), "не в том формате"),
        (respond(200, [{"amount": 1}]), "не в том формате"),
    ],
)
def test_bad_answers_become_grocy_error(
    transport: httpx2.MockTransport, message: str
) -> None:
    adapter, _ = grocy(transport)

    with pytest.raises(GrocyError, match=message):
        adapter.stock()


def test_unreachable_grocy_and_missing_key_are_grocy_errors() -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("нет связи", request=request)

    adapter, seen = grocy(httpx2.MockTransport(down))
    with pytest.raises(GrocyError, match="недоступен"):
        adapter.stock()
    assert len(seen) > 1  # сетевые ошибки повторяются
    with pytest.raises(GrocyError, match="GROCY_API_KEY"):
        Grocy(api_url=URL, api_key=SecretStr(""), client=httpx2.Client())
