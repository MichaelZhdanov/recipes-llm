from collections.abc import Callable

import httpx2
import pytest

from assistant.http import request_with_retries


def make_send(
    responses: list[httpx2.Response | Exception],
) -> tuple[list[int], Callable[[], httpx2.Response]]:
    calls: list[int] = []

    def send() -> httpx2.Response:
        calls.append(1)
        item = responses[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    return calls, send


def test_retries_on_429_using_retry_after() -> None:
    delays: list[float] = []
    calls, send = make_send(
        [httpx2.Response(429, headers={"Retry-After": "7"}), httpx2.Response(200)]
    )

    response = request_with_retries(send, sleep=delays.append)

    assert response.status_code == 200
    assert len(calls) == 2
    assert delays == [7.0]


def test_retries_server_errors_and_network_errors_with_backoff() -> None:
    delays: list[float] = []
    calls, send = make_send(
        [
            httpx2.ConnectError("сеть недоступна"),
            httpx2.Response(503),
            httpx2.Response(200),
        ]
    )

    response = request_with_retries(send, base_delay=1.0, sleep=delays.append)

    assert response.status_code == 200
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_does_not_retry_client_errors() -> None:
    calls, send = make_send([httpx2.Response(401), httpx2.Response(200)])

    response = request_with_retries(send, sleep=lambda _: None)

    assert response.status_code == 401
    assert len(calls) == 1


def test_returns_last_response_when_attempts_are_exhausted() -> None:
    calls, send = make_send([httpx2.Response(503)] * 3)

    response = request_with_retries(send, attempts=3, sleep=lambda _: None)

    assert response.status_code == 503
    assert len(calls) == 3


def test_raises_network_error_after_last_attempt() -> None:
    _, send = make_send([httpx2.ConnectError("нет сети")] * 2)

    with pytest.raises(httpx2.ConnectError):
        request_with_retries(send, attempts=2, sleep=lambda _: None)


def test_retries_overloaded_529() -> None:
    """529 — «перегружен» у Anthropic: временная ошибка, как 503."""
    calls, send = make_send([httpx2.Response(529), httpx2.Response(200)])

    assert request_with_retries(send, sleep=lambda _: None).status_code == 200
    assert len(calls) == 2
