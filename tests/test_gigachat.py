import hashlib
import json
import ssl
import threading
import time
import uuid
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from assistant.config import GIGACHAT_API_URL, GIGACHAT_OAUTH_URL, Settings
from assistant.generation.gigachat import (
    CA_FILE,
    CA_SHA256,
    DEFAULT_MODEL,
    GigaChat,
    LlmError,
    Message,
    llm_model,
    make_llm,
    ssl_context,
)

KEY = "Z2lnYWNoYXQtdGVzdC1rZXk="
USER: list[Message] = [{"role": "user", "content": "как включить OCR"}]


class FakeGigaChat:
    """Подменяет OAuth и чат GigaChat: запоминает запросы, выдаёт токены t1, t2, …"""

    def __init__(
        self,
        chat_statuses: list[int] | None = None,
        oauth_status: int = 200,
        finish_reason: str = "stop",
        token_lifetime: float = 1800,
        now: float = 1_000_000.0,
    ) -> None:
        self.oauth: list[httpx2.Request] = []
        self.chats: list[httpx2.Request] = []
        self.chat_statuses = chat_statuses or []
        self.oauth_status = oauth_status
        self.finish_reason = finish_reason
        self.token_lifetime = token_lifetime
        self.now = now

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == GIGACHAT_OAUTH_URL:
            self.oauth.append(request)
            if self.oauth_status != 200:
                return httpx2.Response(self.oauth_status, text="bad credentials")
            expires_ms = int((self.now + self.token_lifetime) * 1000)
            return httpx2.Response(
                200,
                json={"access_token": f"t{len(self.oauth)}", "expires_at": expires_ms},
            )
        assert str(request.url) == f"{GIGACHAT_API_URL}/chat/completions"
        self.chats.append(request)
        if self.chat_statuses:
            status = self.chat_statuses.pop(0)
            if status != 200:
                return httpx2.Response(status, text="error")
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok": 1}'},
                        "index": 0,
                        "finish_reason": self.finish_reason,
                    }
                ],
                "model": "GigaChat-2:2.0.28.2",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "precached_prompt_tokens": 30,
                    "total_tokens": 90,
                },
            },
        )

    def chat_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.chats]


def make(fake: FakeGigaChat, key: str = KEY) -> GigaChat:
    return GigaChat(
        api_key=SecretStr(key),
        scope="GIGACHAT_API_PERS",
        api_url=GIGACHAT_API_URL,
        oauth_url=GIGACHAT_OAUTH_URL,
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        clock=lambda: fake.now,
        sleep=lambda _: None,
    )


def chat(llm: GigaChat, messages: list[Message] = USER) -> Any:
    return llm.chat(messages, model="GigaChat-2", temperature=1e-4, max_tokens=500)


def test_certificate_in_repository_is_the_mincifry_root() -> None:
    der = ssl.PEM_cert_to_DER_cert(CA_FILE.read_text())
    fingerprint = hashlib.sha256(der).hexdigest().upper()

    assert ":".join(fingerprint[i : i + 2] for i in range(0, 64, 2)) == CA_SHA256


def test_client_trusts_only_this_root() -> None:
    """Ни системного хранилища, ни certifi: только корень Минцифры."""
    assert ssl_context().cert_store_stats()["x509_ca"] == 1


def test_missing_key_explains_where_to_get_it() -> None:
    with pytest.raises(LlmError, match="GIGACHAT_API_KEY") as error:
        make(FakeGigaChat(), key="")

    assert "developers.sber.ru" in str(error.value)


def test_oauth_request_and_bearer_token() -> None:
    fake = FakeGigaChat()

    chat(make(fake))

    oauth = fake.oauth[0]
    assert oauth.headers["Authorization"] == f"Basic {KEY}"
    uuid.UUID(oauth.headers["RqUID"])
    assert oauth.content == b"scope=GIGACHAT_API_PERS"
    assert fake.chats[0].headers["Authorization"] == "Bearer t1"


def test_token_is_reused_and_refreshed_before_expiry() -> None:
    fake = FakeGigaChat(token_lifetime=1800)
    llm = make(fake)

    chat(llm)
    fake.now += 1700  # до конца срока 100 с: токен ещё годен
    chat(llm)
    assert len(fake.oauth) == 1

    fake.now += 50  # до конца 50 с, меньше запаса в 60 с: пора обновить
    chat(llm)
    assert len(fake.oauth) == 2
    assert fake.chats[-1].headers["Authorization"] == "Bearer t2"


def test_401_gets_new_token_and_retries_once() -> None:
    fake = FakeGigaChat(chat_statuses=[401, 200])

    completion = chat(make(fake))

    assert completion.finish_reason == "stop"
    assert [r.headers["Authorization"] for r in fake.chats] == [
        "Bearer t1",
        "Bearer t2",
    ]


def test_second_401_is_an_error() -> None:
    fake = FakeGigaChat(chat_statuses=[401, 401])

    with pytest.raises(LlmError, match="HTTP 401"):
        chat(make(fake))
    assert len(fake.chats) == 2


def test_bad_key_on_oauth_gives_a_hint() -> None:
    with pytest.raises(LlmError, match="ключ авторизации"):
        chat(make(FakeGigaChat(oauth_status=401)))


def test_out_of_tokens_gives_a_hint() -> None:
    with pytest.raises(LlmError, match="Закончились токены"):
        chat(make(FakeGigaChat(chat_statuses=[402])))


def test_personal_data_is_masked_in_the_request_body() -> None:
    fake = FakeGigaChat()
    llm = make(fake)

    chat(
        llm,
        [
            {"role": "system", "content": "Отвечай по источникам."},
            {
                "role": "user",
                "content": "пишу с ivan.petrov@example.com, тел. +7 912 345-67-89",
            },
        ],
    )

    sent = fake.chats[0].content.decode()
    assert "ivan.petrov@example.com" not in sent
    assert "345-67-89" not in sent
    assert fake.chat_bodies()[0]["messages"][0]["content"] == "Отвечай по источникам."
    assert llm.stats.masked["EMAIL"] == 1


def test_request_parameters_and_parsed_completion() -> None:
    fake = FakeGigaChat(finish_reason="length")
    llm = make(fake)
    schema = {"type": "json_schema", "schema": {"type": "object"}, "strict": True}

    completion = llm.chat(
        USER,
        model="GigaChat-2-Pro",
        temperature=1e-4,
        max_tokens=700,
        response_format=schema,
    )

    body = fake.chat_bodies()[0]
    assert body["model"] == "GigaChat-2-Pro"
    assert body["temperature"] == 1e-4
    assert body["max_tokens"] == 700
    assert body["response_format"] == schema
    assert completion.content == '{"ok": 1}'
    assert completion.finish_reason == "length"
    assert completion.model == "GigaChat-2:2.0.28.2"
    assert completion.usage.precached_prompt_tokens == 30
    assert llm.stats.requests == 1
    assert llm.stats.total_tokens == 90


def test_requests_go_one_at_a_time() -> None:
    """На Freemium GigaChat генерирует в один поток: клиент не шлёт параллельно."""
    active = 0
    peak = 0
    fake = FakeGigaChat()

    def slow(request: httpx2.Request) -> httpx2.Response:
        nonlocal active, peak
        if str(request.url).endswith("/chat/completions"):
            active += 1
            peak = max(peak, active)
            time.sleep(0.02)
            active -= 1
        return fake(request)

    llm = GigaChat(
        api_key=SecretStr(KEY),
        scope="GIGACHAT_API_PERS",
        api_url=GIGACHAT_API_URL,
        oauth_url=GIGACHAT_OAUTH_URL,
        client=httpx2.Client(transport=httpx2.MockTransport(slow)),
        clock=lambda: fake.now,
    )
    threads = [threading.Thread(target=chat, args=(llm,)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(fake.chats) == 4
    assert peak == 1


def test_settings_wire_the_client() -> None:
    settings = Settings(gigachat_api_key=SecretStr(KEY))

    assert llm_model(settings) == DEFAULT_MODEL
    assert llm_model(Settings(llm_model="GigaChat-2-Max")) == "GigaChat-2-Max"
    with httpx2.Client() as client:
        assert isinstance(make_llm(settings, client), GigaChat)


def test_busy_single_stream_is_waited_out() -> None:
    """429 на Freemium — второй запрос в один поток: ждём дольше, чем у других API."""
    fake = FakeGigaChat(chat_statuses=[429, 429, 429, 429, 200])
    sleeps: list[float] = []
    llm = GigaChat(
        api_key=SecretStr(KEY),
        scope="GIGACHAT_API_PERS",
        api_url=GIGACHAT_API_URL,
        oauth_url=GIGACHAT_OAUTH_URL,
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        clock=lambda: fake.now,
        sleep=sleeps.append,
    )

    completion = chat(llm)

    assert completion.finish_reason == "stop"
    assert sleeps == [1.0, 2.0, 4.0, 8.0]
