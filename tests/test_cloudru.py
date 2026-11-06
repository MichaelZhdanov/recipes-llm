"""Адаптер чата cloud.ru: формат OpenAI, рассуждения отдельно от ответа, ПДн."""

import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from assistant.config import CLOUD_RU_API_URL
from assistant.generation.cloudru import REASONING_TOKENS, CloudRu, cost_rub
from assistant.generation.gigachat import LlmError, LlmStats, Message

REPLY = {
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "choices": [
        {
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"supported": false}',
                "reasoning_content": "В рецепте нет яйца…",
            },
        }
    ],
    "usage": {
        "prompt_tokens": 95,
        "completion_tokens": 127,
        "prompt_tokens_details": {"cached_tokens": 64},
    },
}
SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"}}}


class FakeCloudRu:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == f"{CLOUD_RU_API_URL}/chat/completions"
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(self.status, text='{"error": "Not enough money"}')
        return httpx2.Response(200, json=REPLY)

    def body(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.requests[-1].content)
        return result


def client(fake: FakeCloudRu) -> CloudRu:
    return CloudRu(
        api_key=SecretStr("secret"),
        api_url=CLOUD_RU_API_URL,
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        sleep=lambda _: None,
    )


MESSAGES: list[Message] = [
    {"role": "system", "content": "Ты проверяешь утверждения."},
    {"role": "user", "content": "Пишите на ivan@example.com: сырники из творога."},
]


def test_schema_goes_in_openai_format_and_reasoning_stays_out_of_the_answer() -> None:
    fake = FakeCloudRu()
    llm = client(fake)

    completion = llm.chat(
        MESSAGES,
        model="deepseek-ai/DeepSeek-V4-Flash",
        temperature=1e-4,
        max_tokens=2000,
        response_format={"type": "json_schema", "schema": SCHEMA, "strict": True},
    )

    body = fake.body()
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": SCHEMA, "strict": True},
    }
    assert body["max_tokens"] == 2000 + REASONING_TOKENS
    assert fake.requests[-1].headers["Authorization"] == "Bearer secret"
    assert completion.content == '{"supported": false}'
    assert completion.finish_reason == "stop"
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (
        95,
        127,
    )
    assert completion.usage.precached_prompt_tokens == 64
    assert llm.stats.requests == 1


def test_personal_data_is_masked_before_the_request() -> None:
    fake = FakeCloudRu()
    llm = client(fake)

    llm.chat(MESSAGES, model="z-ai/glm-5.3-flash", temperature=1e-4, max_tokens=100)

    sent = fake.requests[-1].content.decode()
    assert "ivan@example.com" not in sent
    assert llm.stats.masked["EMAIL"] == 1


def test_functions_are_refused_not_silently_dropped() -> None:
    llm = client(FakeCloudRu())

    with pytest.raises(LlmError, match="не реализован"):
        llm.chat(
            MESSAGES,
            model="z-ai/glm-5.3-flash",
            temperature=1e-4,
            max_tokens=100,
            functions=[{"name": "search_recipes"}],
        )


def test_no_money_error_explains_the_balance() -> None:
    llm = client(FakeCloudRu(status=402))

    with pytest.raises(LlmError, match="нет рублей"):
        llm.chat(MESSAGES, model="z-ai/glm-5.3-flash", temperature=1e-4, max_tokens=10)


def test_empty_key_is_an_error_before_any_request() -> None:
    with pytest.raises(LlmError, match="CLOUD_RU_API_KEY"):
        CloudRu(
            api_key=SecretStr(""),
            api_url=CLOUD_RU_API_URL,
            client=httpx2.Client(transport=httpx2.MockTransport(FakeCloudRu())),
        )


def test_cost_uses_catalog_prices() -> None:
    stats = LlmStats(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    assert cost_rub("z-ai/glm-5.3-flash", stats) == pytest.approx(26.71 + 89.0)
    assert cost_rub("unknown", stats) is None


def test_schema_is_spelled_out_for_models_that_ignore_it() -> None:
    """DeepSeek принимает json_schema, но отвечает своими ключами: схема — текстом."""
    fake = FakeCloudRu()
    llm = client(fake)
    schema_format = {"type": "json_schema", "schema": SCHEMA, "strict": True}

    llm.chat(
        MESSAGES,
        model="deepseek-ai/DeepSeek-V4-Flash",
        temperature=1e-4,
        max_tokens=10,
        response_format=schema_format,
    )
    deepseek = fake.body()["messages"]
    llm.chat(
        MESSAGES,
        model="z-ai/glm-5.3-flash",
        temperature=1e-4,
        max_tokens=10,
        response_format=schema_format,
    )
    glm = fake.body()["messages"]

    assert deepseek[0]["role"] == "system"
    assert deepseek[0]["content"].startswith("Ты проверяешь утверждения.")
    assert '"supported"' in deepseek[0]["content"]
    assert len(deepseek) == len(MESSAGES)
    assert glm[0]["content"] == "Ты проверяешь утверждения."


def test_effort_goes_as_reasoning_effort_only_when_set() -> None:
    fake = FakeCloudRu()
    CloudRu(
        api_key=SecretStr("secret"),
        api_url=CLOUD_RU_API_URL,
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        effort="low",
    ).chat(MESSAGES, model="z-ai/glm-5.3-flash", temperature=1e-4, max_tokens=10)
    assert fake.body()["reasoning_effort"] == "low"

    client(fake).chat(
        MESSAGES, model="z-ai/glm-5.3-flash", temperature=1e-4, max_tokens=10
    )
    assert "reasoning_effort" not in fake.body()
