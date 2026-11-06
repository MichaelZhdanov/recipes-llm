"""Адаптер Claude: перевод формата GigaChat в Messages API Anthropic и обратно."""

import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from assistant.config import ANTHROPIC_API_URL, Settings
from assistant.generation import llm
from assistant.generation.claude import API_VERSION, THINKING_TOKENS, Claude
from assistant.generation.gigachat import (
    GigaChat,
    LlmError,
    Message,
    call_message,
    function_result,
)

SEARCH = {
    "name": "search_recipes",
    "description": "Найти рецепты",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "few_shot_examples": [{"request": "хочу рататуй", "params": {"query": "рататуй"}}],
}
CALL_REPLY = {
    "model": "claude-haiku-5-5",
    "stop_reason": "tool_use",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "x"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "search_recipes",
            "input": {"query": "курица без лука"},
        },
    ],
    "usage": {
        "input_tokens": 40,
        "output_tokens": 120,
        "cache_read_input_tokens": 5000,
        "cache_creation_input_tokens": 0,
    },
}


class FakeAnthropic:
    def __init__(self, status: int = 200, reply: dict[str, Any] | None = None) -> None:
        self.status = status
        self.reply = reply or CALL_REPLY
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == f"{ANTHROPIC_API_URL}/messages"
        self.requests.append(request)
        if self.status != 200:
            return httpx2.Response(
                self.status, text='{"error": "credit balance is too low"}'
            )
        return httpx2.Response(200, json=self.reply)

    def body(self) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.requests[-1].content)
        return result


def make(fake: FakeAnthropic, effort: str = "medium") -> Claude:
    return Claude(
        api_key=SecretStr("sk-test"),
        api_url=ANTHROPIC_API_URL,
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        effort=effort,
        sleep=lambda _: None,
    )


def ask(model: Claude, messages: list[Message]) -> Any:
    return model.chat(
        messages,
        model="claude-haiku-5-5",
        temperature=1e-4,
        max_tokens=1000,
        functions=[SEARCH],
        function_call={"name": "search_recipes"},
    )


SYSTEM: Message = {"role": "system", "content": "Разбери вопрос. Словарь: курица, лук"}
USER: Message = {"role": "user", "content": "курица есть, лук не ем"}


def test_request_is_translated_to_messages_api() -> None:
    fake = FakeAnthropic()

    ask(make(fake), [SYSTEM, USER])

    request = fake.requests[0]
    assert request.headers["x-api-key"] == "sk-test"
    assert request.headers["anthropic-version"] == API_VERSION
    body = fake.body()
    # Системный промпт со словарём одинаковый у всех вопросов — кэшируется.
    assert body["system"] == [
        {
            "type": "text",
            "text": SYSTEM["content"],
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert body["messages"] == [{"role": "user", "content": USER["content"]}]
    # Лимит вызывающего кода — на ответ, размышления сверху.
    assert body["max_tokens"] == 1000 + THINKING_TOKENS
    assert body["output_config"] == {"effort": "medium"}
    # API-версия 2026-10-21 отвечает на temperature ошибкой 400.
    assert "temperature" not in body


def test_functions_become_tools_and_forced_call_becomes_auto() -> None:
    """Принудительный вызов Sonnet и Opus 5.5 не принимают, а у Haiku 5.5 он
    пропускает размышления: всегда auto, о вызове говорит промпт."""
    fake = FakeAnthropic()

    ask(make(fake), [SYSTEM, USER])

    [tool] = fake.body()["tools"]
    assert tool["name"] == "search_recipes"
    assert tool["input_schema"] == SEARCH["parameters"]
    # Примеры GigaChat (few_shot_examples) — текстом в описании.
    assert tool["description"].startswith("Найти рецепты\n\nПримеры")
    assert '«хочу рататуй» → {"query": "рататуй"}' in tool["description"]
    assert fake.body()["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_without_effort_the_model_default_is_used() -> None:
    fake = FakeAnthropic()

    ask(make(fake, effort=""), [SYSTEM, USER])

    assert "output_config" not in fake.body()


def test_tool_use_is_parsed_as_function_call() -> None:
    completion = ask(make(FakeAnthropic()), [SYSTEM, USER])

    assert completion.finish_reason == "function_call"
    assert completion.function_call == {
        "name": "search_recipes",
        "arguments": {"query": "курица без лука"},
    }
    assert completion.model == "claude-haiku-5-5"
    assert completion.usage.precached_prompt_tokens == 5000
    assert completion.usage.completion_tokens == 120


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [("end_turn", "stop"), ("max_tokens", "length"), ("refusal", "blacklist")],
)
def test_stop_reason_is_mapped(stop_reason: str, finish_reason: str) -> None:
    reply = {
        "model": "claude-haiku-5-5",
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": "привет"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    completion = ask(make(FakeAnthropic(reply=reply)), [SYSTEM, USER])

    assert completion.finish_reason == finish_reason
    assert completion.function_call is None
    assert completion.content == "привет"


def test_history_with_a_call_becomes_tool_use_and_tool_result() -> None:
    fake = FakeAnthropic()
    called = ask(make(fake), [SYSTEM, USER])

    ask(
        make(fake),
        [SYSTEM, USER, call_message(called), function_result("search_recipes", [1])],
    )

    _, assistant, result = fake.body()["messages"]
    assert assistant == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_0",
                "name": "search_recipes",
                "input": {"query": "курица без лука"},
            }
        ],
    }
    assert result == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_0", "content": "[1]"}],
    }


def test_personal_data_is_masked_in_the_request() -> None:
    fake = FakeAnthropic()
    model = make(fake)

    ask(
        model, [SYSTEM, {"role": "user", "content": "торт, пишите на olga@example.com"}]
    )

    assert "olga@example.com" not in fake.requests[0].content.decode()
    assert model.stats.masked["EMAIL"] == 1


SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string"}},
    "required": ["status"],
    "additionalProperties": False,
}


def test_response_format_becomes_output_config_format() -> None:
    """JSON по схеме: формат GigaChat → output_config.format, рядом с effort."""
    fake = FakeAnthropic(
        reply={
            "model": "claude-haiku-5-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"status": "answer"}'}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    completion = make(fake).chat(
        [USER],
        model="claude-haiku-5-5",
        temperature=1e-4,
        max_tokens=100,
        response_format={"type": "json_schema", "schema": SCHEMA, "strict": True},
        functions=[SEARCH],
        function_call="auto",
    )

    assert fake.body()["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert "tools" in fake.body()
    assert completion.content == '{"status": "answer"}'


def test_only_json_schema_response_format_is_understood() -> None:
    with pytest.raises(LlmError, match="json_schema"):
        make(FakeAnthropic()).chat(
            [USER],
            model="claude-haiku-5-5",
            temperature=1e-4,
            max_tokens=100,
            response_format={"type": "json_object"},
        )


def test_missing_key_and_empty_balance_explain_what_to_do() -> None:
    with pytest.raises(LlmError, match="подписка Claude ключа не даёт"):
        Claude(api_key=SecretStr(""), api_url=ANTHROPIC_API_URL, client=httpx2.Client())
    with pytest.raises(LlmError, match="Billing"):
        ask(make(FakeAnthropic(status=400)), [SYSTEM, USER])


def test_settings_choose_the_provider() -> None:
    gigachat = Settings(llm_provider="gigachat", gigachat_api_key=SecretStr("key"))
    claude = Settings(
        llm_provider="anthropic",
        anthropic_api_key=SecretStr("sk-test"),
        llm_effort="high",
    )

    with httpx2.Client() as client:
        assert isinstance(llm.make_chat_model(gigachat, client), GigaChat)
        model = llm.make_chat_model(claude, client)
    assert isinstance(model, Claude) and model.effort == "high"
    assert llm.llm_model(gigachat) == "GigaChat-2-Pro"
    assert llm.llm_model(claude) == "claude-haiku-5-5"
    # По умолчанию — GLM-5.1 в cloud.ru (API Anthropic закрыт, writers.md).
    assert Settings().llm_provider == "cloudru"
    assert llm.llm_model(Settings()) == "zai-org/GLM-5.1"
    assert llm.llm_model(Settings(llm_model="claude-sonnet-5-5")) == "claude-sonnet-5-5"
