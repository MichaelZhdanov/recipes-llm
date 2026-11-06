"""Вызов функций в адаптере cloud.ru: формат OpenAI tools и разбор tool_calls."""

import json

import httpx2
import pytest
from pydantic import SecretStr

from assistant.generation.cloudru import CloudRu
from assistant.generation.gigachat import LlmError, Message

SEARCH = {
    "name": "search_recipes",
    "description": "Найти рецепты.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    "few_shot_examples": [{"request": "блины", "params": {"query": "блины"}}],
}


def model(reply: dict[str, object]) -> tuple[CloudRu, list[dict[str, object]]]:
    bodies: list[dict[str, object]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=reply)

    llm = CloudRu(
        api_key=SecretStr("key"),
        api_url="https://cloud.example/v1",
        client=httpx2.Client(transport=httpx2.MockTransport(handle)),
        sleep=lambda _: None,
    )
    return llm, bodies


def tool_reply(arguments: str) -> dict[str, object]:
    return {
        "model": "moonshotai/Kimi-K2.6",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_recipes",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


QUESTION: list[Message] = [{"role": "user", "content": "горячая закуска под водку"}]


def test_forced_function_call_goes_as_openai_tools_and_comes_back_parsed() -> None:
    llm, bodies = model(tool_reply('{"query": "солянка"}'))

    completion = llm.chat(
        QUESTION,
        model="moonshotai/Kimi-K2.6",
        temperature=1e-4,
        max_tokens=500,
        functions=[SEARCH],
        function_call={"name": "search_recipes"},
    )

    assert completion.finish_reason == "function_call"
    assert completion.function_call == {
        "name": "search_recipes",
        "arguments": {"query": "солянка"},
    }
    [body] = bodies
    tools = body["tools"]
    assert isinstance(tools, list) and len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "search_recipes"
    assert "Примеры вопросов и аргументов" in tool["function"]["description"]
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_recipes"},
    }


def test_broken_arguments_and_function_history_are_errors_not_silence() -> None:
    llm, _ = model(tool_reply("{не json"))
    with pytest.raises(LlmError, match="не JSON"):
        llm.chat(
            QUESTION, model="m", temperature=1e-4, max_tokens=10, functions=[SEARCH]
        )

    history: list[Message] = [
        *QUESTION,
        {"role": "function", "name": "search_recipes", "content": "{}"},
    ]
    with pytest.raises(LlmError, match="режим auto"):
        llm.chat(history, model="m", temperature=1e-4, max_tokens=10)
