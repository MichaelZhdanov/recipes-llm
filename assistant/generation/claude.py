"""Клиент Claude (Anthropic Messages API) на httpx2 — второй провайдер LLM.

Тот же интерфейс `ChatModel`, что у GigaChat: ассистент пишет сообщения и
описания функций в формате GigaChat, а этот адаптер переводит их в формат
Anthropic и обратно. Почему свой адаптер, а не SDK — как у GigaChat:
один POST-запрос, а SDK — лишняя поверхность для сканера уязвимостей.

Перевод (документация Anthropic, API-версия 2026-10-21):
- системное сообщение → поле `system`; оно помечено для кэша
  (`cache_control`): словарь ингредиентов в нём одинаковый у всех запросов;
- `functions` → `tools` (`parameters` → `input_schema`); примеры
  `few_shot_examples` — поле GigaChat, у Anthropic такого нет, поэтому
  они дописываются в описание инструмента текстом;
- `function_call` → `tool_choice`. Вызов всегда `auto`: принудительный
  (`tool`) Sonnet 5.5 и Opus 5.5 не принимают, а у Haiku 5.5 он
  пропускает размышления — тогда `effort` ни на что не влиял бы. Что
  ответить нужно вызовом функции, говорит промпт;
- вызов функции в истории → блок `tool_use`, результат → `tool_result`;
- `temperature` не отправляется: API-версия 2026-10-21 её не принимает
  (400). Правило eval «температура 0» здесь не выполнимо, это пишется в отчёт;
- `effort` — сколько модель думает (`LLM_EFFORT`); пусто — по умолчанию модели;
- `response_format` (JSON по схеме, формат GigaChat) → `output_config.format`.
  Документация разрешает его вместе с инструментами: модель вызывает
  функции, а последний ответ пишет JSON по схеме. У Claude каждый объект
  схемы обязан иметь `additionalProperties: false` — это забота схемы.

У моделей 5.5 размышления включены, и `max_tokens` считает их вместе с
ответом. Лимит, который просит вызывающий код, — на сам ответ, поэтому
адаптер добавляет запас THINKING_TOKENS.

Маскирование ПДн — здесь, как у GigaChat (`mask_message`).
"""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx2
from pydantic import SecretStr

from assistant import __version__
from assistant.generation.functions import describe
from assistant.generation.gigachat import (
    Completion,
    FunctionCall,
    FunctionCallMode,
    LlmError,
    LlmStats,
    Message,
    Usage,
    mask_message,
)
from assistant.http import request_with_retries

API_VERSION = "2026-10-21"
# Выбрана по eval этапа 2: conditions@5 0.92 при пороге 0.90, ≈ $0.0003 за вопрос.
DEFAULT_MODEL = "claude-haiku-5-5"
TIMEOUT = 120.0
# Запас на размышления поверх лимита ответа: без него размышления съедают
# max_tokens, и ответ обрывается (stop_reason: max_tokens).
THINKING_TOKENS = 8000

KEY_HELP = "ключ API — platform.claude.com → API keys; подписка Claude ключа не даёт"

# stop_reason Anthropic → finish_reason в терминах GigaChat, который знает ассистент.
_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "tool_use": "function_call",
    "max_tokens": "length",
    "refusal": "blacklist",
}


def _tool(function: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": function["name"],
        "description": describe(function),
        "input_schema": function["parameters"],
    }


def _tool_choice(mode: FunctionCallMode | None) -> dict[str, object] | None:
    if mode is None:
        return None
    if mode == "none":
        return {"type": "none"}
    return {"type": "auto", "disable_parallel_tool_use": True}


def _output_format(response_format: Mapping[str, object]) -> dict[str, object]:
    """`response_format` GigaChat → `output_config.format` Anthropic."""
    if response_format.get("type") != "json_schema" or "schema" not in response_format:
        raise LlmError(
            f"Claude понимает только response_format json_schema: {response_format}"
        )
    return {"type": "json_schema", "schema": response_format["schema"]}


def _messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, object]]]:
    """(system, messages) в формате Anthropic."""
    system: list[str] = []
    result: list[dict[str, object]] = []
    call_ids: list[str] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system.append(message["content"])
        elif role == "function":
            # Результат функции — ответ на последний вызов модели.
            tool_use_id = call_ids[-1] if call_ids else "call_0"
            block = {"type": "tool_result", "tool_use_id": tool_use_id}
            result.append(
                {"role": "user", "content": [{**block, "content": message["content"]}]}
            )
        elif "function_call" in message:
            call = message["function_call"]
            call_ids.append(f"call_{len(call_ids)}")
            content: list[dict[str, object]] = []
            if message["content"]:
                content.append({"type": "text", "text": message["content"]})
            content.append(
                {
                    "type": "tool_use",
                    "id": call_ids[-1],
                    "name": call["name"],
                    "input": call["arguments"],
                }
            )
            result.append({"role": "assistant", "content": content})
        else:
            result.append({"role": role, "content": message["content"]})
    return "\n\n".join(system), result


def _completion(data: Mapping[str, Any], model: str) -> Completion:
    text: list[str] = []
    call: FunctionCall | None = None
    for block in data.get("content", []):
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use" and call is None:
            arguments = block.get("input") or {}
            if not isinstance(arguments, dict):
                raise LlmError(
                    f"Claude вернул аргументы {block.get('name')} не объектом"
                )
            call = {"name": str(block.get("name", "")), "arguments": arguments}
    raw = data.get("usage", {})
    cached = int(raw.get("cache_read_input_tokens") or 0)
    prompt = int(raw.get("input_tokens") or 0) + int(
        raw.get("cache_creation_input_tokens") or 0
    )
    completion_tokens = int(raw.get("output_tokens") or 0)
    stop = str(data.get("stop_reason"))
    return Completion(
        content="".join(text),
        finish_reason=_FINISH.get(stop, stop),
        model=str(data.get("model", model)),
        usage=Usage(prompt, completion_tokens, cached, prompt + completion_tokens),
        function_call=call,
    )


class Claude:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        api_url: str,
        client: httpx2.Client,
        effort: str = "",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise LlmError(
                f"не задан ANTHROPIC_API_KEY: {KEY_HELP}; добавьте строку "
                "ANTHROPIC_API_KEY=… в .env"
            )
        self.stats = LlmStats()
        self.effort = effort
        self._api_key = api_key
        self._url = f"{api_url.rstrip('/')}/messages"
        self._client = client
        self._sleep = sleep

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Mapping[str, object] | None = None,
        functions: Sequence[Mapping[str, object]] | None = None,
        function_call: FunctionCallMode | None = None,
    ) -> Completion:
        # temperature не отправляется: API-версия 2026-10-21 отвечает на неё 400.
        masked = [mask_message(m, self.stats.masked) for m in messages]
        system, converted = _messages(masked)
        body: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens + THINKING_TOKENS,
            "messages": converted,
        }
        if system:
            body["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if functions:
            body["tools"] = [_tool(f) for f in functions]
        if (choice := _tool_choice(function_call)) is not None:
            body["tool_choice"] = choice
        output_config: dict[str, object] = {}
        if self.effort:
            output_config["effort"] = self.effort
        if response_format is not None:
            output_config["format"] = _output_format(response_format)
        if output_config:
            body["output_config"] = output_config

        response = request_with_retries(
            lambda: self._client.post(self._url, json=body, headers=self._headers()),
            sleep=self._sleep,
        )
        if response.status_code != 200:
            raise LlmError(self._error(response, model), status=response.status_code)
        completion = _completion(response.json(), model)
        self.stats.add(completion.usage)
        return completion

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key.get_secret_value(),
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
            "User-Agent": f"recipes-llm/{__version__}",
        }

    @staticmethod
    def _error(response: httpx2.Response, model: str) -> str:
        # Тело обрезаем: нужна причина, а не простыня.
        message = f"Claude ответил HTTP {response.status_code}: {response.text[:300]}"
        if response.status_code == 401:
            return f"{message}. Проверьте ANTHROPIC_API_KEY: {KEY_HELP}"
        if "credit balance" in response.text:
            return f"{message}. На балансе API нет денег: platform.claude.com → Billing"
        if response.status_code == 404:
            return f"{message}. Модели {model} нет: проверьте LLM_MODEL"
        return message


def make_http_client(timeout: float = TIMEOUT) -> httpx2.Client:
    return httpx2.Client(timeout=timeout)
