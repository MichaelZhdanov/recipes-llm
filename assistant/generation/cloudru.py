"""Чат cloud.ru Evolution Foundation Models (OpenAI-совместимый) на httpx2.

Нужен судьям другого семейства, чем Claude, который отвечает: DeepSeek V4
Flash и GLM-5.3-Flash из каталога cloud.ru (решение автора 2026-11-06).
Ключ и адрес — те же, что у эмбеддингов (`CLOUD_RU_API_KEY`,
`CLOUD_RU_API_URL`). Обе модели в каталоге помечены `provider: external`:
cloud.ru передаёт запрос вендору за пределы РФ.

Перевод из формата GigaChat (проверено живым вызовом 2026-11-06):
- сообщения system / user / assistant — как есть, `POST /chat/completions`;
- `response_format` (JSON по схеме) → `{type: json_schema, json_schema:
  {name, schema, strict}}`. GLM схему соблюдает. DeepSeek V4 Flash запрос
  принимает, но схему не соблюдает: возвращает свои ключи («evaluations»
  вместо «statements»; 40 сбоев разбора из 70 в evals/reports/2026-11-06-loop-before-schema-fix.md).
  Таким моделям (SCHEMA_IN_PROMPT) схема дописывается текстом в системное
  сообщение, а ответ, как и у всех, проверяет pydantic;
- рассуждения приходят отдельным полем `reasoning_content`, в content —
  только ответ. `max_tokens` считает их вместе с ответом, поэтому адаптер
  добавляет запас REASONING_TOKENS, как у Claude;
- `effort` → `reasoning_effort`: сколько модель думает. GLM по умолчанию
  думает на максимуме — до двух минут на вызов судьи; на неверное значение
  cloud.ru отвечает 422, значит параметр доходит до модели. Пусто — не
  отправляется (по умолчанию модели). Формат `reasoning: {effort}` cloud.ru
  молча игнорирует (проверено 2026-11-06);
- вызов функций не реализован: судьям он не нужен, а генератор ответов —
  Claude Haiku. Передать `functions` — ошибка, а не тихое «без функций».

Маскирование ПДн — здесь, как у GigaChat и Claude (`mask_message`).
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx2
from pydantic import SecretStr

from assistant import __version__
from assistant.generation.gigachat import (
    Completion,
    FunctionCallMode,
    LlmError,
    LlmStats,
    Message,
    Usage,
    mask_message,
)
from assistant.http import request_with_retries

TIMEOUT = 180.0
# Модели, которые принимают json_schema, но не соблюдают её (проверено 2026-11-06).
SCHEMA_IN_PROMPT = frozenset({"deepseek-ai/DeepSeek-V4-Flash"})
REASONING_TOKENS = 8000
# ₽ за 1M токенов (вход, выход), каталог cloud.ru `GET /v1/models`, 2026-11-06.
PRICE_RUB_PER_MTOK = {
    "deepseek-ai/DeepSeek-V4-Flash": (43.30, 86.58),
    "z-ai/glm-5.3-flash": (26.71, 89.0),
}

KEY_HELP = (
    "ключ — API-ключ сервисного аккаунта уровня «Проект» в cloud.ru "
    "(Сервисные аккаунты → API-ключи, сервис Foundation Models)"
)

_FINISH = {"tool_calls": "function_call", "content_filter": "blacklist"}


def _response_format(response_format: Mapping[str, object]) -> dict[str, object]:
    """Формат GigaChat `{type: json_schema, schema, strict}` → формат OpenAI."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": response_format["schema"],
            "strict": bool(response_format.get("strict", True)),
        },
    }


def _with_schema(
    messages: list[dict[str, str]], response_format: Mapping[str, object]
) -> list[dict[str, str]]:
    """Схема ответа текстом в системном сообщении — для моделей из SCHEMA_IN_PROMPT."""
    schema = json.dumps(response_format["schema"], ensure_ascii=False)
    note = f"JSON-схема ответа (ключи — ровно такие, других не добавляй):\n{schema}"
    if messages and messages[0]["role"] == "system":
        first = {**messages[0], "content": f"{messages[0]['content']}\n\n{note}"}
        return [first, *messages[1:]]
    return [{"role": "system", "content": note}, *messages]


def _completion(data: Mapping[str, Any], model: str) -> Completion:
    choices = data.get("choices") or []
    if not choices:
        raise LlmError(f"cloud.ru вернул ответ без choices для {model}")
    choice = choices[0]
    message = choice.get("message") or {}
    raw = data.get("usage") or {}
    details = raw.get("prompt_tokens_details") or {}
    prompt = int(raw.get("prompt_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or 0)
    finish = str(choice.get("finish_reason"))
    return Completion(
        content=str(message.get("content") or ""),
        finish_reason=_FINISH.get(finish, finish),
        model=str(data.get("model", model)),
        usage=Usage(
            prompt,
            completion_tokens,
            int(details.get("cached_tokens") or 0),
            prompt + completion_tokens,
        ),
    )


class CloudRu:
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
            raise LlmError(f"не задан CLOUD_RU_API_KEY: {KEY_HELP}")
        self.stats = LlmStats()
        self.effort = effort
        self._api_key = api_key
        self._url = f"{api_url.rstrip('/')}/chat/completions"
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
        if functions or function_call:
            raise LlmError("вызов функций в адаптере cloud.ru не реализован")
        masked = [mask_message(m, self.stats.masked) for m in messages]
        converted = [{"role": m["role"], "content": m["content"]} for m in masked]
        if response_format is not None and model in SCHEMA_IN_PROMPT:
            converted = _with_schema(converted, response_format)
        body: dict[str, object] = {
            "model": model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens + REASONING_TOKENS,
        }
        if response_format is not None:
            body["response_format"] = _response_format(response_format)
        if self.effort:
            body["reasoning_effort"] = self.effort
        response = request_with_retries(
            lambda: self._client.post(self._url, json=body, headers=self._headers()),
            sleep=self._sleep,
        )
        if response.status_code != 200:
            raise LlmError(self._error(response, model))
        completion = _completion(response.json(), model)
        self.stats.add(completion.usage)
        return completion

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "User-Agent": f"recipes-llm/{__version__}",
        }

    @staticmethod
    def _error(response: httpx2.Response, model: str) -> str:
        message = f"cloud.ru ответил HTTP {response.status_code}: {response.text[:300]}"
        if response.status_code == 401:
            return f"{message}. Проверьте CLOUD_RU_API_KEY: {KEY_HELP}"
        if response.status_code == 402:
            return f"{message}. На балансе cloud.ru нет рублей (бонусы не тратятся)"
        if response.status_code == 404:
            return f"{message}. Модели {model} нет в каталоге: GET /v1/models"
        return message


def cost_rub(model: str, stats: LlmStats) -> float | None:
    """Цена токенов по каталогу; None — модели нет в таблице."""
    price = PRICE_RUB_PER_MTOK.get(model)
    if price is None:
        return None
    return (stats.prompt_tokens * price[0] + stats.completion_tokens * price[1]) / 1e6


def make_http_client(timeout: float = TIMEOUT) -> httpx2.Client:
    return httpx2.Client(timeout=timeout)
