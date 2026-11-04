"""Клиент GigaChat API на httpx2.

Как устроен доступ (проверено 2026-11-04):
1. Ключ авторизации (`GIGACHAT_API_KEY`) меняется на токен доступа:
   POST на OAuth-адрес с `Authorization: Basic <ключ>`, заголовком `RqUID`
   и формой `scope=GIGACHAT_API_PERS`. Токен живёт 30 минут, `expires_at`
   приходит в миллисекундах.
2. С токеном вызывается `POST /chat/completions` в формате, похожем на OpenAI.

Почему свой адаптер, а не SDK: это два POST-запроса, а SDK — лишние
зависимости и поверхность для сканера уязвимостей (как у эмбеддингов).

Что делает клиент:
- **маскирует ПДн в каждом сообщении** перед отправкой, здесь, а не у
  вызывающего кода: так маскирование нельзя обойти;
- держит токен и обновляет его заранее (за TOKEN_MARGIN до конца) или
  один раз после ответа 401;
- пропускает запросы по одному (замок): на Freemium GigaChat генерирует
  в один поток, а параллельный запрос получил бы 429;
- повторяет сетевые ошибки, 429 и 5xx (`assistant/http.py`).

TLS серверов GigaChat подписан НУЦ Минцифры. Его корневой сертификат лежит
рядом (`certs/`), и этот клиент доверяет только ему: и OAuth-адрес, и API
подписаны им. В систему, certifi и `SSL_CERT_FILE` сертификат не ставится.
"""

import ssl
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import httpx2
from pydantic import SecretStr

from assistant import __version__
from assistant.config import Settings
from assistant.http import request_with_retries
from assistant.security.pii import mask_pii

CA_FILE = Path(__file__).parent / "certs" / "russian_trusted_root_ca.pem"
# SHA-256 отпечаток корневого сертификата Минцифры. Тест сверяет с ним файл:
# подмена сертификата в репозитории не пройдёт незамеченной.
CA_SHA256 = (
    "D2:6D:2D:02:31:B7:C3:9F:92:CC:73:85:12:BA:54:10:"
    "35:19:E4:40:5D:68:B5:BD:70:3E:97:88:CA:8E:CF:31"
)

# Модель, если LLM_MODEL пустой. Выбрана по eval: Pro прошёл все пороги, Lite с
# json_schema на полном промпте отдаёт пустые ответы (evals/reports/2026-11-04-llm.md).
# Ultra — только судья: на платных тарифах её нет.
DEFAULT_MODEL = "GigaChat-2-Pro"
# GigaChat требует temperature > 0. Значение меньше 0.001 включает «строгий
# режим», где ответы повторяются: так выполняется правило «температура 0».
DETERMINISTIC_TEMPERATURE = 1e-4
# Обновлять токен за минуту до конца срока: запрос не должен уйти с истёкшим.
TOKEN_MARGIN = 60.0
TIMEOUT = 120.0

KEY_HELP = (
    "получите ключ авторизации в личном кабинете developers.sber.ru "
    "(проект GigaChat API → Настройки API → Получить ключ)"
)


class LlmError(RuntimeError):
    pass


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    # Токены промпта, взятые из кэша GigaChat: их вычитают из оплаты.
    precached_prompt_tokens: int
    # Всего, как считает API. По документации — за вычетом кэшированных,
    # но на пробе 2026-11-04 было ровно prompt + completion.
    total_tokens: int


@dataclass(frozen=True)
class Completion:
    content: str
    # stop | length | function_call | blacklist | error
    finish_reason: str
    # Модель с версией из ответа API, например «GigaChat-2:2.0.28.2».
    model: str
    usage: Usage


@dataclass
class LlmStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    precached_prompt_tokens: int = 0
    total_tokens: int = 0
    masked: Counter[str] = field(default_factory=Counter)

    def add(self, usage: Usage) -> None:
        self.requests += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.precached_prompt_tokens += usage.precached_prompt_tokens
        self.total_tokens += usage.total_tokens


class ChatModel(Protocol):
    """Что нужно ассистенту от LLM. В тестах вместо GigaChat — заглушка."""

    stats: LlmStats

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Mapping[str, object] | None = None,
    ) -> Completion: ...


def ssl_context() -> ssl.SSLContext:
    """TLS-контекст только с корнем Минцифры: ни системного хранилища, ни certifi."""
    return ssl.create_default_context(cafile=CA_FILE)


def make_http_client(timeout: float = TIMEOUT) -> httpx2.Client:
    return httpx2.Client(verify=ssl_context(), timeout=timeout)


def _error(prefix: str, response: httpx2.Response, hint: str | None) -> LlmError:
    # Тело обрезаем: нужна причина, а не простыня.
    message = f"{prefix} ответил HTTP {response.status_code}: {response.text[:300]}"
    return LlmError(f"{message}. {hint}" if hint else message)


class GigaChat:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        scope: str,
        api_url: str,
        oauth_url: str,
        client: httpx2.Client,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise LlmError(
                f"не задан GIGACHAT_API_KEY: {KEY_HELP} и добавьте строку "
                "GIGACHAT_API_KEY=… в .env"
            )
        self.stats = LlmStats()
        self._api_key = api_key
        self._scope = scope
        self._chat_url = f"{api_url.rstrip('/')}/chat/completions"
        self._oauth_url = oauth_url
        self._client = client
        self._clock = clock
        self._sleep = sleep
        self._token: SecretStr | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Mapping[str, object] | None = None,
    ) -> Completion:
        masked: list[Message] = []
        for message in messages:
            result = mask_pii(message["content"])
            self.stats.masked.update(result.counts)
            masked.append({"role": message["role"], "content": result.text})
        body: dict[str, object] = {
            "model": model,
            "messages": masked,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        with self._lock:
            response = self._post_chat(body)
            if response.status_code == 401:
                # Токен мог истечь раньше срока: берём новый и пробуем один раз.
                self._token = None
                response = self._post_chat(body)
        if response.status_code != 200:
            raise _error("GigaChat", response, self._chat_hint(response, model))

        data = response.json()
        choice = data["choices"][0]
        raw_usage = data.get("usage", {})
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            precached_prompt_tokens=int(raw_usage.get("precached_prompt_tokens", 0)),
            total_tokens=int(raw_usage.get("total_tokens", 0)),
        )
        self.stats.add(usage)
        return Completion(
            content=choice["message"].get("content") or "",
            finish_reason=choice["finish_reason"],
            model=data.get("model", model),
            usage=usage,
        )

    @staticmethod
    def _chat_hint(response: httpx2.Response, model: str) -> str | None:
        return {
            402: f"Закончились токены пакета для {model}: проверьте баланс в личном кабинете",
            404: f"Модели {model} нет у этого ключа: проверьте LLM_MODEL",
        }.get(response.status_code)

    def _post_chat(self, body: dict[str, object]) -> httpx2.Response:
        headers = {
            "Authorization": f"Bearer {self._access_token().get_secret_value()}",
            "Accept": "application/json",
            "User-Agent": f"paperless-assistant/{__version__}",
        }
        return request_with_retries(
            lambda: self._client.post(self._chat_url, json=body, headers=headers),
            sleep=self._sleep,
        )

    def _access_token(self) -> SecretStr:
        if self._token is None or self._clock() >= self._expires_at - TOKEN_MARGIN:
            self._token, self._expires_at = self._request_token()
        return self._token

    def _request_token(self) -> tuple[SecretStr, float]:
        headers = {
            "Authorization": f"Basic {self._api_key.get_secret_value()}",
            # Идентификатор запроса для журнала GigaChat, обязателен.
            "RqUID": str(uuid.uuid4()),
            "Accept": "application/json",
        }
        response = request_with_retries(
            lambda: self._client.post(
                self._oauth_url, data={"scope": self._scope}, headers=headers
            ),
            sleep=self._sleep,
        )
        if response.status_code != 200:
            hint = (
                "Проверьте GIGACHAT_API_KEY (нужен «ключ авторизации», а не Client "
                "Secret) и GIGACHAT_SCOPE (для физлиц — GIGACHAT_API_PERS)"
                if response.status_code in (400, 401)
                else None
            )
            raise _error("OAuth GigaChat", response, hint)
        data = response.json()
        # expires_at — момент окончания в миллисекундах от эпохи.
        return SecretStr(data["access_token"]), data["expires_at"] / 1000


def make_llm(settings: Settings, client: httpx2.Client) -> GigaChat:
    return GigaChat(
        api_key=settings.gigachat_api_key,
        scope=settings.gigachat_scope,
        api_url=settings.gigachat_api_url,
        oauth_url=settings.gigachat_oauth_url,
        client=client,
    )


def llm_model(settings: Settings) -> str:
    return settings.llm_model or DEFAULT_MODEL
