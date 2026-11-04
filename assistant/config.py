"""Настройки ассистента из переменных окружения.

Читается только окружение процесса, без файла `.env`. В Docker переменные
передаёт compose, а на тесты не влияет чей-то локальный `.env`.
Список переменных с пояснениями лежит в `.env.example`.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Пока один провайдер LLM; другие — S2 из плана (смена модели).
LlmProvider = Literal["gigachat"]
# Области доступа GigaChat: физлица (Freemium), ИП и юрлица по предоплате, по постоплате.
GigaChatScope = Literal["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"]
EmbeddingProvider = Literal["cloudru", "voyage"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


# Адреса API Voyage для ключей из dash.voyageai.com и из MongoDB Atlas.
# Ключ, отправленный не на свой адрес, получает 403.
VOYAGE_API_URL = "https://api.voyageai.com/v1"
VOYAGE_ATLAS_API_URL = "https://ai.mongodb.com/v1"
# Evolution Foundation Models: OpenAI-совместимый API, ключ — API-ключ сервисного
# аккаунта уровня «Проект» (аккаунт уровня организации получает 401).
CLOUD_RU_API_URL = "https://foundation-models.api.cloud.ru/v1"
# GigaChat: токен выдаёт отдельный OAuth-адрес, генерация — на api.giga.chat.
GIGACHAT_API_URL = "https://api.giga.chat/v1"
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"


class Settings(BaseSettings):
    """Все настройки сервиса. Секреты хранятся как SecretStr и не видны в repr и логах."""

    model_config = SettingsConfigDict(
        # Пустая строка (`ESCALATION_CHAT_ID=` в .env) означает «взять значение по умолчанию».
        env_ignore_empty=True,
        # Настройки читаются один раз при старте и дальше не меняются.
        frozen=True,
    )

    llm_provider: LlmProvider = "gigachat"
    # Пустая модель — модель по умолчанию (assistant/generation/gigachat.py).
    llm_model: str = ""
    gigachat_api_key: SecretStr = SecretStr("")
    gigachat_scope: GigaChatScope = "GIGACHAT_API_PERS"
    gigachat_api_url: str = GIGACHAT_API_URL
    gigachat_oauth_url: str = GIGACHAT_OAUTH_URL
    # Эмбеддинги. Пустая модель — модель провайдера по умолчанию (DEFAULT_MODELS).
    embedding_provider: EmbeddingProvider = "cloudru"
    embedding_model: str = ""
    cloud_ru_api_key: SecretStr = SecretStr("")
    cloud_ru_api_url: str = CLOUD_RU_API_URL
    # Voyage AI; адрес должен соответствовать кабинету, где создан ключ.
    voyage_api_key: SecretStr = SecretStr("")
    voyage_api_url: str = VOYAGE_API_URL
    # Умолчания рассчитаны на запуск с хоста; контейнеру ассистента адреса docker-сети задаёт compose.
    qdrant_url: str = "http://localhost:6333"
    paperless_source_path: Path = Path("data/paperless-src")
    paperless_ref: str = "v3.4.2"
    paperless_repo_url: str = "https://github.com/MichaelZhdanov/paperless-ngx-rag"
    # Не PAPERLESS_URL: так называется переменная самого Paperless (его публичный адрес).
    paperless_api_url: str = "http://localhost:8000"
    paperless_token: SecretStr = SecretStr("")
    telegram_bot_token: SecretStr = SecretStr("")
    # У групп и каналов Telegram id отрицательный, поэтому int, а не положительное число.
    escalation_chat_id: int | None = None
    # NoDecode: без него pydantic-settings ждёт JSON вида "[1, 2]", а в .env привычнее "1,2".
    engineer_telegram_ids: Annotated[frozenset[int], NoDecode] = frozenset()
    log_level: LogLevel = "INFO"
    # Вход в веб-чат и API (assistant/security/auth.py). NO_AUTH=1 выключает его,
    # по умолчанию вход нужен: открытым сервер не окажется по забывчивости.
    no_auth: bool = False
    web_login: str = "admin"
    web_password: SecretStr = SecretStr("")
    # Ключ подписи cookie сессии; сменить его — разлогинить всех.
    web_session_secret: SecretStr = SecretStr("")

    @field_validator("engineer_telegram_ids", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value
