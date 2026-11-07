"""Настройки ассистента из переменных окружения.

Читается только окружение процесса, без файла `.env`. В Docker переменные
передаёт compose, а на тесты не влияет чей-то локальный `.env`.
Список переменных с пояснениями лежит в `.env.example`.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# LLM ассистента: модель cloud.ru, GigaChat (данные в РФ) или Claude через API Anthropic.
LlmProvider = Literal["gigachat", "anthropic", "cloudru"]
# Сколько думает модель (Claude, cloud.ru); пусто — по умолчанию модели. У GigaChat не используется.
LlmEffort = Literal["", "low", "medium", "high", "xhigh", "max"]
# Режим ответа (решение 7): always — сначала поиск, потом ответ;
# auto — модель сама решает, искать ли.
AnswerMode = Literal["always", "auto"]
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
ANTHROPIC_API_URL = "https://api.anthropic.com/v1"
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

    # По умолчанию GLM-5.1 в cloud.ru (решение автора 2026-11-07). API Anthropic
    # закрыт: организация отключена. Из моделей cloud.ru GLM-5.1 low пишет ответ
    # за 9 с и 1,3–1,5 ₽ (0.94 утверждений без нарушений); Kimi K2.6 точнее
    # (0.98), но 77 с и 6–8,5 ₽ (evals/reports/2026-11-07-writers.md).
    llm_provider: LlmProvider = "cloudru"
    # Пустая модель — модель провайдера по умолчанию (assistant/generation/llm.py).
    llm_model: str = ""
    # low: так GLM-5.1 сравнивался с другими писателями (writers.md).
    llm_effort: LlmEffort = "low"
    # Режим по умолчанию выберет сравнение always и auto в eval этапа 3.
    answer_mode: AnswerMode = "always"
    # Проверка ответа Sonnet перед отправкой и одна доработка (review.py).
    # Выключена: API Anthropic закрыт, а проверка GLM с узкой рубрикой зря
    # отправляла на доработку 15 % верных утверждений
    # (evals/reports/2026-11-07-gate-narrow.md). Без проверки нет и шагов от ИИ.
    answer_review: bool = False
    gigachat_api_key: SecretStr = SecretStr("")
    gigachat_scope: GigaChatScope = "GIGACHAT_API_PERS"
    gigachat_api_url: str = GIGACHAT_API_URL
    gigachat_oauth_url: str = GIGACHAT_OAUTH_URL
    # Ключ API Anthropic (platform.claude.com); подписка Claude его не даёт.
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_api_url: str = ANTHROPIC_API_URL
    # Эмбеддинги. Пустая модель — модель провайдера по умолчанию (DEFAULT_MODELS).
    embedding_provider: EmbeddingProvider = "cloudru"
    embedding_model: str = ""
    # Дописывать ли к вопросу задачу у instruct-моделей (формат из карточки модели).
    # По умолчанию нет: так выбрало сравнение моделей (evals/reports/2026-11-05-models.md).
    embedding_query_instruction: bool = False
    cloud_ru_api_key: SecretStr = SecretStr("")
    cloud_ru_api_url: str = CLOUD_RU_API_URL
    # Voyage AI; адрес должен соответствовать кабинету, где создан ключ.
    voyage_api_key: SecretStr = SecretStr("")
    voyage_api_url: str = VOYAGE_API_URL
    # Умолчания рассчитаны на запуск с хоста; контейнеру ассистента адреса docker-сети задаёт compose.
    qdrant_url: str = "http://localhost:6333"
    # Ревизия (коммит) датасета rogozinushka/povarenok-recipes на HuggingFace:
    # по ней строится индекс, и её же паспорт индекса сверяет с настройками.
    dataset_revision: str = "ffaf49ee7439afe6dbf0653714ceacb9b57ff830"
    # Куда скачивается датасет: файл ревизии лежит в <путь>/<DATASET_REVISION>/.
    data_path: Path = Path("data/recipes")
    # Индекс на выборке из N рецептов (сид SAMPLE_SEED); пусто — весь датасет.
    # На сервере пока выборка: полный индекс — ≈195 ₽ и ≈80 минут (решение автора).
    index_sample: int | None = Field(default=None, gt=0)
    telegram_bot_token: SecretStr = SecretStr("")
    # У групп и каналов Telegram id отрицательный, поэтому int, а не положительное число.
    escalation_chat_id: int | None = None
    # Grocy — склад «что есть дома» (этап 6), только чтение. Адрес — с хоста;
    # контейнерам compose задаёт http://grocy/api сам. Ключ — пользователя Grocy
    # без прав; его заводит make grocy-seed. Пусто — интеграции нет.
    grocy_api_url: str = "http://localhost:9283/api"
    grocy_api_key: SecretStr = SecretStr("")
    log_level: LogLevel = "INFO"
