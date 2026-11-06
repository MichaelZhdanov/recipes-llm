"""Настройки ассистента из переменных окружения.

Читается только окружение процесса, без файла `.env`. В Docker переменные
передаёт compose, а на тесты не влияет чей-то локальный `.env`.
Список переменных с пояснениями лежит в `.env.example`.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# LLM ассистента: GigaChat (данные в РФ) или Claude через API Anthropic.
LlmProvider = Literal["gigachat", "anthropic", "cloudru"]
# Сколько думает Claude; пусто — по умолчанию модели. У GigaChat не используется.
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

    # По умолчанию Claude Haiku 5.5: на разборе вопроса в фильтры она прошла
    # пороги этапа 2, GigaChat-2-Pro — нет (evals/reports/2026-11-05-stage2.md).
    llm_provider: LlmProvider = "anthropic"
    # Пустая модель — модель провайдера по умолчанию (assistant/generation/llm.py).
    llm_model: str = ""
    # medium и high дали 0.92 и 0.93 — в пределах разброса; high думает дольше.
    llm_effort: LlmEffort = "medium"
    # Режим по умолчанию выберет сравнение always и auto в eval этапа 3.
    answer_mode: AnswerMode = "always"
    # Проверка ответа Sonnet перед отправкой и одна доработка Haiku (review.py).
    answer_review: bool = True
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
    log_level: LogLevel = "INFO"
