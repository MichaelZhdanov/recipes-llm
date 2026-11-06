"""LLM ассистента по настройкам: GigaChat или Claude (`LLM_PROVIDER`).

Здесь выбирается модель, которая разбирает вопрос и (с этапа 3) отвечает.
Судьи eval собираются той же фабрикой, но со своим провайдером и своим
HTTP-клиентом (`evals/run.py`, `_judge_llm`): по умолчанию Claude Sonnet 5.5.

HTTP-клиент у провайдеров свой: GigaChat доверяет только корню НУЦ
Минцифры, Claude — обычной цепочке сертификатов.
"""

import httpx2

from assistant.config import Settings
from assistant.generation import claude, cloudru, gigachat
from assistant.generation.gigachat import ChatModel

# cloud.ru — модель выбирается сравнением поисковиков; пока — лидер открытых
# моделей по Artificial Analysis Intelligence Index v4.0.
DEFAULT_MODELS = {
    "gigachat": gigachat.DEFAULT_MODEL,
    "anthropic": claude.DEFAULT_MODEL,
    "cloudru": "moonshotai/Kimi-K2.6",
}


def make_http_client(settings: Settings) -> httpx2.Client:
    if settings.llm_provider == "anthropic":
        return claude.make_http_client()
    if settings.llm_provider == "cloudru":
        return cloudru.make_http_client()
    return gigachat.make_http_client()


def make_chat_model(settings: Settings, client: httpx2.Client) -> ChatModel:
    if settings.llm_provider == "anthropic":
        return claude.Claude(
            api_key=settings.anthropic_api_key,
            api_url=settings.anthropic_api_url,
            client=client,
            effort=settings.llm_effort,
        )
    if settings.llm_provider == "cloudru":
        return cloudru.CloudRu(
            api_key=settings.cloud_ru_api_key,
            api_url=settings.cloud_ru_api_url,
            client=client,
            effort=settings.llm_effort,
        )
    return gigachat.make_llm(settings, client)


def llm_model(settings: Settings) -> str:
    return settings.llm_model or DEFAULT_MODELS[settings.llm_provider]
