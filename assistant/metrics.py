"""Метрики Prometheus (этап 8, F8): что происходит с ботом и сколько это стоит.

Счётчики событий считаются из тех же событий, что пишет журнал
(`bot/journal.py`): `Journal` передаёт каждое событие в `Metrics.record`.
Поэтому журнал и метрики не расходятся, а место вызова в коде бота одно.
Токены и цена — не события, а накопленные счётчики адаптеров (`LlmStats`,
`EmbeddingStats`): их на каждом запросе к `/metrics` читает
`UsageCollector`.

Метки — только перечислимые значения (источник, статус, повод, оценка,
модель): ни chat_id, ни текстов вопросов. Иначе метрика с меткой на
пользователя росла бы без предела и уносила бы ПДн в Prometheus.

У каждого процесса свой реестр (`CollectorRegistry`), а не глобальный:
тесты не делят счётчики друг с другом. Бот отдаёт `/metrics` на
`METRICS_PORT` внутри docker-сети (порт на хост не публикуется — его
читает Prometheus из той же сети), HTTP API — на своём `/metrics`.
"""

import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import httpx2
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)
from prometheus_client.core import CounterMetricFamily, Metric
from prometheus_client.registry import Collector

from assistant.generation.cloudru import cost_rub
from assistant.generation.gigachat import LlmError, LlmStats
from assistant.indexing.build import PRICE_PER_MTOK
from assistant.retrieval.embeddings import EmbeddingError, EmbeddingStats

logger = logging.getLogger(__name__)

METRICS_PORT = 9108
# Ответ бота — от секунд до пары минут (разбор + ответ модели).
SECONDS_BUCKETS = (1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180)
# Окна лимита частоты (bot/limits.py) — подписи вместо секунд.
WINDOWS = {600.0: "10m", 86400.0: "24h"}


@dataclass(frozen=True)
class UsageStats:
    """Накопленные счётчики адаптеров ассистента: LLM (разбор и ответ — один
    клиент) и эмбеддинги."""

    llm_model: str
    llm: LlmStats
    embedding_model: str
    embeddings: EmbeddingStats


def dependency_kind(error: Exception) -> str:
    """Чей сбой: balance — кончились рубли (402 или 403 «Project not found» у
    cloud.ru), llm, embeddings, search (Qdrant, индекс), network."""
    status = getattr(error, "status", None)
    if status == 402 or (status == 403 and "Project not found" in str(error)):
        return "balance"
    if isinstance(error, LlmError):
        return "llm"
    if isinstance(error, EmbeddingError):
        return "embeddings"
    if isinstance(error, httpx2.TransportError):
        return "network"
    return "search"


class UsageCollector(Collector):
    """Токены и цена по каталогу cloud.ru — из счётчиков адаптеров."""

    def __init__(self, usage: Callable[[], UsageStats | None]) -> None:
        self._usage = usage

    def collect(self) -> Iterator[Metric]:
        requests = CounterMetricFamily(
            "recipes_llm_requests",
            "Запросы к LLM (разбор вопроса и ответ)",
            labels=["model"],
        )
        tokens = CounterMetricFamily(
            "recipes_llm_tokens",
            "Токены LLM: prompt, completion и cached — часть prompt из кэша",
            labels=["model", "kind"],
        )
        embedding = CounterMetricFamily(
            "recipes_embedding_tokens", "Токены эмбеддингов вопросов", labels=["model"]
        )
        cost = CounterMetricFamily(
            "recipes_cost_rub",
            "Цена по каталогу cloud.ru, ₽; кэш промпта — по полной цене (верхняя граница)",
            labels=["model"],
        )
        usage = self._usage()
        if usage is not None:
            llm = usage.llm
            requests.add_metric([usage.llm_model], llm.requests)
            for kind, value in (
                ("prompt", llm.prompt_tokens),
                ("completion", llm.completion_tokens),
                ("cached", llm.precached_prompt_tokens),
            ):
                tokens.add_metric([usage.llm_model, kind], value)
            embedding.add_metric([usage.embedding_model], usage.embeddings.tokens)
            llm_rub = cost_rub(usage.llm_model, llm)
            if llm_rub is not None:
                cost.add_metric([usage.llm_model], llm_rub)
            price, currency = PRICE_PER_MTOK.get(usage.embedding_model, (0.0, ""))
            if currency == "₽":
                cost.add_metric(
                    [usage.embedding_model], usage.embeddings.tokens * price / 1e6
                )
        yield from (requests, tokens, embedding, cost)


class Metrics:
    def __init__(
        self,
        usage: Callable[[], UsageStats | None] = lambda: None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        r = self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "recipes_requests",
            "Вопросы к модели: источник (all, stock, near, api) и итог",
            ["source", "status"],
            registry=r,
        )
        self.answer_seconds = Histogram(
            "recipes_answer_seconds",
            "Время ответа целиком",
            buckets=SECONDS_BUCKETS,
            registry=r,
        )
        self.stage_seconds = Histogram(
            "recipes_stage_seconds",
            "Время стадии: retrieval — разбор вопроса и поиск, generation — ответ модели",
            ["stage"],
            buckets=SECONDS_BUCKETS,
            registry=r,
        )
        self.refusals = Counter(
            "recipes_refusals", "«Не нашёл» по причине", ["reason"], registry=r
        )
        self.escalations = Counter(
            "recipes_escalations",
            "Обращения к человеку по поводу",
            ["reason"],
            registry=r,
        )
        self.votes = Counter("recipes_votes", "Оценки ответов", ["vote"], registry=r)
        self.vote_reasons = Counter(
            "recipes_vote_reasons", "Что не так после 👎", ["reason"], registry=r
        )
        self.ticket_votes = Counter(
            "recipes_ticket_votes", "Оценки разговора с человеком", ["vote"], registry=r
        )
        self.rate_limited = Counter(
            "recipes_rate_limited",
            "Срабатывания лимита частоты",
            ["window"],
            registry=r,
        )
        self.tool_calls = Counter(
            "recipes_tool_calls",
            "Вызовы: search_recipes — разбор вопроса в поиск, grocy_stock — склад",
            ["tool", "result"],
            registry=r,
        )
        self.dependency_errors = Counter(
            "recipes_dependency_errors",
            "Сбои внешних API; balance — кончились рубли на cloud.ru",
            ["kind"],
            registry=r,
        )
        self.heartbeat = Gauge(
            "recipes_telegram_heartbeat_timestamp_seconds",
            "Когда Telegram последний раз ответил боту на getMe (unix-время)",
            registry=r,
        )
        r.register(UsageCollector(usage))

    def record(self, event: str, fields: Mapping[str, object]) -> None:
        """Событие журнала → счётчики (см. docstring модуля)."""
        if event == "answer":
            self._answer(fields)
        elif event == "ticket_opened":
            self.escalations.labels(str(fields["reason"])).inc()
        elif event == "vote" and "vote" in fields:
            self.votes.labels(str(fields["vote"])).inc()
        elif event == "vote":
            self.vote_reasons.labels(str(fields["reason"])).inc()
        elif event == "ticket_vote":
            self.ticket_votes.labels(str(fields["vote"])).inc()
        elif event == "limited":
            window = float(str(fields["window"]))
            self.rate_limited.labels(WINDOWS.get(window, str(window))).inc()

    def _answer(self, fields: Mapping[str, object]) -> None:
        status, source = str(fields["status"]), str(fields["source"])
        self.requests.labels(source, status).inc()
        for name, histogram in (
            ("seconds", self.answer_seconds),
            ("retrieval_seconds", self.stage_seconds.labels("retrieval")),
            ("generation_seconds", self.stage_seconds.labels("generation")),
        ):
            value = fields.get(name)
            # None — стадии не было (сбой до ответа модели).
            if isinstance(value, int | float):
                histogram.observe(value)
        if status == "not_found":
            self.refusals.labels(str(fields.get("reason"))).inc()
        if fields.get("search") is not None:
            self.tool_calls.labels("search_recipes", "ok").inc()
        if source in ("stock", "near"):
            self.tool_calls.labels("grocy_stock", "ok").inc()
        if fields.get("stock_down"):
            self.tool_calls.labels("grocy_stock", "error").inc()
            self.dependency_errors.labels("grocy").inc()
        if status == "error":
            self.dependency_errors.labels(str(fields.get("error_kind"))).inc()

    def text(self) -> bytes:
        """Метрики в текстовом формате Prometheus (для /metrics HTTP API)."""
        return generate_latest(self.registry)

    def serve(self, port: int = METRICS_PORT) -> None:
        """/metrics бота: отдельный HTTP-сервер в фоновом потоке."""
        start_http_server(port, registry=self.registry)
        logger.info("метрики Prometheus на порту %s", port)
