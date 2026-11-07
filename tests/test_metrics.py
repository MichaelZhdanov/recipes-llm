"""Метрики Prometheus (этап 8, F8): счётчики из событий журнала, виды сбоев,
токены и цена, /metrics у API и в метках — ни пользователей, ни текстов."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import httpx2
import pytest
from chunk_helpers import CHICKEN
from fastapi.testclient import TestClient
from prometheus_client import generate_latest
from pydantic import SecretStr

from assistant.api import create_app
from assistant.bot.app import heartbeat
from assistant.bot.journal import Journal
from assistant.bot.pages import PageChecker
from assistant.bot.service import BotService
from assistant.bot.store import Store
from assistant.config import Settings
from assistant.generation.answer import (
    Answer,
    AnswerResult,
    Progress,
    SourceLink,
    StockRequest,
    Trace,
)
from assistant.generation.gigachat import LlmError, LlmStats, Message, Usage
from assistant.grocy.client import GrocyError, StockItem
from assistant.metrics import Metrics, UsageStats, dependency_kind
from assistant.retrieval.embeddings import EmbeddingError, EmbeddingStats
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.store import Hit
from assistant.tools.search_recipes import SearchArgs

CHAT = 777123
QUESTION = "курица секретный-вопрос-xyz"


class Answerer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        if self.error is not None:
            raise self.error
        answer = Answer(
            status="answer",
            reason=None,
            message="",
            statements=[],
            sources=[
                SourceLink(
                    number=1, recipe_id=CHICKEN.recipe_id, name="К", url=CHICKEN.url
                )
            ],
            model="zai-org/GLM-5.1",
            prompt_version="recipes_v5",
            dataset_revision="ffaf49ee",
        )
        args = SearchArgs("курица", IngredientFilter())
        trace = Trace([Hit(CHICKEN, 0.9)], [], None, [], 2.0, 9.0, args=args)
        return AnswerResult(answer, trace)


def bot(answerer: Answerer, metrics: Metrics, stock: object = None) -> BotService:
    store = Store(":memory:")
    http = httpx2.Client(transport=httpx2.MockTransport(lambda r: httpx2.Response(200)))
    return BotService(
        answerer,
        store,
        PageChecker(http, store),
        recipes=lambda recipe_id: None,
        make_steps=None,
        fetch_page=lambda url: None,
        fetch_photo=lambda url: None,
        stock=stock,  # type: ignore[arg-type]
        journal=Journal(
            SecretStr("k"), write=lambda line: None, listeners=[metrics.record]
        ),
    )


def value(metrics: Metrics, name: str, **labels: str) -> float:
    return metrics.registry.get_sample_value(name, labels) or 0.0


def test_answer_counts_request_latency_stages_and_the_parse() -> None:
    metrics = Metrics()

    bot(Answerer(), metrics).ask(CHAT, QUESTION)

    assert value(metrics, "recipes_requests_total", source="all", status="answer") == 1
    assert value(metrics, "recipes_answer_seconds_count") == 1
    assert value(metrics, "recipes_stage_seconds_sum", stage="retrieval") == 2.0
    assert value(metrics, "recipes_stage_seconds_sum", stage="generation") == 9.0
    assert (
        value(metrics, "recipes_tool_calls_total", tool="search_recipes", result="ok")
        == 1
    )


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (LlmError("cloud.ru ответил HTTP 402: …", status=402), "balance"),
        (
            LlmError("cloud.ru ответил HTTP 403: Project not found", status=403),
            "balance",
        ),
        (
            EmbeddingError("cloud.ru ответил HTTP 403: Project not found", status=403),
            "balance",
        ),
        (LlmError("cloud.ru ответил HTTP 500", status=500), "llm"),
        (EmbeddingError("HTTP 400", status=400), "embeddings"),
        (httpx2.ConnectError("нет сети"), "network"),
    ],
)
def test_failures_are_counted_by_kind_and_balance_is_separate(
    error: Exception, kind: str
) -> None:
    metrics = Metrics()

    bot(Answerer(error), metrics).ask(CHAT, QUESTION)

    assert dependency_kind(error) == kind
    assert value(metrics, "recipes_requests_total", source="all", status="error") == 1
    assert value(metrics, "recipes_dependency_errors_total", kind=kind) == 1


def test_grocy_down_is_a_failed_tool_call() -> None:
    metrics = Metrics()

    def down() -> list[StockItem]:
        raise GrocyError("Grocy недоступен")

    service = bot(Answerer(), metrics, stock=down)
    service.connect_grocy(CHAT)
    service.ask(CHAT, QUESTION)
    service.choose_source(CHAT, "stock")

    assert (
        value(metrics, "recipes_tool_calls_total", tool="grocy_stock", result="error")
        == 1
    )
    assert value(metrics, "recipes_dependency_errors_total", kind="grocy") == 1


def test_votes_escalations_and_limits_come_from_journal_events() -> None:
    metrics = Metrics()

    metrics.record("vote", {"answer_id": 1, "vote": "down"})
    metrics.record("vote", {"answer_id": 1, "reason": "wrong"})
    metrics.record("ticket_opened", {"ticket": 3, "reason": "health"})
    metrics.record("ticket_vote", {"ticket": 3, "vote": "up"})
    metrics.record("limited", {"window": 600.0, "wait": 120})

    assert value(metrics, "recipes_votes_total", vote="down") == 1
    assert value(metrics, "recipes_vote_reasons_total", reason="wrong") == 1
    assert value(metrics, "recipes_escalations_total", reason="health") == 1
    assert value(metrics, "recipes_ticket_votes_total", vote="up") == 1
    assert value(metrics, "recipes_rate_limited_total", window="10m") == 1


def test_tokens_and_cost_are_read_from_adapter_counters() -> None:
    llm = LlmStats()
    llm.add(Usage(1_000_000, 100_000, 400_000, 1_100_000))
    embeddings = EmbeddingStats(requests=1, tokens=1_000_000)
    usage = UsageStats(
        "zai-org/GLM-5.1", llm, "ai-sage/Giga-Embeddings-instruct-3B", embeddings
    )
    metrics = Metrics(usage=lambda: usage)

    labels = {"model": "zai-org/GLM-5.1"}
    assert value(metrics, "recipes_llm_requests_total", **labels) == 1
    assert (
        value(metrics, "recipes_llm_tokens_total", kind="cached", **labels) == 400_000
    )
    # Каталог cloud.ru: 198,86 ₽ за 1M на входе и 829,6 ₽ на выходе; кэш — по
    # полной цене: 198,86 + 0,1 × 829,6.
    assert value(metrics, "recipes_cost_rub_total", **labels) == pytest.approx(281.82)
    embedding = {"model": "ai-sage/Giga-Embeddings-instruct-3B"}
    assert value(metrics, "recipes_embedding_tokens_total", **embedding) == 1_000_000
    assert value(metrics, "recipes_cost_rub_total", **embedding) == pytest.approx(32.22)
    # Ассистент ещё не создан — метрики токенов пустые, а не ошибка.
    assert b"recipes_llm_tokens_total{" not in generate_latest(Metrics().registry)


def test_no_user_or_question_in_metrics() -> None:
    metrics = Metrics()
    bot(Answerer(), metrics).ask(CHAT, QUESTION)

    text = generate_latest(metrics.registry).decode()

    assert str(CHAT) not in text and "секретный" not in text


def test_api_serves_metrics_and_counts_its_answers() -> None:
    client = TestClient(create_app(Settings(), answerer=Answerer()))

    client.post("/ask", json={"question": "курица"})
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'recipes_requests_total{source="api",status="answer"} 1.0' in response.text


def test_heartbeat_updates_the_metric_after_telegram_answers(tmp_path: Path) -> None:
    metrics = Metrics()

    class FakeBot:
        async def get_me(self) -> None:
            return None

    async def scenario() -> None:
        task = asyncio.create_task(
            heartbeat(FakeBot(), tmp_path / "beat", every=60, metrics=metrics)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(scenario())

    assert value(metrics, "recipes_telegram_heartbeat_timestamp_seconds") > 0
    assert (tmp_path / "beat").exists()
