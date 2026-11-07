import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from assistant.config import CLOUD_RU_API_URL, Settings
from assistant.retrieval.embeddings import (
    CLOUD_RU,
    DEFAULT_MODELS,
    RETRIEVAL_TASK,
    VOYAGE,
    EmbeddingError,
    HttpEmbedder,
    batches,
    make_embedder,
    query_template,
)

KEY = "pa-test-key-1234567890"


class FakeApi:
    """Подменяет API эмбеддингов: запоминает запросы и отвечает векторами [индекс, длина]."""

    def __init__(
        self, statuses: list[int] | None = None, shuffle: bool = False
    ) -> None:
        self.requests: list[httpx2.Request] = []
        self.statuses = statuses or []
        self.shuffle = shuffle

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return httpx2.Response(
                    status, headers={"Retry-After": "3"}, text="limit"
                )
        body = json.loads(request.content)
        data = [
            {"index": i, "embedding": [float(i), float(len(text))]}
            for i, text in enumerate(body["input"])
        ]
        if self.shuffle:
            data.reverse()
        return httpx2.Response(200, json={"data": data, "usage": {"total_tokens": 7}})

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def make(
    fake: FakeApi, sleeps: list[float] | None = None, key: str = KEY
) -> HttpEmbedder:
    return HttpEmbedder(
        provider=VOYAGE,
        api_key=SecretStr(key),
        model="voyage-4",
        base_url="https://api.voyageai.com/v1/",
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_personal_data_never_reaches_the_api() -> None:
    fake = FakeApi()
    embedder = make(fake)

    embedder.embed_query(
        "я ivan@example.com, звоните +7 999 123-45-67, почему не работает OCR"
    )

    sent = fake.requests[0].content.decode()
    assert "ivan@example.com" not in sent
    assert "123-45-67" not in sent
    assert fake.bodies()[0]["input"] == [
        "я [EMAIL], звоните [PHONE], почему не работает OCR"
    ]
    assert embedder.stats.masked == {"EMAIL": 1, "PHONE": 1}


def test_request_shape_and_auth_header() -> None:
    fake = FakeApi()

    make(fake).embed_documents(["a", "b"])

    request = fake.requests[0]
    assert str(request.url) == "https://api.voyageai.com/v1/embeddings"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert fake.bodies()[0] == {
        "input": ["a", "b"],
        "model": "voyage-4",
        "input_type": "document",
    }


def test_query_uses_query_input_type() -> None:
    fake = FakeApi()

    vector = make(fake).embed_query("вопрос")

    assert fake.bodies()[0]["input_type"] == "query"
    assert vector == [0.0, 6.0]


def test_documents_are_sent_in_batches_and_order_is_kept() -> None:
    fake = FakeApi(shuffle=True)
    embedder = make(fake)
    embedder.max_batch_texts = 2

    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(b["input"]) for b in fake.bodies()] == [2, 2, 1]
    assert [v[1] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert embedder.stats.requests == 3
    assert embedder.stats.tokens == 21


def test_batches_respect_character_limit() -> None:
    assert list(batches(["aaaa", "bbbb", "cc"], max_texts=10, max_chars=8)) == [
        ["aaaa", "bbbb"],
        ["cc"],
    ]


def test_rate_limit_is_retried_after_retry_after() -> None:
    sleeps: list[float] = []
    fake = FakeApi(statuses=[429, 200])

    make(fake, sleeps).embed_query("x")

    assert len(fake.requests) == 2
    assert sleeps == [3.0]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_is_clear_and_does_not_leak_key(status: int) -> None:
    fake = FakeApi(statuses=[status])

    with pytest.raises(EmbeddingError, match=f"HTTP {status}") as error:
        make(fake).embed_query("x")

    assert error.value.status == status
    # 403 бывает, когда ключ из одного кабинета, а адрес от другого.
    assert "VOYAGE_API_URL" in str(error.value)
    assert "ai.mongodb.com" in str(error.value)
    assert KEY not in str(error.value)


def test_other_errors_have_no_auth_hint() -> None:
    fake = FakeApi(statuses=[400])

    with pytest.raises(EmbeddingError, match="HTTP 400") as error:
        make(fake).embed_query("x")

    assert "VOYAGE_API_URL" not in str(error.value)


def test_missing_key_gives_clear_error() -> None:
    with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
        make(FakeApi(), key="")


def test_key_is_not_visible_in_repr() -> None:
    assert KEY not in repr(make(FakeApi()).__dict__)


GIGA = "ai-sage/Giga-Embeddings-instruct-3B"


def make_cloudru(fake: FakeApi, model: str = GIGA) -> HttpEmbedder:
    return HttpEmbedder(
        provider=CLOUD_RU,
        api_key=SecretStr(KEY),
        model=model,
        base_url=CLOUD_RU_API_URL,
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        query_template=query_template(model),
        sleep=lambda _: None,
    )


def test_cloudru_query_gets_instruction_after_masking() -> None:
    fake = FakeApi()

    make_cloudru(fake).embed_query("я ivan@example.com, как настроить почту")

    body = fake.bodies()[0]
    assert "input_type" not in body
    assert body["input"] == [
        f"Instruct: {RETRIEVAL_TASK}\nQuery: я [EMAIL], как настроить почту"
    ]
    assert "ivan@example.com" not in fake.requests[0].content.decode()


def test_cloudru_documents_go_without_instruction() -> None:
    fake = FakeApi()

    make_cloudru(fake).embed_documents(["Usage > Mail\n\nFetch mail from IMAP."])

    assert str(fake.requests[0].url) == f"{CLOUD_RU_API_URL}/embeddings"
    assert fake.bodies()[0] == {
        "input": ["Usage > Mail\n\nFetch mail from IMAP."],
        "model": GIGA,
    }


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("ai-sage/Giga-Embeddings-instruct-480M", "Instruct: {task}\nQuery: {query}"),
        ("Qwen/Qwen3-Embedding-0.6B", None),
        ("BAAI/bge-m3", None),
        ("voyage-4", None),
    ],
)
def test_query_template_depends_on_model(model: str, expected: str | None) -> None:
    assert query_template(model) == expected


def test_model_without_instruction_sends_question_as_is() -> None:
    fake = FakeApi()

    make_cloudru(fake, model="BAAI/bge-m3").embed_query("как настроить почту")

    assert fake.bodies()[0]["input"] == ["как настроить почту"]


@pytest.mark.parametrize(
    ("status", "hint"), [(401, "уровня «Проект»"), (402, "Пополните баланс")]
)
def test_cloudru_errors_explain_what_to_check(status: int, hint: str) -> None:
    fake = FakeApi(statuses=[status])

    with pytest.raises(
        EmbeddingError, match=f"cloud.ru ответил HTTP {status}"
    ) as error:
        make_cloudru(fake).embed_query("x")

    assert hint in str(error.value)
    assert KEY not in str(error.value)


def test_make_embedder_picks_provider_and_its_default_model() -> None:
    with httpx2.Client() as http:
        cloudru = make_embedder(
            Settings(
                embedding_provider="cloudru",
                embedding_model="",
                cloud_ru_api_key=SecretStr(KEY),
            ),
            http,
        )
        voyage = make_embedder(
            Settings(
                embedding_provider="voyage",
                embedding_model="",
                voyage_api_key=SecretStr(KEY),
            ),
            http,
        )

    assert (cloudru.provider, cloudru.model) == (CLOUD_RU, DEFAULT_MODELS["cloudru"])
    # По итогам сравнения моделей вопрос уходит без инструкции.
    assert cloudru.query_template is None
    assert (voyage.provider, voyage.model) == (VOYAGE, "voyage-4")
    assert voyage.query_template is None


def test_explicit_model_wins_over_default() -> None:
    with httpx2.Client() as http:
        embedder = make_embedder(
            Settings(
                embedding_provider="cloudru",
                embedding_model="BAAI/bge-m3",
                cloud_ru_api_key=SecretStr(KEY),
            ),
            http,
        )

    assert embedder.model == "BAAI/bge-m3"
    assert embedder.query_template is None


def test_query_instruction_can_be_turned_on() -> None:
    """Для сравнения на eval: та же instruct-модель с задачей к вопросу."""
    with httpx2.Client() as http:
        embedder = make_embedder(
            Settings(
                embedding_model="ai-sage/Giga-Embeddings-instruct-480M",
                embedding_query_instruction=True,
                cloud_ru_api_key=SecretStr(KEY),
            ),
            http,
        )

    assert embedder.query_template == "Instruct: {task}\nQuery: {query}"
