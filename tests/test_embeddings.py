import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from assistant.retrieval.embeddings import EmbeddingError, VoyageEmbedder, batches

KEY = "pa-test-key-1234567890"


class FakeVoyage:
    """Подменяет API Voyage: запоминает запросы и отвечает векторами [индекс, длина]."""

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
    fake: FakeVoyage, sleeps: list[float] | None = None, key: str = KEY
) -> VoyageEmbedder:
    return VoyageEmbedder(
        api_key=SecretStr(key),
        model="voyage-4",
        base_url="https://api.voyageai.com/v1/",
        client=httpx2.Client(transport=httpx2.MockTransport(fake)),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_personal_data_never_reaches_the_api() -> None:
    fake = FakeVoyage()
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
    fake = FakeVoyage()

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
    fake = FakeVoyage()

    vector = make(fake).embed_query("вопрос")

    assert fake.bodies()[0]["input_type"] == "query"
    assert vector == [0.0, 6.0]


def test_documents_are_sent_in_batches_and_order_is_kept() -> None:
    fake = FakeVoyage(shuffle=True)
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
    fake = FakeVoyage(statuses=[429, 200])

    make(fake, sleeps).embed_query("x")

    assert len(fake.requests) == 2
    assert sleeps == [3.0]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_is_clear_and_does_not_leak_key(status: int) -> None:
    fake = FakeVoyage(statuses=[status])

    with pytest.raises(EmbeddingError, match=f"HTTP {status}") as error:
        make(fake).embed_query("x")

    # 403 бывает, когда ключ из одного кабинета, а адрес от другого.
    assert "VOYAGE_API_URL" in str(error.value)
    assert "ai.mongodb.com" in str(error.value)
    assert KEY not in str(error.value)


def test_other_errors_have_no_auth_hint() -> None:
    fake = FakeVoyage(statuses=[400])

    with pytest.raises(EmbeddingError, match="HTTP 400") as error:
        make(fake).embed_query("x")

    assert "VOYAGE_API_URL" not in str(error.value)


def test_missing_key_gives_clear_error() -> None:
    with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
        make(FakeVoyage(), key="")


def test_key_is_not_visible_in_repr() -> None:
    assert KEY not in repr(make(FakeVoyage()).__dict__)
