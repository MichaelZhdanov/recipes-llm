from collections.abc import Sequence

import httpx2
import pytest
from fastapi.testclient import TestClient

from assistant import __version__
from assistant.api import create_app
from assistant.config import Settings
from assistant.generation.answer import Answer, AnswerResult, Progress, Trace
from assistant.generation.gigachat import LlmError, Message
from assistant.retrieval.embeddings import EmbeddingError
from assistant.retrieval.store import IndexMissingError

SETTINGS = Settings()


def test_health_reports_version_and_dataset_revision() -> None:
    client = TestClient(create_app(Settings(dataset_revision="v9.9.9")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "dataset_revision": "v9.9.9",
    }


def test_settings_come_from_environment_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASET_REVISION", "v1.2.3")
    client = TestClient(create_app())

    assert client.get("/health").json()["dataset_revision"] == "v1.2.3"


class FakeAnswerer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[str] = []

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
    ) -> AnswerResult:
        self.questions.append(question)
        if self.error:
            raise self.error
        answer = Answer(
            status="not_found",
            reason="model_not_found",
            statements=[],
            sources=[],
            model="GigaChat-2:1.0",
            prompt_version="answer_v1",
            dataset_revision="v3.4.2",
        )
        trace = Trace(
            hits=[],
            sources=[],
            completion=None,
            checked=[],
            retrieval_seconds=0.0,
            generation_seconds=0.0,
        )
        return AnswerResult(answer=answer, trace=trace)


def test_ask_returns_the_answer() -> None:
    fake = FakeAnswerer()
    client = TestClient(create_app(SETTINGS, answerer=fake))

    response = client.post("/ask", json={"question": "  как подключить 1С  "})

    assert response.status_code == 200
    assert response.json()["reason"] == "model_not_found"
    assert response.json()["model"] == "GigaChat-2:1.0"
    assert fake.questions == ["как подключить 1С"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_empty_or_huge_question_is_rejected(question: str) -> None:
    fake = FakeAnswerer()
    client = TestClient(create_app(SETTINGS, answerer=fake))

    assert client.post("/ask", json={"question": question}).status_code == 422
    assert fake.questions == []


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (IndexMissingError("индекс не построен"), "поиск недоступен"),
        (
            EmbeddingError("cloud.ru ответил HTTP 402"),
            "поиск недоступен",
        ),
        (LlmError("GigaChat ответил HTTP 402"), "модель недоступна"),
        (httpx2.ConnectError("нет сети"), "внешний сервис не отвечает"),
    ],
)
def test_unavailable_dependency_is_503_without_details(
    error: Exception, detail: str
) -> None:
    client = TestClient(create_app(SETTINGS, answerer=FakeAnswerer(error)))

    response = client.post("/ask", json={"question": "?"})

    assert response.status_code == 503
    assert response.json() == {"detail": detail}


def test_without_keys_ask_is_503_but_health_works() -> None:
    """Настоящий ассистент собирается при первом вопросе: без ключа — 503, а не падение."""
    with TestClient(create_app(SETTINGS)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/ask", json={"question": "?"}).status_code == 503


def test_chat_pages_and_login_are_gone() -> None:
    """Канал — Telegram: у HTTP API нет ни страниц чата, ни входа."""
    client = TestClient(create_app(SETTINGS, answerer=FakeAnswerer()))

    for path in ("/", "/chat.js", "/login", "/session"):
        assert client.get(path).status_code == 404, path


def test_lazy_answerer_passes_dialogue_history() -> None:
    """Бот передаёт историю диалога: ленивый ассистент отдаёт её дальше."""
    from assistant.service import LazyAnswerer

    seen: list[Sequence[Message]] = []

    class Inner:
        def answer(
            self,
            question: str,
            history: Sequence[Message] = (),
            progress: Progress | None = None,
        ) -> AnswerResult:
            seen.append(history)
            raise LlmError("нет сети")

    lazy = LazyAnswerer(Settings())
    lazy._answerer = Inner()
    history: list[Message] = [{"role": "user", "content": "блины"}]

    with pytest.raises(LlmError):
        lazy.answer("а во втором?", history)
    lazy.close()

    assert seen == [history]
