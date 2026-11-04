import re

import httpx2
import pytest
from fastapi.testclient import TestClient

from assistant import __version__
from assistant.api import WEB_DIR, create_app
from assistant.config import Settings
from assistant.generation.answer import Answer, AnswerResult, Trace
from assistant.generation.gigachat import LlmError
from assistant.retrieval.embeddings import EmbeddingError
from assistant.retrieval.store import IndexMissingError

# Здесь проверяется API, а не вход: он выключен. Вход — tests/test_auth.py.
OPEN = Settings(no_auth=True)


def test_health_reports_version_and_paperless_ref() -> None:
    client = TestClient(create_app(Settings(paperless_ref="v9.9.9", no_auth=True)))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": __version__,
        "paperless_ref": "v9.9.9",
    }


def test_settings_come_from_environment_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAPERLESS_REF", "v1.2.3")
    monkeypatch.setenv("NO_AUTH", "1")
    client = TestClient(create_app())

    assert client.get("/health").json()["paperless_ref"] == "v1.2.3"


class FakeAnswerer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.questions: list[str] = []

    def answer(self, question: str) -> AnswerResult:
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
            paperless_ref="v3.4.2",
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
    client = TestClient(create_app(OPEN, answerer=fake))

    response = client.post("/ask", json={"question": "  как подключить 1С  "})

    assert response.status_code == 200
    assert response.json()["reason"] == "model_not_found"
    assert response.json()["model"] == "GigaChat-2:1.0"
    assert fake.questions == ["как подключить 1С"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_empty_or_huge_question_is_rejected(question: str) -> None:
    fake = FakeAnswerer()
    client = TestClient(create_app(OPEN, answerer=fake))

    assert client.post("/ask", json={"question": question}).status_code == 422
    assert fake.questions == []


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (IndexMissingError("индекс не построен"), "поиск по документации недоступен"),
        (
            EmbeddingError("cloud.ru ответил HTTP 402"),
            "поиск по документации недоступен",
        ),
        (LlmError("GigaChat ответил HTTP 402"), "модель GigaChat недоступна"),
        (httpx2.ConnectError("нет сети"), "внешний сервис не отвечает"),
    ],
)
def test_unavailable_dependency_is_503_without_details(
    error: Exception, detail: str
) -> None:
    client = TestClient(create_app(OPEN, answerer=FakeAnswerer(error)))

    response = client.post("/ask", json={"question": "?"})

    assert response.status_code == 503
    assert response.json() == {"detail": detail}


def test_without_keys_ask_is_503_but_health_works() -> None:
    """Настоящий ассистент собирается при первом вопросе: без ключа — 503, а не падение."""
    with TestClient(create_app(OPEN)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/ask", json={"question": "?"}).status_code == 503


@pytest.mark.parametrize(
    ("path", "media_type", "marker"),
    [
        ("/", "text/html", 'id="ask"'),
        ("/chat.js", "text/javascript", 'fetch("/ask"'),
        ("/chat.css", "text/css", "prefers-color-scheme"),
    ],
)
def test_chat_page_is_served_with_strict_headers(
    path: str, media_type: str, marker: str
) -> None:
    client = TestClient(create_app(OPEN, answerer=FakeAnswerer()))

    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert marker in response.text
    policy = response.headers["content-security-policy"]
    assert "default-src 'self'" in policy and "script-src 'self'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"


def test_chat_has_no_external_resources_or_innerhtml() -> None:
    """Страница работает без интернета и не вставляет HTML из ответа модели."""
    web = WEB_DIR
    page = "".join(
        (web / name).read_text() for name in ("index.html", "login.html", "chat.css")
    )
    script = (web / "chat.js").read_text() + (web / "login.js").read_text()

    assert "http://" not in page and "https://" not in page
    assert not re.search(r"\.innerHTML|insertAdjacentHTML\(|document\.write", script)
