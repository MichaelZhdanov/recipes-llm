import pytest
from fastapi.testclient import TestClient

from assistant import __version__
from assistant.api import create_app
from assistant.config import Settings


def test_health_reports_version_and_paperless_ref() -> None:
    client = TestClient(create_app(Settings(paperless_ref="v9.9.9")))

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
    client = TestClient(create_app())

    assert client.get("/health").json()["paperless_ref"] == "v1.2.3"
