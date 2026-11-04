import pytest
from pydantic import ValidationError

from assistant.config import Settings


def test_defaults_without_environment() -> None:
    settings = Settings()

    assert settings.llm_provider == "gigachat"
    assert settings.llm_model == ""
    assert settings.gigachat_scope == "GIGACHAT_API_PERS"
    assert settings.paperless_ref == "v3.4.2"
    assert settings.paperless_api_url == "http://localhost:8000"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.embedding_provider == "cloudru"
    # Пустая модель: берётся модель провайдера по умолчанию.
    assert settings.embedding_model == ""
    assert settings.engineer_telegram_ids == frozenset()
    assert settings.escalation_chat_id is None
    assert settings.gigachat_api_key.get_secret_value() == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1, 2,3", {1, 2, 3}),
        ("42", {42}),
        (" 7 , ,8 ", {7, 8}),
    ],
)
def test_engineer_ids_are_parsed_from_comma_separated_string(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: set[int]
) -> None:
    monkeypatch.setenv("ENGINEER_TELEGRAM_IDS", raw)

    assert Settings().engineer_telegram_ids == frozenset(expected)


def test_engineer_ids_reject_non_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENGINEER_TELEGRAM_IDS", "1,abc")

    with pytest.raises(ValidationError):
        Settings()


def test_empty_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESCALATION_CHAT_ID", "")
    monkeypatch.setenv("PAPERLESS_REF", "")
    monkeypatch.setenv("ENGINEER_TELEGRAM_IDS", "")

    settings = Settings()

    assert settings.escalation_chat_id is None
    assert settings.paperless_ref == "v3.4.2"
    assert settings.engineer_telegram_ids == frozenset()


def test_negative_chat_id_for_telegram_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESCALATION_CHAT_ID", "-1001234567890")

    assert Settings().escalation_chat_id == -1001234567890


def test_unknown_llm_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "chatgpt")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    "env_name", ["GIGACHAT_API_KEY", "PAPERLESS_TOKEN", "TELEGRAM_BOT_TOKEN"]
)
def test_secrets_are_hidden_in_repr_and_dumps(
    monkeypatch: pytest.MonkeyPatch, env_name: str
) -> None:
    secret = "do-not-leak-1234567890"
    monkeypatch.setenv(env_name, secret)

    settings = Settings()

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in settings.model_dump_json()
    assert getattr(settings, env_name.lower()).get_secret_value() == secret
