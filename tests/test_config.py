import pytest
from pydantic import ValidationError

from assistant.config import Settings


def test_defaults_without_environment() -> None:
    settings = Settings()

    assert settings.llm_provider == "cloudru"
    assert settings.llm_model == ""
    assert settings.llm_effort == "low"
    # Проверку перед отправкой делал Sonnet, API Anthropic закрыт.
    assert settings.answer_review is False
    assert settings.gigachat_scope == "GIGACHAT_API_PERS"
    assert settings.dataset_revision == "ffaf49ee7439afe6dbf0653714ceacb9b57ff830"
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.embedding_provider == "cloudru"
    # Пустая модель: берётся модель провайдера по умолчанию.
    assert settings.embedding_model == ""
    assert settings.escalation_chat_id is None
    assert settings.gigachat_api_key.get_secret_value() == ""


def test_empty_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESCALATION_CHAT_ID", "")
    monkeypatch.setenv("DATASET_REVISION", "")

    settings = Settings()

    assert settings.escalation_chat_id is None
    assert settings.dataset_revision == "ffaf49ee7439afe6dbf0653714ceacb9b57ff830"


def test_negative_chat_id_for_telegram_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ESCALATION_CHAT_ID", "-1001234567890")

    assert Settings().escalation_chat_id == -1001234567890


def test_unknown_llm_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "chatgpt")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    "env_name", ["GIGACHAT_API_KEY", "CLOUD_RU_API_KEY", "TELEGRAM_BOT_TOKEN"]
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


def test_index_sample_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """INDEX_SAMPLE= (пусто) — весь датасет; число — выборка; ноль и минус — ошибка."""
    monkeypatch.setenv("INDEX_SAMPLE", "")
    assert Settings().index_sample is None

    monkeypatch.setenv("INDEX_SAMPLE", "10000")
    assert Settings().index_sample == 10000

    monkeypatch.setenv("INDEX_SAMPLE", "0")
    with pytest.raises(ValidationError):
        Settings()
