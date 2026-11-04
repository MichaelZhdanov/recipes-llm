import pytest

from assistant.config import Settings


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убирает переменные Settings из окружения перед каждым тестом.

    Иначе экспортированный в shell LLM_API_KEY или PAPERLESS_REF поменял бы
    значения по умолчанию, и тесты зависели бы от машины.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
