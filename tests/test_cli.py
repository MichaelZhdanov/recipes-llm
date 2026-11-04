import pytest
from eval_helpers import MANIFEST

from assistant.cli import main, rebuild_reason
from assistant.retrieval.store import VectorStore


def test_index_without_embeddings_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["index"])

    assert code == 1
    error = capsys.readouterr().err
    assert "CLOUD_RU_API_KEY" in error
    assert "уровня «Проект»" in error


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["nope"])


def test_search_without_embeddings_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["search", "как включить OCR"])

    assert code == 1
    assert "CLOUD_RU_API_KEY" in capsys.readouterr().err


def test_ask_without_gigachat_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUD_RU_API_KEY", "test-key")

    code = main(["ask", "как включить OCR"])

    assert code == 1
    assert "GIGACHAT_API_KEY" in capsys.readouterr().err


def test_index_is_rebuilt_only_when_needed(store: VectorStore) -> None:
    """index --if-needed: для деплоя и таймера, пересборка стоит денег."""
    assert "не построен" in (
        rebuild_reason(store, ref="v3.4.2", model="fake-model") or ""
    )

    store.recreate(MANIFEST)

    assert rebuild_reason(store, ref="v3.4.2", model="fake-model") is None
    assert "v3.4.3" in (rebuild_reason(store, ref="v3.4.3", model="fake-model") or "")
    assert "other" in (rebuild_reason(store, ref="v3.4.2", model="other") or "")
