import pytest

from assistant.cli import main


@pytest.mark.parametrize("command", ["nope", "index", "sources"])
def test_unknown_command_is_rejected(command: str) -> None:
    """index вернётся на этапе 1 вместе с загрузчиком рецептов."""
    with pytest.raises(SystemExit):
        main([command])


def test_search_without_embeddings_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["search", "как включить OCR"])

    assert code == 1
    error = capsys.readouterr().err
    assert "CLOUD_RU_API_KEY" in error
    assert "уровня «Проект»" in error


def test_ask_without_gigachat_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUD_RU_API_KEY", "test-key")

    code = main(["ask", "как включить OCR"])

    assert code == 1
    assert "GIGACHAT_API_KEY" in capsys.readouterr().err
