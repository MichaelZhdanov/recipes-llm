import pytest

from assistant.cli import main


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
