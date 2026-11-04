import pytest

from assistant.cli import main


def test_index_without_voyage_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["index"])

    assert code == 1
    assert "VOYAGE_API_KEY" in capsys.readouterr().err


def test_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["nope"])
