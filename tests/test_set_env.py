import io
import stat
from pathlib import Path

import pytest
from set_env import EnvValueError, main, parse_values, set_values

ENV = "# комментарий\nGIGACHAT_API_KEY=keep\nWEB_LOGIN=admin\nWEB_PASSWORD=old\n"


def test_replaces_values_and_keeps_the_rest() -> None:
    text = set_values(ENV, {"WEB_PASSWORD": "new$pass #1", "WEB_LOGIN": "demo"})

    assert text == (
        "# комментарий\nGIGACHAT_API_KEY=keep\nWEB_LOGIN='demo'\n"
        "WEB_PASSWORD='new$pass #1'\n"
    )


def test_missing_key_is_appended() -> None:
    assert set_values("A=1\n", {"WEB_LOGIN": "demo"}) == "A=1\nWEB_LOGIN='demo'\n"


@pytest.mark.parametrize(
    "line",
    [
        # Деплой не может переписать ключи API или секреты Paperless.
        "GIGACHAT_API_KEY=stolen",
        "PAPERLESS_SECRET_KEY=x",
        "WEB_PASSWORD=",
        "WEB_PASSWORD=it's",
        "garbage",
    ],
)
def test_only_allowed_non_empty_values(line: str) -> None:
    with pytest.raises(EnvValueError) as error:
        parse_values(line + "\n")

    # Значение (возможно, пароль) в сообщение об ошибке не попадает.
    value = line.partition("=")[2]
    assert not value or value not in str(error.value)


def test_main_writes_env_readable_only_by_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".env"
    target.write_text(ENV, encoding="utf-8")
    target.chmod(0o644)
    monkeypatch.setattr("sys.stdin", io.StringIO("WEB_PASSWORD=s3cret\n"))

    assert main([str(target)]) == 0

    assert "WEB_PASSWORD='s3cret'" in target.read_text(encoding="utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_main_rejects_bad_input_and_leaves_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".env"
    target.write_text(ENV, encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO("GIGACHAT_API_KEY=stolen\n"))

    assert main([str(target)]) == 2

    assert target.read_text(encoding="utf-8") == ENV
