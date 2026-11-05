import io
import stat
from pathlib import Path

import pytest
from set_env import EnvValueError, main, parse_values, set_values

ENV = "# комментарий\nGIGACHAT_API_KEY=old\nLOG_LEVEL=INFO\nCLOUD_RU_API_KEY=old\n"


def test_replaces_values_and_keeps_the_rest() -> None:
    text = set_values(ENV, {"CLOUD_RU_API_KEY": "new$key #1", "GIGACHAT_API_KEY": "g"})

    assert text == (
        "# комментарий\nGIGACHAT_API_KEY='g'\nLOG_LEVEL=INFO\n"
        "CLOUD_RU_API_KEY='new$key #1'\n"
    )


def test_missing_key_is_appended() -> None:
    assert (
        set_values("A=1\n", {"GIGACHAT_API_KEY": "k"}) == "A=1\nGIGACHAT_API_KEY='k'\n"
    )


@pytest.mark.parametrize(
    "line",
    [
        # Остальные настройки деплой не меняет.
        "LOG_LEVEL=DEBUG",
        # Пароля входа на сайт больше нет: такой ключ — опечатка.
        "WEB_PASSWORD=x",
        "GIGACHAT_API_KEY=",
        "GIGACHAT_API_KEY=it's",
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
    monkeypatch.setattr("sys.stdin", io.StringIO("GIGACHAT_API_KEY=s3cret\n"))

    assert main([str(target)]) == 0

    assert "GIGACHAT_API_KEY='s3cret'" in target.read_text(encoding="utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_main_rejects_bad_input_and_leaves_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".env"
    target.write_text(ENV, encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO("LOG_LEVEL=DEBUG\n"))

    assert main([str(target)]) == 2

    assert target.read_text(encoding="utf-8") == ENV


def test_api_keys_from_github_replace_the_ones_on_the_server() -> None:
    values = parse_values("CLOUD_RU_API_KEY=new-cloud\nGIGACHAT_API_KEY=bmV3LWdpZ2E=\n")

    assert set_values(ENV, values) == (
        "# комментарий\nGIGACHAT_API_KEY='bmV3LWdpZ2E='\nLOG_LEVEL=INFO\n"
        "CLOUD_RU_API_KEY='new-cloud'\n"
    )


def test_anthropic_key_is_accepted_on_deploy() -> None:
    """LLM ассистента по умолчанию — Claude: ключ приходит из секретов GitHub."""
    values = parse_values("ANTHROPIC_API_KEY=sk-ant-test\n")

    assert set_values("A=1\n", values) == "A=1\nANTHROPIC_API_KEY='sk-ant-test'\n"
