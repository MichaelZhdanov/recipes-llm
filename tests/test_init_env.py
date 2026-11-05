import re
import stat
from pathlib import Path

import init_env
import pytest
from init_env import add_missing_secrets, create_env, render_env, update_env

ROOT = Path(__file__).resolve().parent.parent

# Механизм проверяется на своих секретах: список проекта меняется от этапа к этапу.
PROJECT_SECRETS = dict(init_env.GENERATED_SECRETS)
SECRETS = {"APP_SECRET_KEY": 48, "APP_PASSWORD": 16, "SESSION_SECRET": 48}

TEMPLATE = """\
# комментарий
APP_SECRET_KEY=
APP_USER=admin
APP_PASSWORD=
GIGACHAT_API_KEY=
SESSION_SECRET=
"""


@pytest.fixture(autouse=True)
def fake_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(init_env, "GENERATED_SECRETS", SECRETS)


def parse(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key] = value
    return pairs


def test_fills_only_generated_secrets() -> None:
    env = parse(render_env(TEMPLATE))

    assert env["APP_SECRET_KEY"]
    assert env["APP_PASSWORD"]
    assert env["APP_USER"] == "admin"
    # Ключ к LLM скрипт не выдумывает: его вводит сам пользователь.
    assert env["GIGACHAT_API_KEY"] == ""


def test_keeps_comments_and_order() -> None:
    def keys(text: str) -> list[str]:
        return [line.partition("=")[0] for line in text.splitlines()]

    assert keys(render_env(TEMPLATE)) == keys(TEMPLATE)


def test_does_not_replace_value_that_is_already_set() -> None:
    assert render_env("APP_SECRET_KEY=keep-me\n") == "APP_SECRET_KEY=keep-me\n"


def test_each_run_generates_new_secrets() -> None:
    assert render_env(TEMPLATE) != render_env(TEMPLATE)


def test_generated_values_need_no_escaping() -> None:
    env = parse(render_env(TEMPLATE))

    for key in SECRETS:
        assert re.fullmatch(r"[A-Za-z0-9_-]{16,}", env[key]), key


def test_creates_file_readable_only_by_owner(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text(TEMPLATE, encoding="utf-8")
    target = tmp_path / ".env"

    assert create_env(example, target) is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert parse(target.read_text(encoding="utf-8"))["APP_SECRET_KEY"]


def test_keeps_existing_env_untouched(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text(TEMPLATE, encoding="utf-8")
    target = tmp_path / ".env"
    target.write_text("MY_VALUE=1\n", encoding="utf-8")

    assert create_env(example, target) is False
    assert target.read_text(encoding="utf-8") == "MY_VALUE=1\n"


def test_real_env_example_leaves_generated_secrets_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Здесь нужен настоящий список секретов проекта.
    monkeypatch.setattr(init_env, "GENERATED_SECRETS", PROJECT_SECRETS)
    example = parse((ROOT / ".env.example").read_text(encoding="utf-8"))
    rendered = parse(render_env((ROOT / ".env.example").read_text(encoding="utf-8")))

    for key in PROJECT_SECRETS:
        assert example[key] == "", f"{key} в .env.example должен быть пустым"
        assert rendered[key], f"{key} не сгенерирован"


def test_old_env_gets_new_secrets_and_keeps_its_values(tmp_path: Path) -> None:
    """.env из старой версии: в нём нет SESSION_SECRET, а APP_PASSWORD пустой."""
    target = tmp_path / ".env"
    target.write_text(
        "# мой комментарий\nAPP_SECRET_KEY=keep-me\nAPP_PASSWORD=\n"
        "GIGACHAT_API_KEY=my-key\n",
        encoding="utf-8",
    )
    target.chmod(0o600)

    added = update_env(target)

    text = target.read_text(encoding="utf-8")
    env = parse(text)
    assert added == ["APP_PASSWORD", "SESSION_SECRET"]
    assert env["APP_SECRET_KEY"] == "keep-me"
    assert env["GIGACHAT_API_KEY"] == "my-key"
    assert env["APP_PASSWORD"] and env["SESSION_SECRET"]
    assert text.startswith("# мой комментарий\nAPP_SECRET_KEY=keep-me\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_complete_env_is_not_rewritten(tmp_path: Path) -> None:
    text = "".join(f"{key}=set\n" for key in SECRETS)

    assert add_missing_secrets(text) == (text, [])
