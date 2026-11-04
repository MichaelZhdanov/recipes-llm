import re
import stat
from pathlib import Path

from init_env import (
    GENERATED_SECRETS,
    add_missing_secrets,
    create_env,
    render_env,
    update_env,
)

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = """\
# комментарий
PAPERLESS_SECRET_KEY=
PAPERLESS_ADMIN_USER=admin
PAPERLESS_ADMIN_PASSWORD=
GIGACHAT_API_KEY=
WEB_PASSWORD=
WEB_SESSION_SECRET=
"""


def parse(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key] = value
    return pairs


def test_fills_only_generated_secrets() -> None:
    env = parse(render_env(TEMPLATE))

    assert env["PAPERLESS_SECRET_KEY"]
    assert env["PAPERLESS_ADMIN_PASSWORD"]
    assert env["PAPERLESS_ADMIN_USER"] == "admin"
    # Ключ к LLM скрипт не выдумывает: его вводит сам пользователь.
    assert env["GIGACHAT_API_KEY"] == ""


def test_keeps_comments_and_order() -> None:
    def keys(text: str) -> list[str]:
        return [line.partition("=")[0] for line in text.splitlines()]

    assert keys(render_env(TEMPLATE)) == keys(TEMPLATE)


def test_does_not_replace_value_that_is_already_set() -> None:
    assert (
        render_env("PAPERLESS_SECRET_KEY=keep-me\n") == "PAPERLESS_SECRET_KEY=keep-me\n"
    )


def test_each_run_generates_new_secrets() -> None:
    assert render_env(TEMPLATE) != render_env(TEMPLATE)


def test_generated_values_need_no_escaping() -> None:
    env = parse(render_env(TEMPLATE))

    for key in GENERATED_SECRETS:
        assert re.fullmatch(r"[A-Za-z0-9_-]{16,}", env[key]), key


def test_creates_file_readable_only_by_owner(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text(TEMPLATE, encoding="utf-8")
    target = tmp_path / ".env"

    assert create_env(example, target) is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert parse(target.read_text(encoding="utf-8"))["PAPERLESS_SECRET_KEY"]


def test_keeps_existing_env_untouched(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text(TEMPLATE, encoding="utf-8")
    target = tmp_path / ".env"
    target.write_text("MY_VALUE=1\n", encoding="utf-8")

    assert create_env(example, target) is False
    assert target.read_text(encoding="utf-8") == "MY_VALUE=1\n"


def test_real_env_example_leaves_generated_secrets_empty() -> None:
    example = parse((ROOT / ".env.example").read_text(encoding="utf-8"))
    rendered = parse(render_env((ROOT / ".env.example").read_text(encoding="utf-8")))

    for key in GENERATED_SECRETS:
        assert example[key] == "", f"{key} в .env.example должен быть пустым"
        assert rendered[key], f"{key} не сгенерирован"


def test_old_env_gets_new_secrets_and_keeps_its_values(tmp_path: Path) -> None:
    """.env создан до появления входа: в нём нет WEB_SESSION_SECRET, а WEB_PASSWORD пустой."""
    target = tmp_path / ".env"
    target.write_text(
        "# мой комментарий\nPAPERLESS_SECRET_KEY=keep-me\nWEB_PASSWORD=\n"
        "GIGACHAT_API_KEY=my-key\n",
        encoding="utf-8",
    )
    target.chmod(0o600)

    added = update_env(target)

    text = target.read_text(encoding="utf-8")
    env = parse(text)
    assert added == ["WEB_PASSWORD", "PAPERLESS_ADMIN_PASSWORD", "WEB_SESSION_SECRET"]
    assert env["PAPERLESS_SECRET_KEY"] == "keep-me"
    assert env["GIGACHAT_API_KEY"] == "my-key"
    assert env["WEB_PASSWORD"] and env["WEB_SESSION_SECRET"]
    assert text.startswith("# мой комментарий\nPAPERLESS_SECRET_KEY=keep-me\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_complete_env_is_not_rewritten(tmp_path: Path) -> None:
    text = "".join(f"{key}=set\n" for key in GENERATED_SECRETS)

    assert add_missing_secrets(text) == (text, [])
