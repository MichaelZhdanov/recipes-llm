import re
import stat
from pathlib import Path

from init_env import GENERATED_SECRETS, create_env, render_env

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = """\
# комментарий
PAPERLESS_SECRET_KEY=
PAPERLESS_ADMIN_USER=admin
PAPERLESS_ADMIN_PASSWORD=
GIGACHAT_API_KEY=
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
