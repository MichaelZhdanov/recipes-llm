"""Переменные окружения описаны в нескольких местах, тесты не дают им разойтись.

Источник правды — поля `Settings`. Каждое поле должно быть в `.env.example`,
а секреты в `.env.example` должны быть пустыми.
"""

from pathlib import Path

from pydantic import SecretStr

from assistant.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def read_env_file(path: Path) -> dict[str, str]:
    """Пары KEY=value без комментариев и пустых строк."""
    pairs: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pairs[key] = value
    return pairs


def setting_env_names() -> set[str]:
    return {name.upper() for name in Settings.model_fields}


def secret_env_names() -> set[str]:
    return {
        name.upper()
        for name, field in Settings.model_fields.items()
        if field.annotation is SecretStr
    }


def test_env_example_lists_every_setting() -> None:
    missing = setting_env_names() - read_env_file(ROOT / ".env.example").keys()

    assert not missing, f"нет в .env.example: {sorted(missing)}"


def test_env_example_has_no_secret_values() -> None:
    example = read_env_file(ROOT / ".env.example")

    filled = {name for name in secret_env_names() if example.get(name)}

    assert not filled, f"в .env.example заполнены секреты: {sorted(filled)}"
