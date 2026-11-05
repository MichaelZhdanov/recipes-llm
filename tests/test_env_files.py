"""Переменные окружения описаны в нескольких местах, тесты не дают им разойтись.

Источник правды — поля `Settings`. Каждое поле должно быть в `.env.example`
и в `environment` сервиса assistant в `docker-compose.yml`. Секреты в
`.env.example` пустые.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr

from assistant.config import Settings

ROOT = Path(__file__).resolve().parent.parent

# Адреса внутри docker-сети. В .env лежат адреса для запуска с хоста (localhost),
# поэтому контейнеру ассистента compose задаёт эти значения сам.
CONTAINER_OVERRIDES = {"QDRANT_URL": "http://qdrant:6333"}


def load_compose() -> dict[str, Any]:
    with (ROOT / "docker-compose.yml").open(encoding="utf-8") as compose_file:
        compose: dict[str, Any] = yaml.safe_load(compose_file)
    return compose


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


def test_assistant_gets_exactly_its_settings_in_compose() -> None:
    environment = load_compose()["services"]["assistant"]["environment"]

    assert set(environment) == setting_env_names()
    for name, value in environment.items():
        # Пустое значение по умолчанию: тогда Settings берёт своё значение по умолчанию.
        expected = CONTAINER_OVERRIDES.get(name, f"${{{name}:-}}")
        assert value == expected, name


def test_no_service_loads_whole_env_file() -> None:
    services = load_compose()["services"]

    with_env_file = [
        name for name, service in services.items() if "env_file" in service
    ]

    assert not with_env_file, f"env_file отдаёт сервису все секреты: {with_env_file}"


def test_ports_are_published_only_on_localhost() -> None:
    services = load_compose()["services"]

    for name, service in services.items():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), f"{name}: {port}"
