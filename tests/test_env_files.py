"""Переменные окружения описаны в нескольких местах, тесты не дают им разойтись.

Источник правды — поля `Settings`. Каждое поле должно быть в `.env.example`
и в `environment` сервисов assistant и bot в `docker-compose.yml`. Секреты в
`.env.example` пустые.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import SecretStr

from assistant.config import Settings

ROOT = Path(__file__).resolve().parent.parent

# Адреса внутри docker-сети. В .env лежат адреса для запуска с хоста (localhost),
# поэтому контейнеру ассистента compose задаёт эти значения сам.
CONTAINER_OVERRIDES = {
    "QDRANT_URL": "http://qdrant:6333",
    "GROCY_API_URL": "http://grocy/api",
}


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


@pytest.mark.parametrize("service", ["assistant", "bot"])
def test_assistant_and_bot_get_exactly_the_settings_in_compose(service: str) -> None:
    environment = load_compose()["services"][service]["environment"]

    assert set(environment) == setting_env_names()
    for name, value in environment.items():
        # Пустое значение по умолчанию: тогда Settings берёт своё значение по умолчанию.
        expected = CONTAINER_OVERRIDES.get(name, f"${{{name}:-}}")
        assert value == expected, name


def test_api_and_bot_are_profiles_qdrant_always_runs() -> None:
    services = load_compose()["services"]

    assert services["assistant"]["profiles"] == ["api"]
    assert services["bot"]["profiles"] == ["bot"]
    assert "profiles" not in services["qdrant"]
    assert read_env_file(ROOT / ".env.example")["COMPOSE_PROFILES"] == "api"


def load_grocy_compose() -> dict[str, Any]:
    with (ROOT / "deploy" / "grocy" / "docker-compose.yml").open(encoding="utf-8") as f:
        compose: dict[str, Any] = yaml.safe_load(f)
    return compose


def test_grocy_is_a_separate_project_joined_by_a_shared_network() -> None:
    """Склад Grocy — свой compose-проект (на сервере /opt/grocy); ассистент,
    бот и сид видят его через внешнюю сеть grocy."""
    grocy = load_grocy_compose()
    services = load_compose()["services"]

    assert "grocy" not in services
    assert grocy["name"] == "grocy"
    assert grocy["networks"]["default"] == {"name": "grocy", "external": True}
    assert load_compose()["networks"]["grocy"] == {"name": "grocy", "external": True}
    for name in ("assistant", "bot", "grocy-seed"):
        assert "grocy" in services[name]["networks"], name
    # Сид пишет ключ в базу Grocy: uid как у пользователя образа (Dockerfile).
    assert grocy["services"]["grocy"]["environment"]["PUID"] == "10001"
    for port in grocy["services"]["grocy"]["ports"]:
        assert port.startswith("127.0.0.1:"), port


def test_grocy_seed_alone_gets_the_admin_password() -> None:
    """Сид — свой профиль; пароль admin Grocy получает только он."""
    services = load_compose()["services"]

    assert services["grocy-seed"]["profiles"] == ["grocy-seed"]
    assert "GROCY_ADMIN_PASSWORD" in services["grocy-seed"]["environment"]
    for name in ("assistant", "bot"):
        assert "GROCY_ADMIN_PASSWORD" not in services[name]["environment"]


def test_bot_keeps_its_database_in_a_volume_and_has_a_healthcheck() -> None:
    bot = load_compose()["services"]["bot"]

    assert "bot-data:/app/data/bot" in bot["volumes"]
    assert bot["healthcheck"]["test"][-2:] == ["-m", "assistant.bot.health"]


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
