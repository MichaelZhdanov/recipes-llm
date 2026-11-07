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
    """Кроме Caddy: его 80 и 443 — единственные публичные порты (решение 82,
    test_only_caddy_is_public_and_it_proxies_grafana_and_grocy)."""
    services = load_compose()["services"]

    for name, service in services.items():
        if name == "caddy":
            continue
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:"), f"{name}: {port}"


def test_observability_runs_in_its_profiles_with_pinned_images_and_local_ports() -> (
    None
):
    """Prometheus и Grafana — профиль observability, Alertmanager — alerts
    (решения 80–81); образы — точной версией, порты — только 127.0.0.1."""
    services = load_compose()["services"]

    assert services["prometheus"]["profiles"] == ["observability"]
    assert services["grafana"]["profiles"] == ["observability"]
    assert services["alertmanager"]["profiles"] == ["alerts"]
    for name in ("prometheus", "grafana", "alertmanager"):
        image = services[name]["image"]
        assert ":" in image and not image.endswith(":latest"), image
        for port in services[name].get("ports", []):
            assert port.startswith("127.0.0.1:"), port
    # Порта метрик бота на хосте нет: Prometheus читает его из docker-сети.
    assert "ports" not in services["bot"]


def test_alertmanager_gets_telegram_secrets_as_files_not_in_its_config() -> None:
    compose = load_compose()
    config = (ROOT / "observability/alertmanager/alertmanager.yml").read_text(
        encoding="utf-8"
    )

    assert compose["secrets"] == {
        "telegram_bot_token": {"environment": "TELEGRAM_BOT_TOKEN"},
        "escalation_chat_id": {"environment": "ESCALATION_CHAT_ID"},
    }
    assert compose["services"]["alertmanager"]["secrets"] == [
        "telegram_bot_token",
        "escalation_chat_id",
    ]
    assert "bot_token_file: /run/secrets/telegram_bot_token" in config
    assert "chat_id_file: /run/secrets/escalation_chat_id" in config
    assert "bot_token:" not in config.replace("bot_token_file:", "")
    # Grafana — только пароль admin, без ключей ассистента.
    grafana_env = compose["services"]["grafana"]["environment"]
    assert not any("API_KEY" in name or "TOKEN" in name for name in grafana_env)


def test_dashboard_queries_only_metrics_the_code_exports() -> None:
    """Опечатка в имени метрики — пустая панель без ошибки: сверяем с кодом."""
    import json
    import re

    from prometheus_client import generate_latest

    from assistant.generation.gigachat import LlmStats
    from assistant.hostlog import HostLogMetrics
    from assistant.metrics import Metrics, UsageStats
    from assistant.retrieval.embeddings import EmbeddingStats

    usage = UsageStats("m", LlmStats(), "e", EmbeddingStats())
    exported = "".join(
        generate_latest(registry).decode()
        for registry in (
            Metrics(usage=lambda: usage).registry,
            HostLogMetrics().registry,
        )
    )
    names = set(re.findall(r"^# TYPE (\w+) ", exported, flags=re.M))
    known = names | {f"{n}_total" for n in names} | {f"{n}_bucket" for n in names}
    dashboards = sorted((ROOT / "observability/grafana/dashboards").glob("*.json"))
    assert [d.name for d in dashboards] == ["recipes.json", "server.json"]
    rules = (ROOT / "observability/prometheus/alerts.yml").read_text(encoding="utf-8")
    exprs = [rules] + [
        t["expr"]
        for d in dashboards
        for p in json.loads(d.read_text(encoding="utf-8"))["panels"]
        for t in p["targets"]
    ]

    used = {m for e in exprs for m in re.findall(r"\b(recipes_\w+)", e)}
    assert used and used <= known, used - known


def test_only_caddy_is_public_and_it_proxies_grafana_and_grocy() -> None:
    """Решение 82: публичные порты — только 80 и 443 у Caddy (профиль tls)."""
    services = load_compose()["services"]
    caddy = services["caddy"]
    caddyfile = (ROOT / "deploy/caddy/Caddyfile").read_text(encoding="utf-8")

    assert caddy["profiles"] == ["tls"]
    assert caddy["image"].startswith("docker.io/library/caddy:2.")
    assert sorted(caddy["ports"]) == ["443:443", "80:80"]
    assert "grocy" in caddy["networks"]
    assert (
        "{$GRAFANA_DOMAIN} {\n" in caddyfile
        and "\treverse_proxy grafana:3000" in caddyfile
    )
    assert "{$GROCY_DOMAIN} {\n\treverse_proxy grocy:80" in caddyfile
    grafana = services["grafana"]["environment"]
    assert (
        grafana["GF_SERVER_ROOT_URL"] == "${GRAFANA_DOMAIN:+https://${GRAFANA_DOMAIN}/}"
    )


def test_node_exporter_sees_the_host_but_listens_only_on_docker0() -> None:
    """Сеть хоста — чтобы видеть трафик сервера; без файрвола 0.0.0.0:9100 был
    бы открыт в интернет, поэтому — только мост docker0 (172.17.0.1)."""
    services = load_compose()["services"]
    node = services["node-exporter"]

    assert node["profiles"] == ["host"]
    assert node["network_mode"] == "host" and node["pid"] == "host"
    assert "--web.listen-address=172.17.0.1:9100" in node["command"]
    assert node["volumes"] == ["/:/host:ro,rslave"]
    assert "host.docker.internal:host-gateway" in services["prometheus"]["extra_hosts"]
    prometheus = (ROOT / "observability/prometheus/prometheus.yml").read_text(
        encoding="utf-8"
    )
    assert "targets: [host.docker.internal:9100]" in prometheus
    # Публичный /metrics Grafana закрыт: метрики читает только Prometheus.
    caddyfile = (ROOT / "deploy/caddy/Caddyfile").read_text(encoding="utf-8")
    assert "respond /metrics 404" in caddyfile


def test_hostlog_reads_host_logs_read_only_as_group_adm() -> None:
    hostlog = load_compose()["services"]["hostlog"]

    assert hostlog["profiles"] == ["host"]
    assert hostlog["volumes"] == ["/var/log:/host/log:ro"]
    assert hostlog["group_add"] == ["4"]
    assert "ports" not in hostlog
