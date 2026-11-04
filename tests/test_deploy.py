"""Скрипты сервера (deploy/): то, что можно проверить без сервера."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


@pytest.mark.parametrize(
    "command",
    [
        "",
        "bash",
        "rm -rf /",
        "deploy",
        "deploy abc123",
        "deploy " + "g" * 40,
        "deploy " + "a" * 40 + " ; rm -rf /",
        "deploy " + "a" * 40 + " extra",
    ],
)
def test_deploy_key_runs_nothing_but_deploy_sha(command: str, tmp_path: Path) -> None:
    """Ключ деплоя лежит в GitHub: даже с ним на сервере можно только задеплоить коммит."""
    result = subprocess.run(
        ["bash", str(DEPLOY / "deploy.sh")],
        env={
            "SSH_ORIGINAL_COMMAND": command,
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin",
        },
        input="WEB_PASSWORD=x\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "разрешена только команда" in result.stderr
    # Отказ — до любых действий: даже файл замка не создан.
    assert list(tmp_path.iterdir()) == []


def test_uv_image_is_the_same_everywhere() -> None:
    """uv на сервере берётся из того же образа, на котором собран ассистент."""
    image = re.search(r"^FROM (\S+)", (ROOT / "Dockerfile").read_text(), re.M)
    makefile = re.search(r"^UV_IMAGE := (\S+)", (ROOT / "Makefile").read_text(), re.M)
    bootstrap = re.search(
        r"^UV_IMAGE=(\S+)", (DEPLOY / "bootstrap.sh").read_text(), re.M
    )

    assert image and makefile and bootstrap
    assert image[1] == makefile[1] == bootstrap[1]
    assert "uv 0.13.5" in (DEPLOY / "bootstrap.sh").read_text()
    assert image[1].startswith("ghcr.io/astral-sh/uv:0.13.5-")


def test_nginx_template_substitutes_only_its_two_names() -> None:
    """envsubst получает список имён: $binary_remote_addr и прочее nginx остаются."""
    template = (DEPLOY / "nginx.conf.template").read_text()

    assert set(re.findall(r"\$\{(\w+)\}", template)) == {"DOMAIN", "ASSISTANT_PORT"}
    assert (
        "envsubst '${DOMAIN} ${ASSISTANT_PORT}'"
        in (DEPLOY / "bootstrap.sh").read_text()
    )
    assert re.search(r"location = /login \{[^}]*limit_req zone=pa_login", template)


def test_paths_agree_between_bootstrap_and_systemd() -> None:
    app_dir = re.search(r"^APP_DIR=(\S+)", (DEPLOY / "bootstrap.sh").read_text(), re.M)
    service = (DEPLOY / "paperless-refresh.service").read_text()

    assert app_dir
    assert f"ExecStart={app_dir[1]}/deploy/refresh.sh" in service
    assert f"WorkingDirectory={app_dir[1]}" in service
    assert "User=deploy" in service
