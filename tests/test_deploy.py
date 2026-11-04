"""Скрипты сервера (deploy/): то, что можно проверить без сервера."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


def test_deploy_needs_a_commit(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(DEPLOY / "deploy.sh")],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "использование" in result.stderr
    # Без коммита ничего не делается: даже файл замка не создан.
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
    assert "User=root" in service


DEPLOY_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDeployKeyDeployKeyDeployKeyDeployKey00 ops@bypass"
ADMIN_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAdminKeyAdminKeyAdminKeyAdminKeyAdmin0 me@laptop"


def add_deploy_key(authorized_keys: Path, key: str) -> subprocess.CompletedProcess[str]:
    script = (
        f'source "{DEPLOY / "bootstrap.sh"}"; '
        f'AUTHORIZED_KEYS="{authorized_keys}"; add_deploy_key "$1"'
    )
    return subprocess.run(
        ["bash", "-c", script, "bash", key], capture_output=True, text=True, check=False
    )


def test_deploy_key_is_appended_and_root_keys_stay(tmp_path: Path) -> None:
    """Деплой идёт от root: перезаписать authorized_keys — значит закрыть себе вход."""
    keys = tmp_path / ".ssh" / "authorized_keys"
    keys.parent.mkdir()
    keys.write_text(ADMIN_KEY)  # без перевода строки в конце

    for _ in range(2):  # повторный запуск bootstrap ничего не дублирует
        assert add_deploy_key(keys, DEPLOY_KEY).returncode == 0

    assert keys.read_text().splitlines() == [ADMIN_KEY, DEPLOY_KEY]


def test_key_already_in_authorized_keys_is_left_as_is(tmp_path: Path) -> None:
    keys = tmp_path / "authorized_keys"
    keys.write_text(f"{DEPLOY_KEY}\n{ADMIN_KEY}\n")

    assert add_deploy_key(keys, DEPLOY_KEY).returncode == 0

    assert keys.read_text() == f"{DEPLOY_KEY}\n{ADMIN_KEY}\n"
