"""Скрипты сервера (deploy/): то, что можно проверить без сервера."""

import re
import subprocess
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    ("line", "ok"),
    [
        (DEPLOY_KEY, True),
        (
            "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTY= deploy",
            True,
        ),
        ("ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC=", True),
        # Перепутали файлы: приватный ключ в authorized_keys не пишется.
        ("-----BEGIN OPENSSH PRIVATE KEY-----", False),
        ("", False),
    ],
)
def test_bootstrap_accepts_any_public_key_type(line: str, ok: bool) -> None:
    script = f'source "{DEPLOY / "bootstrap.sh"}"; is_public_key "$1"'
    result = subprocess.run(["bash", "-c", script, "bash", line], check=False)

    assert (result.returncode == 0) is ok


@pytest.mark.parametrize(
    "path",
    ["deploy/deploy.key", "deploy/deploy.key.pub", "ops", "ops.pub"],
)
def test_deploy_keys_are_ignored_by_git(path: str) -> None:
    """Приватный ключ лежит в папке репозитория: git add -A не должен его взять."""
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False)

    assert result.returncode == 0, f"{path} не в .gitignore"


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def commit(repo: Path, path: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(path)
    git("add", ".", cwd=repo)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", path, cwd=repo)


def update_repo(origin: Path, app_dir: Path) -> subprocess.CompletedProcess[str]:
    script = (
        f'source "{DEPLOY / "bootstrap.sh"}"; '
        f'REPO_URL="{origin}"; APP_DIR="{app_dir}"; update_repo'
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_rerun_updates_the_clone_even_after_a_deploy(tmp_path: Path) -> None:
    """Первый запуск — до push (на GitHub нет deploy/), второй — после."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    commit(origin, "README.md")
    app_dir = tmp_path / "app"

    first = update_repo(origin, app_dir)
    assert first.returncode == 2
    assert "Сначала git push" in first.stderr

    commit(origin, "deploy/deploy.sh")
    # Деплой оставляет клон не на ветке, а на коммите.
    git("checkout", "-q", "--detach", "HEAD", cwd=app_dir)

    assert update_repo(origin, app_dir).returncode == 0
    assert (app_dir / "deploy" / "deploy.sh").exists()


def test_deploy_accepts_only_the_rebuild_flag(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(DEPLOY / "deploy.sh"), "abc123", "--yes-rebuild"],
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--rebuild-index" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_ci_allows_index_rebuild_only_when_variable_is_1() -> None:
    """Полная сборка индекса стоит ≈ 200 ₽: по умолчанию деплой её не делает."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "ALLOW_INDEX_REBUILD: ${{ vars.ALLOW_INDEX_REBUILD }}" in ci
    assert '[ "$ALLOW_INDEX_REBUILD" = "1" ]; then rebuild=" --rebuild-index"' in ci
    deploy = (DEPLOY / "deploy.sh").read_text()
    # Сборка — только в ветке с флагом, после проверки индекса.
    assert deploy.index("index-check") < deploy.index('"--rebuild-index" ]]; then')
    assert deploy.count("index-if-needed YES=1") == 1


def test_fail2ban_bans_ssh_and_the_docker_published_https_ports() -> None:
    """Порты Caddy опубликованы Docker: пакеты к ним идут через DNAT мимо INPUT,
    поэтому правило бана — на prerouting до DNAT (проверено на пакете Ubuntu
    1.1.0-9: nft list table inet f2b-table)."""
    jail = (DEPLOY / "fail2ban" / "recipes-llm.local").read_text()
    deploy = (DEPLOY / "deploy.sh").read_text()

    assert "port = 22,80,443" in jail
    assert 'chain_hook="prerouting", chain_priority="-150", blocktype="drop"' in jail
    assert (
        "maxretry = 5" in jail and "findtime = 10m" in jail and "bantime = 1h" in jail
    )
    assert "/etc/fail2ban/jail.d/recipes-llm.local" in deploy
    # Ставится после checkout: копируется версия jail этого же коммита.
    assert deploy.index('git checkout --quiet --detach "$sha"') < deploy.index(
        "== fail2ban"
    )


def test_https_profile_only_when_both_domains_are_in_env(tmp_path: Path) -> None:
    """Caddy (профиль tls) — только с обоими доменами: иначе Caddyfile без
    адреса сайта. Домены — переменные Actions, деплой пишет их в .env."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "GRAFANA_DOMAIN: ${{ vars.GRAFANA_DOMAIN }}" in ci
    assert "GROCY_DOMAIN: ${{ vars.GROCY_DOMAIN }}" in ci
    deploy = (DEPLOY / "deploy.sh").read_text()
    assert 'profiles="$profiles,tls"' in deploy

    # env_value из deploy.sh: кавычки set_env.py снимаются, нет ключа — пусто.
    function = deploy[deploy.index("env_value() {") : deploy.index("\n}\n") + 2]
    (tmp_path / ".env").write_text("GRAFANA_DOMAIN='g.example.org'\nGROCY_DOMAIN=\n")
    script = f'{function}\necho "[$(env_value GRAFANA_DOMAIN)][$(env_value GROCY_DOMAIN)][$(env_value NOPE)]"'
    result = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[g.example.org][][]"


@pytest.mark.parametrize(
    ("stdin", "message"),
    [
        ("", "две строки"),
        ("203.0.113.7\n/nonexistent/key\n", "ключ не читается"),
    ],
)
def test_upload_index_needs_host_and_readable_key(stdin: str, message: str) -> None:
    result = subprocess.run(
        ["bash", str(DEPLOY / "upload_index.sh")],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_index_snapshot_stays_out_of_git_and_docker_context() -> None:
    assert "index/" in (ROOT / ".gitignore").read_text().splitlines()
    assert "index/" in (ROOT / ".dockerignore").read_text().splitlines()
