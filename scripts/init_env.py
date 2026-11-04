"""Создаёт .env из .env.example и подставляет случайные секреты для demo-Paperless.

Вызывается из `make up` (цель `.env`) в Docker-образе uv с Python 3.14, где
зависимости проекта ещё не установлены, поэтому только стандартная библиотека.
Существующий .env не трогает.
"""

import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Какие пустые переменные заполнять и сколько байт случайности брать.
# token_urlsafe даёт только [A-Za-z0-9_-]: без `$`, кавычек и пробелов,
# поэтому значение не нужно экранировать ни для compose, ни для shell.
GENERATED_SECRETS = {
    "PAPERLESS_SECRET_KEY": 48,
    "PAPERLESS_ADMIN_PASSWORD": 16,
}


def render_env(template: str) -> str:
    """Текст .env: шаблон, в котором пустые GENERATED_SECRETS заполнены случайными значениями."""
    lines = []
    for line in template.splitlines():
        key, sep, value = line.partition("=")
        if sep and not value and key in GENERATED_SECRETS:
            line = f"{key}={secrets.token_urlsafe(GENERATED_SECRETS[key])}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def create_env(example: Path, target: Path) -> bool:
    """Создаёт target из example. Возвращает False, если target уже существует."""
    if target.exists():
        return False
    content = render_env(example.read_text(encoding="utf-8"))
    # O_EXCL: не перезаписать файл, появившийся между проверкой и записью.
    # 0o600: файл с секретами читает и пишет только владелец.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as env_file:
        env_file.write(content)
    return True


def main() -> int:
    if not create_env(ROOT / ".env.example", ROOT / ".env"):
        print(".env уже существует, оставляю как есть.")
        return 0
    print("Создан .env, сгенерированы PAPERLESS_SECRET_KEY и PAPERLESS_ADMIN_PASSWORD.")
    print(
        "Вход в Paperless: http://localhost:8000, "
        "логин и пароль — PAPERLESS_ADMIN_USER и PAPERLESS_ADMIN_PASSWORD в .env."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
