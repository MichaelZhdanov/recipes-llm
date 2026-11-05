"""Создаёт .env из .env.example и подставляет случайные секреты.

Вызывается из `make up` в Docker-образе uv с Python 3.14, где зависимости
проекта ещё не установлены, поэтому только стандартная библиотека.

Если .env уже есть, в нём меняется одно: дописываются сгенерированные секреты,
которых там нет или они пустые. Так обновление проекта, добавившее новый
секрет, не ломает запуск со старым .env. Заданные
значения не трогаются никогда.
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


def _secret(key: str) -> str:
    return secrets.token_urlsafe(GENERATED_SECRETS[key])


def render_env(template: str) -> str:
    """Текст .env: шаблон, в котором пустые GENERATED_SECRETS заполнены случайными значениями."""
    lines = []
    for line in template.splitlines():
        key, sep, value = line.partition("=")
        if sep and not value and key in GENERATED_SECRETS:
            line = f"{key}={_secret(key)}"
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


def add_missing_secrets(text: str) -> tuple[str, list[str]]:
    """Существующий .env с заполненными недостающими секретами и их список.

    Пустой секрет заполняется на месте, отсутствующий дописывается в конец.
    """
    lines = text.splitlines()
    present: set[str] = set()
    added: list[str] = []
    for index, line in enumerate(lines):
        key, sep, value = line.partition("=")
        if sep and key in GENERATED_SECRETS:
            present.add(key)
            if not value:
                lines[index] = f"{key}={_secret(key)}"
                added.append(key)
    for key in GENERATED_SECRETS:
        if key not in present:
            lines.append(f"{key}={_secret(key)}")
            added.append(key)
    return "\n".join(lines) + "\n", added


def write_private(target: Path, content: str) -> None:
    """Перезаписывает файл с секретами: права 0600, атомарно."""
    temporary = target.with_name(f"{target.name}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as env_file:
        env_file.write(content)
    # replace атомарен: .env либо старый, либо новый целиком.
    os.replace(temporary, target)


def update_env(target: Path) -> list[str]:
    """Дописывает в target недостающие секреты. Права 0600 сохраняются."""
    content, added = add_missing_secrets(target.read_text(encoding="utf-8"))
    if added:
        write_private(target, content)
    return added


def main() -> int:
    target = ROOT / ".env"
    if create_env(ROOT / ".env.example", target):
        print(
            "Создан .env, сгенерированы секреты: " + ", ".join(GENERATED_SECRETS) + "."
        )
        print(
            "Вход в Paperless: http://localhost:8000, "
            "логин и пароль — PAPERLESS_ADMIN_USER и PAPERLESS_ADMIN_PASSWORD в .env."
        )
        return 0
    added = update_env(target)
    if added:
        print(f"В .env добавлены новые секреты: {', '.join(added)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
