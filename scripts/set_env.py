"""Задаёт значения в .env из stdin, по строке `KEY=value`. Для деплоя.

    printf 'GIGACHAT_API_KEY=%s\n' "$key" | python scripts/set_env.py .env

Деплой из GitHub Actions передаёт ключи API через stdin ssh, а не
аргументами: аргументы команды видит любой пользователь сервера в списке
процессов (ps), stdin — нет.

Принимаются ключи из ALLOWED — то, что задаётся в секретах GitHub.
Секреты demo-Paperless генерирует bootstrap на сервере, деплой их не трогает.

Значение пишется в одинарных кавычках: так Docker Compose и uv берут его
буквально, и `$` или `#` в ключе не станут подстановкой или комментарием.
Одинарная кавычка и перевод строки в значении поэтому запрещены.
Только стандартная библиотека: зависимости проекта здесь не нужны.
"""

import sys
from pathlib import Path

from init_env import write_private

ALLOWED = frozenset({"CLOUD_RU_API_KEY", "GIGACHAT_API_KEY"})


class EnvValueError(ValueError):
    pass


def parse_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        if not sep or key not in ALLOWED:
            # Значение в сообщение не попадает: это может быть ключ.
            raise EnvValueError(
                f"разрешены только {', '.join(sorted(ALLOWED))}, а не {key!r}"
            )
        if not value:
            raise EnvValueError(f"{key}: пустое значение")
        if "'" in value or "\r" in value:
            raise EnvValueError(f"{key}: одинарная кавычка и \\r не допускаются")
        values[key] = value
    return values


def set_values(text: str, values: dict[str, str]) -> str:
    """Заменяет строки KEY=… или дописывает новые; остальное не трогает."""
    lines = text.splitlines()
    left = dict(values)
    for index, line in enumerate(lines):
        key = line.partition("=")[0]
        if key in left:
            lines[index] = f"{key}='{left.pop(key)}'"
    lines += [f"{key}='{value}'" for key, value in left.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(
            "использование: set_env.py <путь к .env>  (значения — в stdin)",
            file=sys.stderr,
        )
        return 2
    target = Path(argv[0])
    try:
        values = parse_values(sys.stdin.read())
    except EnvValueError as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return 2
    if not values:
        print("в stdin нет значений, .env не меняется", file=sys.stderr)
        return 0
    write_private(target, set_values(target.read_text(encoding="utf-8"), values))
    print(f"в {target} заданы: {', '.join(values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
