#!/usr/bin/env bash
# Деплой на сервере. Это forced command ключа деплоя в authorized_keys
# пользователя deploy: по этому ключу нельзя выполнить ничего другого.
#
# GitHub Actions вызывает его так:
#   printf 'WEB_LOGIN=%s\nWEB_PASSWORD=%s\n' … | ssh deploy@<сервер> "deploy <sha>"
# Команда приходит в SSH_ORIGINAL_COMMAND, логин и пароль — в stdin:
# аргументы видны в списке процессов, stdin — нет.
#
# Шаги: логин и пароль в .env → код ровно того коммита, что проверил CI →
# стек → исходники Paperless и индекс, если их ещё нет → проверка /health.
#
# Весь код — в функции main: bash читает скрипт по ходу выполнения, а git
# checkout ниже может заменить этот самый файл. Функцию bash разбирает целиком
# до запуска, поэтому новая версия скрипта подействует со следующего деплоя.
set -euo pipefail

main() {
    local command="" sha="" extra=""
    read -r command sha extra <<<"${SSH_ORIGINAL_COMMAND:-}" || true
    if [[ "$command" != "deploy" || ! "$sha" =~ ^[0-9a-f]{40}$ || -n "$extra" ]]; then
        echo "разрешена только команда: deploy <полный SHA коммита>" >&2
        exit 2
    fi

    cd "$(dirname "$(readlink -f "$0")")/.."
    # Деплой и суточный таймер не идут одновременно: оба могут строить индекс.
    exec 9>"$HOME/.paperless-deploy.lock"
    flock -w 1800 9

    # На сервере не нужны линтеры и тесты: uv ставит только зависимости сервиса.
    export UV_NO_DEV=1

    echo "== логин и пароль входа из секретов GitHub → .env"
    uv run --no-project --python 3.14 scripts/set_env.py .env

    echo "== код коммита $sha"
    git fetch --quiet origin main
    git checkout --quiet --detach "$sha"

    echo "== стек"
    make --no-print-directory up

    echo "== исходники Paperless и индекс (если их ещё нет)"
    make --no-print-directory sources
    make --no-print-directory index-if-needed

    local port
    port="$(grep -s '^ASSISTANT_PORT=' .env | cut -d= -f2 | tr -d "'\"" || true)"
    echo "== проверка"
    curl -fsS "http://127.0.0.1:${port:-8080}/health"
    echo
    echo "деплой $sha готов"
}

main "$@"
exit
