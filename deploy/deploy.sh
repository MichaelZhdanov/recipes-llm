#!/usr/bin/env bash
# Деплой на сервере: deploy/deploy.sh <коммит>. Запускается от root.
#
# GitHub Actions сначала переводит клон на коммит, потом вызывает скрипт:
#   printf 'GIGACHAT_API_KEY=%s\n' … | ssh root@<сервер> \
#     "cd /opt/recipes-llm && git fetch origin main && git checkout --detach <sha> && deploy/deploy.sh <sha>"
# Вручную на сервере достаточно deploy/deploy.sh <sha>: checkout ниже повторится.
# Ключи API — в stdin, а не аргументами: аргументы видны в списке
# процессов (ps), stdin — нет.
#
# Шаги: ключи API в .env → код ровно того коммита, что проверил CI →
# стек → проверка /health. Сборка индекса вернётся на этапе 1.
#
# Весь код — в функции main: bash читает скрипт по ходу выполнения, а git
# checkout ниже может заменить этот самый файл. Функцию bash разбирает целиком
# до запуска, поэтому новая версия скрипта подействует со следующего деплоя.
set -euo pipefail

main() {
    if [[ $# -ne 1 ]]; then
        echo "использование: $0 <коммит>  (ключи API — в stdin)" >&2
        exit 2
    fi
    local sha="$1"

    cd "$(dirname "$(readlink -f "$0")")/.."
    # Два деплоя (из CI и ручной) не идут одновременно: оба могут строить индекс.
    exec 9>"$HOME/.recipes-llm-deploy.lock"
    flock -w 1800 9

    # На сервере не нужны линтеры и тесты: uv ставит только зависимости сервиса.
    export UV_NO_DEV=1

    echo "== ключи API из секретов GitHub → .env"
    uv run --no-project --python 3.14 scripts/set_env.py .env

    echo "== код коммита $sha"
    git fetch --quiet origin main
    git checkout --quiet --detach "$sha"

    echo "== стек"
    make --no-print-directory up

    local port
    port="$(grep -s '^ASSISTANT_PORT=' .env | cut -d= -f2 | tr -d "'\"" || true)"
    echo "== проверка"
    curl -fsS "http://127.0.0.1:${port:-8080}/health"
    echo
    echo "деплой $sha готов"
}

main "$@"
exit
