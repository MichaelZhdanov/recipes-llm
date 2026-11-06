#!/usr/bin/env bash
# Деплой на сервере: deploy/deploy.sh <коммит> [--rebuild-index]. Запускается от root.
#
# GitHub Actions сначала переводит клон на коммит, потом вызывает скрипт:
#   printf 'GIGACHAT_API_KEY=%s\n' … | ssh root@<сервер> \
#     "cd /opt/recipes-llm && git fetch origin main && git checkout --detach <sha> && deploy/deploy.sh <sha>"
# Вручную на сервере достаточно deploy/deploy.sh <sha>: checkout ниже повторится.
# Ключи API — в stdin, а не аргументами: аргументы видны в списке
# процессов (ps), stdin — нет.
#
# Шаги: ключи API в .env → код ровно того коммита, что проверил CI →
# стек (Qdrant и бот) → индекс тот? → состояние сервисов.
#
# Индекс сам не строится: полная сборка — ≈ 200 ₽ эмбеддингов и около часа, а
# снимок уже посчитан локально и загружается deploy/upload_index.sh. Индекса
# нет или он не тот — деплой падает с подсказкой. Строит только с флагом
# --rebuild-index: его передаёт CI, если переменная Actions ALLOW_INDEX_REBUILD
# равна 1 (по умолчанию 0).
#
# Весь код — в функции main: bash читает скрипт по ходу выполнения, а git
# checkout ниже может заменить этот самый файл. Функцию bash разбирает целиком
# до запуска, поэтому новая версия скрипта подействует со следующего деплоя.
set -euo pipefail

main() {
    if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--rebuild-index" ) ]]; then
        echo "использование: $0 <коммит> [--rebuild-index]  (ключи API — в stdin)" >&2
        exit 2
    fi
    local sha="$1" rebuild="${2:-}"

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

    # На сервере — только бот: HTTP API наружу не нужен (решение автора).
    # Переменная перекрывает COMPOSE_PROFILES из .env сервера. API от прошлых
    # деплоев останавливается.
    echo "== стек"
    export COMPOSE_PROFILES=bot
    docker compose rm --stop --force assistant
    make --no-print-directory up

    # Индекс тот, если совпадают ревизия, модель и размер выборки (INDEX_SAMPLE
    # в .env сервера; пусто — весь датасет).
    echo "== индекс"
    if make --no-print-directory index-check; then
        :
    elif [[ "$rebuild" == "--rebuild-index" ]]; then
        # YES=1: сборку дороже 50 ₽ явно разрешили переменной в GitHub.
        echo "== пересборка индекса разрешена (ALLOW_INDEX_REBUILD)"
        make --no-print-directory data
        make --no-print-directory index-if-needed YES=1
    else
        echo "Индекса на сервере нет или он не тот, а пересборка не разрешена." >&2
        echo "  Загрузить снимок с машины, где лежит index/: deploy/upload_index.sh" >&2
        echo "  (спросит IP сервера и путь к ключу ssh), потом снова деплой." >&2
        echo "  Или пересобрать (≈ 200 ₽, около часа): переменная ALLOW_INDEX_REBUILD=1" >&2
        echo "  в GitHub (Settings → Environments → prod → Variables) и Run workflow." >&2
        exit 1
    fi

    # make up уже дождался healthy (--wait): здесь — только что запущено.
    echo "== проверка"
    docker compose ps --format '{{.Service}}: {{.Status}}'
    echo "деплой $sha готов"
}

main "$@"
exit
