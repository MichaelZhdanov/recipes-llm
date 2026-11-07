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
# Шаги: ключи API и модель ассистента в .env → код ровно того коммита, что
# проверил CI → fail2ban → стек (Qdrant, бот, наблюдаемость) → индекс тот? →
# состояние сервисов.
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

# Значение ключа из .env без кавычек (set_env.py пишет одинарные); нет — пусто.
env_value() {
    grep -s "^$1=" .env | tail -n 1 | cut -d= -f2- | tr -d "'\"" || true
}

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

    # Модель на сервере задаёт репозиторий (решение автора 2026-11-07): .env
    # сервера создан, когда по умолчанию был Claude, и сам не обновится.
    # GLM-5.1 low разбирает вопрос и отвечает, проверки перед отправкой нет.
    echo "== модель ассистента → .env"
    printf '%s\n' LLM_PROVIDER=cloudru LLM_MODEL=zai-org/GLM-5.1 LLM_EFFORT=low \
        ANSWER_REVIEW=false | uv run --no-project --python 3.14 scripts/set_env.py .env

    echo "== код коммита $sha"
    git fetch --quiet origin main
    git checkout --quiet --detach "$sha"

    # fail2ban: перебор паролей SSH — тысячи попыток в сутки. Пакет Ubuntu — один
    # раз, наш jail (сроки бана, порты 22, 80, 443) — на каждом деплое.
    echo "== fail2ban"
    if ! command -v fail2ban-client >/dev/null; then
        apt-get -o DPkg::Lock::Timeout=300 update -qq
        DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 \
            install -y -qq fail2ban
    fi
    install -m 644 deploy/fail2ban/recipes-llm.local /etc/fail2ban/jail.d/recipes-llm.local
    systemctl enable --quiet --now fail2ban
    # reload перечитывает jail, но не пересоздаёт уже созданную цепочку nftables:
    # после установки пакета она стоит на input (только 22, reject). Не на
    # prerouting — restart: действие запускается заново, баны fail2ban
    # восстанавливает из своей базы.
    if nft list chain inet f2b-table f2b-chain 2>/dev/null | grep -q "hook prerouting"; then
        fail2ban-client reload >/dev/null
    else
        systemctl restart fail2ban
        sleep 3
    fi
    fail2ban-client status sshd | grep -E "Currently banned|Total banned" || true

    # На сервере — только бот: HTTP API наружу не нужен (решение автора 53).
    # Склад Grocy — отдельный проект в /opt/grocy (решение автора): make up
    # кладёт туда его compose, поднимает и наполняет демо-складом; секреты Grocy
    # генерирует сам. Prometheus, Grafana и Alertmanager → Telegram — на
    # сервере (решения 80–81). Переменные перекрывают значения из .env
    # сервера. API от прошлых деплоев останавливается.
    echo "== стек"
    # HTTPS для Grafana и Grocy (Caddy, решение 82) — только когда в .env оба
    # домена: их пишет set_env.py из переменных Actions.
    local profiles=bot,observability,alerts,host
    if [[ -n "$(env_value GRAFANA_DOMAIN)" && -n "$(env_value GROCY_DOMAIN)" ]]; then
        profiles="$profiles,tls"
    else
        echo "домены GRAFANA_DOMAIN и GROCY_DOMAIN не заданы — Grafana и Grocy только по ssh-туннелю"
    fi
    export COMPOSE_PROFILES="$profiles" GROCY_DIR=/opt/grocy
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
