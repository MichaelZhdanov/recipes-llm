#!/usr/bin/env bash
# Загрузка полного индекса на сервер снимком Qdrant. Запускается локально, из
# клона репозитория; IP сервера и путь к ключу ssh — в stdin, по строке:
#   printf '%s\n%s\n' 203.0.113.7 deploy/deploy.key | deploy/upload_index.sh
# Без stdin скрипт спросит их сам.
#
# Зачем: сборка полного индекса на сервере — ≈ 200 ₽ эмбеддингов и около
# часа, а он уже посчитан локально. Деплой без ALLOW_INDEX_REBUILD индекс не
# строит, поэтому на новый сервер индекс загружают этим скриптом, а потом
# запускают деплой.
#
# Снимок — единственный index/<коллекция>.snapshot (каталог в .gitignore):
#   curl -X POST localhost:6333/collections/<коллекция>/snapshots — снять,
#   затем скачать файл из /collections/<коллекция>/snapshots/<имя>.
# Паспорт индекса (ревизия, модель, выборка) едет внутри снимка, поэтому
# деплой после загрузки видит, что индекс тот.
#
# Шаги: Qdrant на сервере поднят → алиас recipes уже на этой коллекции —
# выход → снимок по scp → коллекция восстанавливается из снимка → алиас
# переключается одним запросом (поиск не видит момента без индекса) →
# прежняя коллекция алиаса и файл снимка на сервере удаляются.
set -euo pipefail

APP_DIR=/opt/recipes-llm
QDRANT=http://127.0.0.1:6333
ALIAS=recipes

main() {
    local host="" key=""
    [[ -t 0 ]] && printf 'IP сервера: ' >&2
    read -r host || true
    [[ -t 0 ]] && printf 'Путь к ключу ssh: ' >&2
    read -r key || true
    if [[ -z "$host" || -z "$key" ]]; then
        echo "в stdin нужны две строки: IP сервера и путь к ключу ssh" >&2
        exit 2
    fi
    if [[ ! -r "$key" ]]; then
        echo "ключ не читается: $key" >&2
        exit 2
    fi

    cd "$(dirname "$0")/.."
    shopt -s nullglob
    local snapshots=(index/*.snapshot)
    if [[ ${#snapshots[@]} -ne 1 ]]; then
        echo "в index/ нужен ровно один *.snapshot, найдено: ${#snapshots[@]}" >&2
        exit 2
    fi
    local snapshot="${snapshots[0]}" collection
    collection="$(basename "$snapshot" .snapshot)"

    # Ключ сервера не проверяется, как и в деплое (решение автора: сервер
    # может переезжать). -n у ssh: не читать stdin скрипта.
    local ssh_opts=(-i "$key" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30)
    remote() {
        # Команда собирается здесь и выполняется на сервере: подстановки — наши.
        # shellcheck disable=SC2029
        ssh -n "${ssh_opts[@]}" "root@$host" "$1"
    }

    echo "== Qdrant на сервере"
    remote "cd $APP_DIR && docker compose up -d --wait qdrant"

    local current
    current="$(remote "curl -fsS $QDRANT/aliases" |
        sed -n "s/.*\"alias_name\":\"$ALIAS\",\"collection_name\":\"\([^\"]*\)\".*/\1/p")"
    if [[ "$current" == "$collection" ]]; then
        echo "алиас $ALIAS уже на $collection — загружать нечего"
        exit 0
    fi

    echo "== снимок $snapshot ($(du -h "$snapshot" | cut -f1)) → сервер"
    remote "mkdir -p $APP_DIR/index"
    scp "${ssh_opts[@]}" "$snapshot" "root@$host:$APP_DIR/index/"

    echo "== восстановление коллекции $collection"
    remote "curl -fsS -X POST '$QDRANT/collections/$collection/snapshots/upload?priority=snapshot' \
        -F snapshot=@$APP_DIR/$snapshot"
    echo

    echo "== алиас $ALIAS → $collection${current:+ (было $current)}"
    local actions="{\"create_alias\":{\"collection_name\":\"$collection\",\"alias_name\":\"$ALIAS\"}}"
    if [[ -n "$current" ]]; then
        actions="{\"delete_alias\":{\"alias_name\":\"$ALIAS\"}},$actions"
    fi
    remote "curl -fsS -X POST $QDRANT/collections/aliases -H 'Content-Type: application/json' \
        -d '{\"actions\":[$actions]}'"
    echo

    if [[ -n "$current" ]]; then
        # Прежний индекс больше не нужен, а память на сервере — 2 ГБ.
        echo "== удаляю прежнюю коллекцию $current"
        remote "curl -fsS -X DELETE $QDRANT/collections/$current"
        echo
    fi
    remote "rm -f $APP_DIR/$snapshot"

    echo "== готово"
    remote "curl -fsS $QDRANT/collections/$collection" | grep -o '"points_count":[0-9]*' || true
    echo "дальше — деплой: push в main или Run workflow"
}

main "$@"
