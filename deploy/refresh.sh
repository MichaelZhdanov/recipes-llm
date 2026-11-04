#!/usr/bin/env bash
# Раз в сутки (systemd: paperless-refresh.timer): исходники Paperless на теге
# PAPERLESS_REF и индекс по ним, если их ещё нет. Тег не меняется, поэтому
# обычно оба шага ничего не делают; они нужны после чистой установки или
# если папку data/ или том Qdrant удалили. Новые релизы Paperless не
# отслеживаются: тег меняют осознанно, вместе с проверкой на eval.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
# Тот же замок, что у deploy.sh: не строить индекс одновременно с деплоем.
exec 9>"$HOME/.paperless-deploy.lock"
flock -w 1800 9

export UV_NO_DEV=1
make --no-print-directory sources
make --no-print-directory index-if-needed
