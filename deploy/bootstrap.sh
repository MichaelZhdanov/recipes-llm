#!/usr/bin/env bash
# Первичная настройка сервера: один раз, от root, на Ubuntu или Debian.
#
#   sudo bash bootstrap.sh <домен> <e-mail для Let's Encrypt> <файл публичного ключа деплоя>
#
# До запуска: A-запись домена указывает на сервер, порты 80 и 443 открыты.
# Ключ деплоя создаётся на своей машине: ssh-keygen -t ed25519 -f deploy_key -N ''
# (deploy_key.pub — сюда, deploy_key — в секрет GitHub DEPLOY_SSH_KEY).
#
# Что делает:
# 1. ставит Docker (из официального apt-репозитория Docker), nginx, certbot;
# 2. берёт uv той же версии, что в Dockerfile, из того же образа;
# 3. заводит пользователя deploy и клонирует репозиторий в /opt/paperless-assistant;
# 4. разрешает ключу деплоя только deploy/deploy.sh (forced command);
# 5. создаёт .env со случайными секретами;
# 6. настраивает nginx и получает сертификат Let's Encrypt;
# 7. включает суточный таймер paperless-refresh.
# Ключи API (GigaChat, cloud.ru) скрипт не спрашивает: их вписывают в .env руками.
# Повторный запуск безопасен: готовые шаги пропускаются.
set -euo pipefail

APP_DIR=/opt/paperless-assistant
REPO_URL=https://github.com/MichaelZhdanov/paperless-assistant.git
# Тот же образ, что в Dockerfile и Makefile (tests/test_deploy.py это проверяет).
UV_IMAGE=ghcr.io/astral-sh/uv:0.13.5-python3.14-trixie-slim
SITE=/etc/nginx/sites-available/paperless-assistant

main() {
    if [[ $# -ne 3 ]]; then
        echo "использование: sudo bash $0 <домен> <e-mail> <файл публичного ключа деплоя>" >&2
        exit 2
    fi
    local domain="$1" email="$2" key_file="$3"
    [[ $EUID -eq 0 ]] || { echo "нужен root: sudo bash $0 …" >&2; exit 2; }
    [[ -r "$key_file" ]] || { echo "не читается файл ключа $key_file" >&2; exit 2; }
    local key
    key="$(head -n 1 "$key_file")"
    [[ "$key" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+(\ .*)?$ ]] || {
        echo "в $key_file не ключ ssh-ed25519" >&2
        exit 2
    }
    # shellcheck source=/dev/null
    . /etc/os-release
    [[ "$ID" == ubuntu || "$ID" == debian ]] || { echo "поддерживаются Ubuntu и Debian, а не $ID" >&2; exit 2; }

    echo "== 1. пакеты"
    apt-get update -q
    apt-get install -y -q ca-certificates curl git make nginx certbot python3-certbot-nginx gettext-base
    if ! command -v docker >/dev/null; then
        install -m 0755 -d /etc/apt/keyrings
        curl -fsSL "https://download.docker.com/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
        chmod a+r /etc/apt/keyrings/docker.asc
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID $VERSION_CODENAME stable" \
            >/etc/apt/sources.list.d/docker.list
        apt-get update -q
        apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi

    echo "== 2. uv из образа $UV_IMAGE"
    if ! uv --version 2>/dev/null | grep -q "^uv 0.13.5"; then
        local container
        container="$(docker create "$UV_IMAGE")"
        docker cp "$container:/usr/local/bin/uv" /usr/local/bin/uv
        docker rm "$container" >/dev/null
    fi

    echo "== 3. пользователь deploy и репозиторий"
    id deploy >/dev/null 2>&1 || useradd --create-home --shell /bin/bash deploy
    # Группа docker — это фактически root на сервере. Поэтому ключ деплоя
    # ограничен одной командой (шаг 4), а пароля у deploy нет.
    usermod -aG docker deploy
    if [[ ! -d "$APP_DIR/.git" ]]; then
        install -d -o deploy -g deploy "$APP_DIR"
        sudo -u deploy git clone --quiet "$REPO_URL" "$APP_DIR"
    fi

    echo "== 4. ключ деплоя: только deploy/deploy.sh"
    local ssh_dir=/home/deploy/.ssh
    install -d -m 700 -o deploy -g deploy "$ssh_dir"
    # restrict: без пробросов портов, агента, X11 и терминала.
    echo "restrict,command=\"$APP_DIR/deploy/deploy.sh\" $key" >"$ssh_dir/authorized_keys"
    chown deploy:deploy "$ssh_dir/authorized_keys"
    chmod 600 "$ssh_dir/authorized_keys"

    echo "== 5. .env со случайными секретами"
    sudo -u deploy make -C "$APP_DIR" --no-print-directory .env
    local port
    port="$(grep -s '^ASSISTANT_PORT=' "$APP_DIR/.env" | cut -d= -f2 | tr -d "'\"" || true)"

    echo "== 6. nginx и сертификат для $domain"
    # Одинарные кавычки нарочно: это список имён для envsubst, а не подстановка shell.
    # shellcheck disable=SC2016
    DOMAIN="$domain" ASSISTANT_PORT="${port:-8080}" \
        envsubst '${DOMAIN} ${ASSISTANT_PORT}' <"$APP_DIR/deploy/nginx.conf.template" >"$SITE"
    ln -sf "$SITE" /etc/nginx/sites-enabled/paperless-assistant
    nginx -t
    systemctl reload nginx
    if [[ ! -d "/etc/letsencrypt/live/$domain" ]]; then
        certbot --nginx -d "$domain" -m "$email" --agree-tos --no-eff-email \
            --redirect --hsts --non-interactive
    fi

    echo "== 7. суточный таймер"
    install -m 644 "$APP_DIR/deploy/paperless-refresh.service" "$APP_DIR/deploy/paperless-refresh.timer" \
        /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now paperless-refresh.timer

    cat <<DONE

Готово. Осталось:
1. Вписать ключи API в $APP_DIR/.env (sudo -u deploy nano $APP_DIR/.env):
   CLOUD_RU_API_KEY, GIGACHAT_API_KEY. NO_AUTH должен остаться 0.
2. В GitHub → Settings → Environments → production:
   секреты DEPLOY_HOST=$domain, DEPLOY_SSH_KEY (приватный ключ деплоя),
   WEB_PASSWORD (пароль входа), DEPLOY_KNOWN_HOSTS — строка ниже;
   переменная WEB_LOGIN (логин входа).
   $domain $(cut -d' ' -f1,2 /etc/ssh/ssh_host_ed25519_key.pub)
3. Запустить деплой: push в main или Actions → CI → Run workflow.
   Первый деплой строит индекс (≈27 ₽ на cloud.ru, пара минут).
DONE
}

main "$@"
