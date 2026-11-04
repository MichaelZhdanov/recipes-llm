"""Вход в веб-чат и API по логину и паролю.

Ассистент открыт в интернет, а каждый вопрос тратит токены GigaChat. Поэтому
на сервере чат и `/ask` доступны только после входа. Устроено просто:
- логин и пароль приходят из окружения (`WEB_LOGIN`, `WEB_PASSWORD`): на
  сервере их кладёт в `.env` деплой из секретов GitHub, локально генерирует
  `make up`;
- после входа сервер ставит cookie сессии. Её подписывает готовый
  `SessionMiddleware` из Starlette (библиотека itsdangerous) ключом
  `WEB_SESSION_SECRET`: без ключа cookie не подделать и не изменить;
- в сессии лежит отпечаток текущих логина и пароля. Смена пароля в CI
  выкидывает всех, кто вошёл со старым;
- без сессии страница перенаправляет на `/login`, API отвечает 401.
  Открыты только `/health` (для мониторинга) и сама страница входа;
- `NO_AUTH=1` выключает вход, это для локальной разработки.

Чего здесь нет и почему:
- пользователей и ролей: для демо один общий вход, роли по Telegram id — этап 4;
- хранения паролей: пароль один, он приходит из окружения и сравнивается
  `secrets.compare_digest`, время сравнения не выдаёт, сколько символов совпало;
- защиты от перебора: её делает nginx (`limit_req` на `/login`).
"""

import hashlib
import hmac
import secrets

from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from assistant.config import Settings

SESSION_COOKIE = "pa_session"
SESSION_MAX_AGE = 7 * 24 * 3600
# Короче — подпись cookie можно подобрать.
MIN_SECRET_CHARS = 32
# Что открыто без входа: проверка жизни и всё, что нужно странице входа.
PUBLIC_PATHS = frozenset({"/health", "/login", "/login.js", "/chat.css"})


class AuthConfigError(RuntimeError):
    """Вход включён, но не настроен: лучше не стартовать, чем стартовать открытым."""


class LoginRequest(BaseModel):
    login: str = Field(max_length=200)
    password: str = Field(max_length=200)


def check_config(settings: Settings) -> None:
    if not settings.web_password.get_secret_value():
        raise AuthConfigError(
            "вход включён, но WEB_PASSWORD пустой: задайте его в .env "
            "(make up сгенерирует) или выключите вход NO_AUTH=1"
        )
    if len(settings.web_session_secret.get_secret_value()) < MIN_SECRET_CHARS:
        raise AuthConfigError(
            f"WEB_SESSION_SECRET короче {MIN_SECRET_CHARS} символов: "
            "make up сгенерирует его в .env"
        )


def credentials_match(settings: Settings, login: str, password: str) -> bool:
    """Оба сравнения выполняются всегда: по времени ответа не понять, что не так."""
    login_ok = secrets.compare_digest(login.encode(), settings.web_login.encode())
    password_ok = secrets.compare_digest(
        password.encode(), settings.web_password.get_secret_value().encode()
    )
    return login_ok and password_ok


def fingerprint(settings: Settings) -> str:
    """Отпечаток текущих логина и пароля. По нему сессия узнаёт, что пароль сменили.

    HMAC с ключом сессии, а не хеш пароля: cookie подписана, но не зашифрована,
    и из отпечатка без ключа пароль не подобрать.
    """
    key = settings.web_session_secret.get_secret_value().encode()
    message = f"{settings.web_login}\0{settings.web_password.get_secret_value()}"
    return hmac.new(key, message.encode(), hashlib.sha256).hexdigest()


def is_logged_in(session: dict[str, object], settings: Settings) -> bool:
    value = session.get("auth")
    return isinstance(value, str) and secrets.compare_digest(
        value, fingerprint(settings)
    )


def _wants_html(scope: Scope) -> bool:
    for name, value in scope.get("headers", []):
        if name == b"accept":
            return b"text/html" in value
    return False


class LoginRequired:
    """ASGI middleware: без входа — редирект на /login для страниц и 401 для API.

    Стоит внутри SessionMiddleware: к этому моменту сессия уже прочитана из cookie.
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"] in PUBLIC_PATHS
            or is_logged_in(scope["session"], self.settings)
        ):
            await self.app(scope, receive, send)
            return
        response: Response
        if scope["method"] == "GET" and _wants_html(scope):
            response = RedirectResponse("/login", status_code=303)
        else:
            response = JSONResponse({"detail": "нужен вход"}, status_code=401)
        await response(scope, receive, send)
