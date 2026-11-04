import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from test_api import FakeAnswerer

from assistant.api import create_app
from assistant.config import Settings
from assistant.security.auth import SESSION_COOKIE, AuthConfigError

SECRET = "s" * 40
AUTH = Settings(
    web_login="admin",
    web_password=SecretStr("correct horse"),
    web_session_secret=SecretStr(SECRET),
)
HTML = {"Accept": "text/html,application/xhtml+xml"}


def client(settings: Settings = AUTH) -> TestClient:
    # https: cookie сессии с флагом Secure по http клиент не вернёт, как и браузер.
    return TestClient(
        create_app(settings, answerer=FakeAnswerer()), base_url="https://testserver"
    )


def logged_in(settings: Settings = AUTH) -> TestClient:
    c = client(settings)
    response = c.post("/login", json={"login": "admin", "password": "correct horse"})
    assert response.status_code == 200
    return c


def test_without_login_pages_go_to_login_and_api_is_401() -> None:
    c = client()

    page = c.get("/", headers=HTML, follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/login"
    assert c.get("/docs", headers=HTML, follow_redirects=False).status_code == 303
    assert c.post("/ask", json={"question": "OCR"}).status_code == 401
    assert c.get("/openapi.json").status_code == 401
    assert c.get("/chat.js").status_code == 401


def test_health_and_login_page_are_open() -> None:
    c = client()

    assert c.get("/health").status_code == 200
    assert 'id="login"' in c.get("/login").text
    assert c.get("/login.js").status_code == 200
    assert c.get("/chat.css").status_code == 200


@pytest.mark.parametrize(
    ("login", "password"),
    [("admin", "wrong"), ("root", "correct horse"), ("", "")],
)
def test_wrong_credentials_are_401_without_session(login: str, password: str) -> None:
    c = client()

    response = c.post("/login", json={"login": login, "password": password})

    assert response.status_code == 401
    assert SESSION_COOKIE not in response.cookies
    assert c.post("/ask", json={"question": "OCR"}).status_code == 401


def test_login_opens_chat_and_api_with_a_strict_cookie() -> None:
    c = client()

    response = c.post("/login", json={"login": "admin", "password": "correct horse"})

    assert response.json() == {"user": "admin"}
    cookie = response.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    assert c.get("/", headers=HTML).status_code == 200
    assert c.post("/ask", json={"question": "OCR"}).status_code == 200
    assert c.get("/session").json() == {"user": "admin"}


def test_logout_closes_the_session() -> None:
    c = logged_in()

    assert c.post("/logout").json() == {"user": None}

    assert c.post("/ask", json={"question": "OCR"}).status_code == 401


def test_password_change_logs_everyone_out() -> None:
    """Пароль сменили в CI: старая cookie подписана верно, но отпечаток уже другой."""
    cookie = logged_in().cookies[SESSION_COOKIE]
    rotated = client(
        Settings(
            web_password=SecretStr("new password"),
            web_session_secret=SecretStr(SECRET),
        )
    )

    rotated.cookies.set(SESSION_COOKIE, cookie)

    assert rotated.post("/ask", json={"question": "OCR"}).status_code == 401


def test_cookie_signed_with_another_secret_is_rejected() -> None:
    cookie = logged_in().cookies[SESSION_COOKIE]
    other = client(
        Settings(
            web_password=SecretStr("correct horse"),
            web_session_secret=SecretStr("x" * 40),
        )
    )

    other.cookies.set(SESSION_COOKIE, cookie)

    assert other.post("/ask", json={"question": "OCR"}).status_code == 401


@pytest.mark.parametrize(
    "settings",
    [
        Settings(web_session_secret=SecretStr(SECRET)),
        Settings(web_password=SecretStr("pw"), web_session_secret=SecretStr("short")),
    ],
)
def test_login_without_password_or_secret_refuses_to_start(settings: Settings) -> None:
    """Лучше не стартовать, чем стартовать открытым."""
    with pytest.raises(AuthConfigError):
        create_app(settings, answerer=FakeAnswerer())


def test_no_auth_opens_everything() -> None:
    c = client(Settings(no_auth=True))

    assert c.get("/", headers=HTML).status_code == 200
    assert c.post("/ask", json={"question": "OCR"}).status_code == 200
    assert c.get("/session").json() == {"user": None}
    assert c.get("/login").status_code == 404
