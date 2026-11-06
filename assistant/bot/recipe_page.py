"""Шаги со страницы рецепта для кнопки «📖 Шаги с фото» (решение автора 2026-11-06).

В датасете шагов нет, на странице povarenok.ru они есть — с фото. Вариант
«б»: бот присылает шаги в чат — альбомом фото с текстом шага в подписи —
с пометкой «источник: povarenok.ru» и ссылкой. Права на шаги и фото — у
авторов рецептов: это MVP для автора; перед показом другим — прочитать
пользовательское соглашение povarenok или вернуть вариант «в» (фото блюда и
число шагов, сами шаги по ссылке).

Страница размечена по schema.org: главное фото — первое `itemprop="image"`,
шаги — пункты `li.cooking-bl` внутри `<ul itemprop="recipeInstructions">`;
у шага может быть фото: крупное — в ссылке `a href` (640×480), миниатюра —
в `img src`. Разбор — парсером, без модели: текст и фото берутся как есть.
Фото бот скачивает сам и отдаёт Telegram файлом: по ссылке Telegram
забирает картинки povarenok через раз («failed to get HTTP URL content»).
"""

from dataclasses import dataclass, field
from html.parser import HTMLParser

import httpx2

from assistant import __version__

TIMEOUT = 5.0
# Страница рецепта — около 400 КБ; больше — не страница рецепта.
MAX_BYTES = 2_000_000
# Фото берём только с сайта рецептов: адрес берётся из чужого HTML.
PHOTO_PREFIX = "https://www.povarenok.ru/"
# Telegram принимает фото до 10 МБ; фото шага на povarenok — около 100 КБ.
MAX_PHOTO_BYTES = 5_000_000


@dataclass(frozen=True)
class Step:
    text: str
    photo: str | None


@dataclass(frozen=True)
class RecipePage:
    photo: str | None
    steps: list[Step] = field(default_factory=list)


def _photo(url: str | None) -> str | None:
    return url if url and url.startswith(PHOTO_PREFIX) else None


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.photo: str | None = None
        self.steps: list[Step] = []
        self._in_instructions = False
        self._in_step = False
        self._text: list[str] = []
        self._large: str | None = None
        self._small: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "ul" and attr.get("itemprop") == "recipeInstructions":
            self._in_instructions = True
        elif (
            self._in_instructions
            and tag == "li"
            and "cooking-bl" in (attr.get("class") or "").split()
        ):
            self._close_step()
            self._in_step = True
        elif self._in_step and tag == "a" and (attr.get("href") or "").endswith(".jpg"):
            self._large = self._large or _photo(attr.get("href"))
        elif tag == "img" and attr.get("itemprop") == "image":
            if self._in_step:
                self._small = self._small or _photo(attr.get("src"))
            elif self.photo is None and not self._in_instructions:
                self.photo = _photo(attr.get("src"))

    def handle_data(self, data: str) -> None:
        if self._in_step:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "ul" and self._in_instructions:
            self._close_step()
            self._in_instructions = False

    def _close_step(self) -> None:
        if not self._in_step:
            return
        text = " ".join(" ".join(self._text).split())
        photo = self._large or self._small
        if text or photo:
            self.steps.append(Step(text, photo))
        self._in_step = False
        self._text, self._large, self._small = [], None, None


def parse_page(html: str) -> RecipePage:
    parser = _PageParser()
    parser.feed(html)
    parser.close()
    return RecipePage(parser.photo, parser.steps)


def _get(client: httpx2.Client, url: str) -> httpx2.Response | None:
    try:
        return client.get(
            url,
            follow_redirects=False,
            timeout=TIMEOUT,
            headers={"User-Agent": f"recipes-llm/{__version__}"},
        )
    except httpx2.HTTPError:
        return None


def fetch_page(client: httpx2.Client, url: str) -> RecipePage | None:
    """Шаги страницы; None — страница не ответила 200 или слишком велика."""
    response = _get(client, url)
    if response is None or response.status_code != 200:
        return None
    if len(response.content) > MAX_BYTES:
        return None
    # Кодировка — из заголовка (windows-1251 у povarenok); httpx2 её учитывает.
    return parse_page(response.text)


def fetch_photo(client: httpx2.Client, url: str) -> bytes | None:
    """Картинка; None — не с сайта рецептов, не 200, не картинка или слишком велика."""
    if not url.startswith(PHOTO_PREFIX):
        return None
    response = _get(client, url)
    if response is None or response.status_code != 200:
        return None
    if not response.headers.get("Content-Type", "").startswith("image/"):
        return None
    if len(response.content) > MAX_PHOTO_BYTES:
        return None
    return response.content
