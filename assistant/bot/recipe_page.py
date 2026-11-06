"""Карточка страницы рецепта для кнопки «📖 Шаги с фото» (решение автора 2026-11-06).

В датасете шагов нет, на странице povarenok.ru они есть — с фото. Бот не
перепечатывает чужие шаги и фото целиком (их права у авторов рецептов):
вариант «в» — в чате главное фото блюда и сколько на странице шагов и
фото, сами шаги — по ссылке. MVP для автора; перед показом другим —
прочитать пользовательское соглашение povarenok.

Страница размечена по schema.org: главное фото — первое `itemprop="image"`,
шаги — пункты `li.cooking-bl` внутри `<ul itemprop="recipeInstructions">`,
у шага может быть своё фото. Разбор — парсером, без модели: текст и фото
берутся как есть, ничего не пересказывается. Разметка не нашлась —
карточки нет, остаётся ссылка.
"""

from dataclasses import dataclass
from html.parser import HTMLParser

import httpx2

from assistant import __version__

TIMEOUT = 5.0
# Страница рецепта — около 400 КБ; больше — не страница рецепта.
MAX_BYTES = 2_000_000
# Фото отдаём в Telegram только с сайта рецептов: адрес берётся из чужого HTML.
PHOTO_PREFIX = "https://www.povarenok.ru/"


@dataclass(frozen=True)
class RecipeCard:
    photo: str | None
    steps: int
    step_photos: int


class _CardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.photo: str | None = None
        self.steps = 0
        self.step_photos = 0
        self._in_instructions = False
        self._step_has_photo = False

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
            self.steps += 1
            self._step_has_photo = False
        elif tag == "img" and attr.get("itemprop") == "image":
            src = attr.get("src") or ""
            if self._in_instructions and self.steps:
                self._step_has_photo = True
            elif self.photo is None and src.startswith(PHOTO_PREFIX):
                self.photo = src

    def handle_endtag(self, tag: str) -> None:
        if tag == "ul" and self._in_instructions:
            self._close_step()
            self._in_instructions = False

    def _close_step(self) -> None:
        if self._step_has_photo:
            self.step_photos += 1
            self._step_has_photo = False


def parse_card(html: str) -> RecipeCard:
    parser = _CardParser()
    parser.feed(html)
    parser.close()
    return RecipeCard(parser.photo, parser.steps, parser.step_photos)


def fetch_card(client: httpx2.Client, url: str) -> RecipeCard | None:
    """Карточка страницы; None — страница не ответила 200 или слишком велика."""
    try:
        response = client.get(
            url,
            follow_redirects=False,
            timeout=TIMEOUT,
            headers={"User-Agent": f"recipes-llm/{__version__}"},
        )
    except httpx2.HTTPError:
        return None
    if response.status_code != 200 or len(response.content) > MAX_BYTES:
        return None
    # Кодировка — из заголовка (windows-1251 у povarenok); httpx2 её учитывает.
    return parse_card(response.text)
