"""Жива ли страница рецепта на povarenok.ru — для ссылки «Способ приготовления».

Шагов приготовления в датасете нет, они — на странице рецепта. Перед тем как
поставить ссылку, бот проверяет страницу (решение автора 2026-11-06):
- запрос без редиректов, таймаут 3 с, тело страницы не читается;
- 200 — ссылка; другое — «страница недоступна» и кнопка шагов от ИИ;
- на неделю кэшируются только определённые ответы сайта (200, 3xx, 404,
  410). Сбой сети, 5xx, 403 и 429 — «недоступна» только для этого
  сообщения: иначе минутная авария сайта спрятала бы ссылки на неделю.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx2

from assistant import __version__
from assistant.bot.store import Store

PAGE_TTL = 7 * 24 * 60 * 60
TIMEOUT = 3.0
_DEFINITE = frozenset({200, 404, 410})


def _definite(status: int) -> bool:
    return status in _DEFINITE or 300 <= status < 400


class PageChecker:
    def __init__(self, client: httpx2.Client, store: Store) -> None:
        self._client = client
        self._store = store

    def alive(self, url: str) -> bool:
        cached = self._store.page_ok(url, PAGE_TTL)
        if cached is not None:
            return cached
        try:
            with self._client.stream(
                "GET",
                url,
                follow_redirects=False,
                timeout=TIMEOUT,
                headers={"User-Agent": f"recipes-llm/{__version__}"},
            ) as response:
                status = response.status_code
        except httpx2.HTTPError:
            return False
        if _definite(status):
            self._store.save_page(url, status == 200)
        return status == 200

    def alive_many(self, urls: Sequence[str]) -> dict[str, bool]:
        """Все страницы ответа параллельно: в худшем случае +3 с, а не +15."""
        unique = list(dict.fromkeys(urls))
        if not unique:
            return {}
        with ThreadPoolExecutor(max_workers=len(unique)) as pool:
            return dict(zip(unique, pool.map(self.alive, unique), strict=True))
