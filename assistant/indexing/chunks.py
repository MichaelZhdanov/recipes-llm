"""Кусок индекса и ссылки на источник.

Каждый кусок знает, откуда он: путь, строки, тег. Ссылка `url` ведёт на форк
на этом теге с номерами строк, поэтому она всегда указывает ровно на тот текст,
который попал в индекс. Для документации есть ещё `site_url` на
docs.paperless-ngx.com, но сайт показывает последнюю версию, а не тег.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkKind = Literal["doc", "code"]

# site_url из zensical.toml на v3.4.2.
DOCS_SITE_URL = "https://docs.paperless-ngx.com/"


class Chunk(BaseModel):
    """Кусок документации или кода. Все поля обязательны (приёмка F1)."""

    model_config = ConfigDict(frozen=True)

    kind: ChunkKind
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    title: str = Field(min_length=1)
    # Путь заголовков для документации («Configuration > OCR settings > …»)
    # или «файл > класс > метод» для кода.
    breadcrumb: str = Field(min_length=1)
    # Чем кусок является как источник. Части одного длинного раздела делят source_id.
    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    site_url: str | None
    ref: str = Field(min_length=1)
    text: str = Field(min_length=1)

    def embedding_text(self) -> str:
        """Текст для эмбеддинга: путь заголовков даёт куску контекст раздела."""
        return f"{self.breadcrumb}\n\n{self.text}"


def blob_url(repo_url: str, ref: str, path: str, start: int, end: int) -> str:
    """Ссылка на строки файла в форке. Для .md нужен ?plain=1, иначе GitHub
    покажет отрендеренную страницу, где якорей на строки нет."""
    plain = "?plain=1" if path.endswith(".md") else ""
    return f"{repo_url.rstrip('/')}/blob/{ref}/{path}{plain}#L{start}-L{end}"


def docs_site_url(path: str, anchor: str) -> str:
    """docs/usage.md + anchor → https://docs.paperless-ngx.com/usage/#anchor."""
    page = path.removeprefix("docs/").removesuffix(".md")
    page_url = DOCS_SITE_URL if page == "index" else f"{DOCS_SITE_URL}{page}/"
    return f"{page_url}#{anchor}" if anchor else page_url
