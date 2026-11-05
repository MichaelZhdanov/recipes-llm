"""Кусок индекса и ссылка на источник.

Каждый кусок знает, откуда он: путь, строки, версия данных (`ref`). Ссылка
`url` ведёт ровно на тот текст, который попал в индекс; `site_url` — на ту же
страницу на сайте, если она есть. На этапе 1 кусок станет рецептом.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkKind = Literal["doc", "code"]


class Chunk(BaseModel):
    """Кусок индекса. Все поля обязательны (приёмка F1)."""

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
