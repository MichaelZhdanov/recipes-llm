"""HTTP API ассистента.

Запуск: `uvicorn assistant.api:create_app --factory`.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from assistant import __version__
from assistant.config import Settings


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    paperless_ref: str


def create_app(settings: Settings | None = None) -> FastAPI:
    """Собирает приложение. Глобального состояния нет: всё, что нужно, передаётся сюда.

    Uvicorn с `--factory` вызывает функцию без аргументов, и настройки читаются
    из окружения. Тесты передают свои `Settings`.
    """
    config = settings if settings is not None else Settings()
    app = FastAPI(title="Paperless Assistant", version=__version__)

    @app.get("/health")
    def health() -> HealthResponse:
        """Liveness: процесс жив и отвечает. Qdrant и Paperless здесь не проверяются."""
        return HealthResponse(
            status="ok", version=__version__, paperless_ref=config.paperless_ref
        )

    return app
