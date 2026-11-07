"""Фейковый Bot API для тестов бота: без сети и настоящего токена.

Сессия aiogram записывает исходящие вызовы и отвечает, как ответил бы
Telegram; апдейты — словари, как из getUpdates.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.methods import (
    CopyMessage,
    CreateForumTopic,
    EditMessageText,
    SendMessage,
    SendPhoto,
    TelegramMethod,
)
from aiogram.methods.base import TelegramType


class FakeTelegram(BaseSession):
    """Bot API без сети: записывает вызовы и отвечает, как ответил бы Telegram."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []
        self._message_id = 100
        self.topics = 0
        # Тип метода, на который Telegram ответит ошибкой 400.
        self.fail: type[TelegramMethod[Any]] | None = None

    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[TelegramType],
        timeout: int | None = None,
    ) -> TelegramType:
        self.calls.append(method)
        result: object = True
        if isinstance(method, SendMessage):
            self._message_id += 1
            result = self._message(method.chat_id, self._message_id, method.text)
        elif isinstance(method, SendPhoto):
            self._message_id += 1
            result = self._message(method.chat_id, self._message_id, method.caption)
        elif isinstance(method, EditMessageText):
            result = self._message(method.chat_id, method.message_id, method.text)
        elif isinstance(method, CreateForumTopic):
            self.topics += 1
            result = {
                "message_thread_id": 500 + self.topics,
                "name": method.name,
                "icon_color": 7322096,
            }
        elif isinstance(method, CopyMessage):
            self._message_id += 1
            result = {"message_id": self._message_id}
        if self.fail is not None and isinstance(method, self.fail):
            content = json.dumps(
                {"ok": False, "error_code": 400, "description": "Bad Request: test"}
            )
            return self.check_response(bot, method, 400, content).result  # type: ignore[return-value]
        content = json.dumps({"ok": True, "result": result})
        return self.check_response(bot, method, 200, content).result  # type: ignore[return-value]

    @staticmethod
    def _message(
        chat_id: object, message_id: object, text: str | None
    ) -> dict[str, object]:
        return {
            "message_id": message_id,
            "date": int(time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        }

    async def close(self) -> None:
        pass

    async def stream_content(
        self, *args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        raise NotImplementedError
        yield b""

    def sent(self) -> list[str]:
        return [m.text for m in self.calls if isinstance(m, SendMessage)]

    def edits(self) -> list[EditMessageText]:
        return [m for m in self.calls if isinstance(m, EditMessageText)]


def update(
    text: str | None,
    *,
    chat_id: int = 7,
    chat_type: str = "private",
    age: float = 0.0,
    update_id: int = 1,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": update_id,
        "date": int(time.time() - age),
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": chat_id, "is_bot": False, "first_name": "Тест"},
    }
    if text is None:
        message["photo"] = [
            {"file_id": "p", "file_unique_id": "p", "width": 1, "height": 1}
        ]
    else:
        message["text"] = text
    return {"update_id": update_id, "message": message}


def press(
    data: str, *, message_id: int, markup: dict[str, Any] | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "message_id": message_id,
        "date": int(time.time()),
        "chat": {"id": 7, "type": "private"},
        "text": "ответ",
    }
    if markup is not None:
        message["reply_markup"] = markup
    return {
        "update_id": 100 + message_id,
        "callback_query": {
            "id": f"cb{message_id}",
            "from": {"id": 7, "is_bot": False, "first_name": "Тест"},
            "chat_instance": "ci",
            "data": data,
            "message": message,
        },
    }
