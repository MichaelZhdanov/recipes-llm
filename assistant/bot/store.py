"""Хранилище бота — SQLite из стандартной библиотеки, файл в томе `data/bot/`.

Что хранится и зачем:
- `sessions`, `messages` — история диалога: ассистент получает последние
  HISTORY_LIMIT реплик текущей сессии; `/new` или SESSION_TIMEOUT тишины
  начинают новую. Вопрос пишется после маскирования ПДн: в базе их нет.
  Ответ — текстом со ссылками povarenok.ru: по ним ассистент находит
  рецепты для уточнений («а во втором сколько муки?»);
- `answers` — ответы бота с оценками 👍/👎 и причиной 👎 (повод эскалации —
  этап 5);
- `pages` — кэш проверки страницы рецепта; `steps` — кэш шагов приготовления,
  которые написал ИИ: одни для всех пользователей.

Одно соединение на процесс под замком: запросы короткие, а бот отвечает
из нескольких потоков.
"""

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from assistant.bot.recipe_page import RecipeCard
from assistant.generation.gigachat import Message
from assistant.security.pii import mask_pii

HISTORY_LIMIT = 10
SESSION_TIMEOUT = 30 * 60
SCHEMA_VERSION = 2

Vote = Literal["up", "down"]
VoteReason = Literal["unclear", "wrong", "failed"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    started_at REAL NOT NULL,
    last_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_chat ON sessions (chat_id, id);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions (id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_session ON messages (session_id, id);
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    message_id INTEGER,
    status TEXT NOT NULL,
    reason TEXT,
    model TEXT NOT NULL,
    recipe_ids TEXT NOT NULL,
    seconds REAL NOT NULL,
    vote TEXT CHECK (vote IN ('up', 'down')),
    vote_reason TEXT CHECK (vote_reason IN ('unclear', 'wrong', 'failed')),
    created_at REAL NOT NULL,
    voted_at REAL
);
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    ok INTEGER NOT NULL,
    checked_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    recipe_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    approved INTEGER NOT NULL,
    prompt_version TEXT NOT NULL,
    created_at REAL NOT NULL
);
-- Карточка страницы рецепта: главное фото и сколько шагов (recipe_page.py).
CREATE TABLE IF NOT EXISTS cards (
    url TEXT PRIMARY KEY,
    photo TEXT,
    steps INTEGER NOT NULL,
    step_photos INTEGER NOT NULL,
    checked_at REAL NOT NULL
);
"""


@dataclass(frozen=True)
class SavedSteps:
    text: str
    approved: bool
    prompt_version: str


class Store:
    def __init__(
        self, path: Path | str, clock: Callable[[], float] = time.time
    ) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._db.commit()
        self._lock = threading.Lock()
        self._clock = clock

    def close(self) -> None:
        self._db.close()

    # --- История диалога ---

    def _session(self, chat_id: int, *, create: bool) -> int | None:
        """Текущая сессия чата; истёкшая — как будто её нет."""
        row = self._db.execute(
            "SELECT id, last_at FROM sessions WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        now = self._clock()
        if row is not None and now - row[1] <= SESSION_TIMEOUT:
            return int(row[0])
        if not create:
            return None
        cursor = self._db.execute(
            "INSERT INTO sessions (chat_id, started_at, last_at) VALUES (?, ?, ?)",
            (chat_id, now, now),
        )
        return int(cursor.lastrowid or 0)

    def history(self, chat_id: int, limit: int = HISTORY_LIMIT) -> list[Message]:
        """Последние limit реплик текущей сессии, по порядку."""
        with self._lock:
            session = self._session(chat_id, create=False)
            if session is None:
                return []
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session, limit),
            ).fetchall()
        messages: list[Message] = []
        for role, content in reversed(rows):
            messages.append({"role": role, "content": content})
        return messages

    def reset(self, chat_id: int) -> None:
        """/new: следующая реплика начнёт новую сессию."""
        with self._lock, self._db:
            now = self._clock()
            self._db.execute(
                "INSERT INTO sessions (chat_id, started_at, last_at) VALUES (?, ?, ?)",
                (chat_id, now, now),
            )

    def add_turn(self, chat_id: int, question: str, answer: str) -> None:
        """Вопрос (без ПДн) и ответ — одной транзакцией: история всегда парами."""
        with self._lock, self._db:
            session = self._session(chat_id, create=True)
            now = self._clock()
            self._db.executemany(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                [
                    (session, "user", mask_pii(question).text, now),
                    (session, "assistant", answer, now),
                ],
            )
            self._db.execute(
                "UPDATE sessions SET last_at = ? WHERE id = ?", (now, session)
            )

    # --- Ответы и оценки ---

    def save_answer(
        self,
        chat_id: int,
        *,
        status: str,
        reason: str | None,
        model: str,
        recipe_ids: Sequence[int],
        seconds: float,
    ) -> int:
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO answers (chat_id, status, reason, model, recipe_ids, seconds, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chat_id,
                    status,
                    reason,
                    model,
                    json.dumps(list(recipe_ids)),
                    seconds,
                    self._clock(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def set_message_id(self, answer_id: int, message_id: int) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE answers SET message_id = ? WHERE id = ?",
                (message_id, answer_id),
            )

    def vote(self, answer_id: int, chat_id: int, vote: Vote) -> bool:
        """Оценка ответа; False — ответа нет или он из другого чата."""
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE answers SET vote = ?, voted_at = ? WHERE id = ? AND chat_id = ?",
                (vote, self._clock(), answer_id, chat_id),
            )
            return cursor.rowcount == 1

    def vote_reason(self, answer_id: int, chat_id: int, reason: VoteReason) -> bool:
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE answers SET vote_reason = ? WHERE id = ? AND chat_id = ? AND vote = 'down'",
                (reason, answer_id, chat_id),
            )
            return cursor.rowcount == 1

    def recipe_ids(self, answer_id: int, chat_id: int) -> list[int] | None:
        """id рецептов ответа — для кнопки шагов; None — чужой или неизвестный ответ."""
        with self._lock:
            row = self._db.execute(
                "SELECT recipe_ids FROM answers WHERE id = ? AND chat_id = ?",
                (answer_id, chat_id),
            ).fetchone()
        return None if row is None else [int(i) for i in json.loads(row[0])]

    # --- Кэши ---

    def page_ok(self, url: str, max_age: float) -> bool | None:
        """Жива ли страница по кэшу; None — не проверяли или проверка устарела."""
        with self._lock:
            row = self._db.execute(
                "SELECT ok, checked_at FROM pages WHERE url = ?", (url,)
            ).fetchone()
        if row is None or self._clock() - row[1] > max_age:
            return None
        return bool(row[0])

    def save_page(self, url: str, ok: bool) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO pages (url, ok, checked_at) VALUES (?, ?, ?) "
                "ON CONFLICT (url) DO UPDATE SET ok = excluded.ok, checked_at = excluded.checked_at",
                (url, int(ok), self._clock()),
            )

    def card(self, url: str, max_age: float) -> RecipeCard | None:
        """Карточка страницы из кэша; None — не разбирали или разбор устарел."""
        with self._lock:
            row = self._db.execute(
                "SELECT photo, steps, step_photos, checked_at FROM cards WHERE url = ?",
                (url,),
            ).fetchone()
        if row is None or self._clock() - row[3] > max_age:
            return None
        return RecipeCard(row[0], int(row[1]), int(row[2]))

    def save_card(self, url: str, card: RecipeCard) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO cards (url, photo, steps, step_photos, checked_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (url) DO UPDATE SET "
                "photo = excluded.photo, steps = excluded.steps, "
                "step_photos = excluded.step_photos, checked_at = excluded.checked_at",
                (url, card.photo, card.steps, card.step_photos, self._clock()),
            )

    def steps(self, recipe_id: int, prompt_version: str) -> SavedSteps | None:
        """Шаги из кэша — только той же версии промпта."""
        with self._lock:
            row = self._db.execute(
                "SELECT text, approved, prompt_version FROM steps "
                "WHERE recipe_id = ? AND prompt_version = ?",
                (recipe_id, prompt_version),
            ).fetchone()
        return None if row is None else SavedSteps(row[0], bool(row[1]), row[2])

    def save_steps(
        self, recipe_id: int, text: str, approved: bool, prompt_version: str
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO steps (recipe_id, text, approved, prompt_version, created_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (recipe_id) DO UPDATE SET "
                "text = excluded.text, approved = excluded.approved, "
                "prompt_version = excluded.prompt_version, created_at = excluded.created_at",
                (recipe_id, text, int(approved), prompt_version, self._clock()),
            )
