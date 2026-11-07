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
  которые написал ИИ: одни для всех пользователей;
- `tickets` — обращения к человеку (этап 5): повод, сводка, тема в группе
  операторов, открыто или закрыто, оценка разговора. Открытое обращение у
  чата одно — это гарантирует уникальный индекс, а не код: пока оно открыто,
  сообщения чата идут человеку, ассистент молчит. Состояние в базе, поэтому
  переживает перезапуск бота. `last_at` — последнее сообщение в любую
  сторону: обращение без сообщений TICKET_IDLE закрывается само;
- `grocy` — склад Grocy (этап 6): включён ли он у чата и откуда искать в
  текущем разговоре («из того, что есть» или «по всем рецептам») — выбор
  привязан к сессии, новый разговор спрашивает заново; `pending` — вопрос,
  который ждёт этого выбора.

Одно соединение на процесс под замком: запросы короткие, а бот отвечает
из нескольких потоков.
"""

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from assistant.generation.gigachat import Message
from assistant.security.pii import mask_pii

HISTORY_LIMIT = 10
SESSION_TIMEOUT = 30 * 60
SCHEMA_VERSION = 7

Vote = Literal["up", "down"]
VoteReason = Literal["unclear", "wrong", "failed"]
TicketStatus = Literal["open", "closed"]
# Откуда искать рецепты при включённом Grocy: из того, что есть, или по всем.
Source = Literal["stock", "all"]

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
    voted_at REAL,
    question TEXT,
    missing TEXT
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
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    thread_id INTEGER,
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    vote TEXT CHECK (vote IN ('up', 'down')),
    opened_at REAL NOT NULL,
    closed_at REAL,
    last_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS tickets_open_chat ON tickets (chat_id)
    WHERE status = 'open';
CREATE INDEX IF NOT EXISTS tickets_thread ON tickets (thread_id);
CREATE TABLE IF NOT EXISTS grocy (
    chat_id INTEGER PRIMARY KEY,
    enabled INTEGER NOT NULL,
    source TEXT CHECK (source IN ('stock', 'all')),
    session_id INTEGER
);
CREATE TABLE IF NOT EXISTS pending (
    chat_id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS asks (
    chat_id INTEGER NOT NULL,
    at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS asks_chat ON asks (chat_id, at);
"""


@dataclass(frozen=True)
class SavedSteps:
    text: str
    approved: bool
    prompt_version: str


@dataclass(frozen=True)
class Ticket:
    """Обращение к человеку. thread_id — тема в группе операторов; None — ещё
    не создана (между записью в базу и ответом Telegram)."""

    id: int
    chat_id: int
    thread_id: int | None
    reason: str
    summary: str
    status: TicketStatus


_TICKET_COLUMNS = "id, chat_id, thread_id, reason, summary, status"


def _ticket(row: Any) -> Ticket | None:
    """Строка таблицы tickets (столбцы _TICKET_COLUMNS) → Ticket."""
    if row is None:
        return None
    id_, chat_id, thread_id, reason, summary, status = row
    return Ticket(id_, chat_id, thread_id, reason, summary, status)


class Store:
    def __init__(
        self, path: Path | str, clock: Callable[[], float] = time.time
    ) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._migrate()
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._db.commit()
        self._lock = threading.Lock()
        self._clock = clock

    def _migrate(self) -> None:
        """Версия 3 → 4: в tickets нет last_at (база создана до автозакрытия).
        CREATE TABLE IF NOT EXISTS столбцы в старую таблицу не добавляет."""
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(tickets)")}
        if "last_at" not in columns:
            self._db.execute("ALTER TABLE tickets ADD COLUMN last_at REAL")
        self._db.execute("UPDATE tickets SET last_at = opened_at WHERE last_at IS NULL")
        # 4 → 5: вопрос ответа — для расширенного поиска по складу.
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(answers)")}
        if "question" not in columns:
            self._db.execute("ALTER TABLE answers ADD COLUMN question TEXT")
        # 5 → 6: чего не хватает рецептам ответа — для шагов со страницы.
        if "missing" not in columns:
            self._db.execute("ALTER TABLE answers ADD COLUMN missing TEXT")

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
        question: str = "",
        missing: Mapping[int, Sequence[str]] | None = None,
    ) -> int:
        """question — без ПДн (маскируется): по нему расширенный поиск по складу;
        missing — чего не хватает рецептам расширенного поиска (recipe_id → продукты)."""
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO answers (chat_id, status, reason, model, recipe_ids, seconds, "
                "created_at, question, missing) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chat_id,
                    status,
                    reason,
                    model,
                    json.dumps(list(recipe_ids)),
                    seconds,
                    self._clock(),
                    mask_pii(question).text if question else None,
                    json.dumps({str(k): list(v) for k, v in missing.items()})
                    if missing
                    else None,
                ),
            )
            return int(cursor.lastrowid or 0)

    def missing_of(self, answer_id: int, chat_id: int, recipe_id: int) -> list[str]:
        """Чего не хватает рецепту этого ответа; пусто — всё есть или не по складу."""
        with self._lock:
            row = self._db.execute(
                "SELECT missing FROM answers WHERE id = ? AND chat_id = ?",
                (answer_id, chat_id),
            ).fetchone()
        if row is None or row[0] is None:
            return []
        return [str(name) for name in json.loads(row[0]).get(str(recipe_id), [])]

    def question_of(self, answer_id: int, chat_id: int) -> str | None:
        """Вопрос ответа этого чата; None — чужой, неизвестный или без вопроса."""
        with self._lock:
            row = self._db.execute(
                "SELECT question FROM answers WHERE id = ? AND chat_id = ?",
                (answer_id, chat_id),
            ).fetchone()
        return None if row is None or row[0] is None else str(row[0])

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

    def recent_statuses(self, chat_id: int, limit: int = 2) -> list[str]:
        """Статусы последних ответов чата, от нового к старому (только ответы
        текущей сессии): «два „не нашёл“ подряд» — повод предложить человека."""
        since = self._clock() - SESSION_TIMEOUT
        with self._lock:
            rows = self._db.execute(
                "SELECT status FROM answers WHERE chat_id = ? AND created_at >= ? "
                "ORDER BY id DESC LIMIT ?",
                (chat_id, since, limit),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def recipe_ids(self, answer_id: int, chat_id: int) -> list[int] | None:
        """id рецептов ответа — для кнопки шагов; None — чужой или неизвестный ответ."""
        with self._lock:
            row = self._db.execute(
                "SELECT recipe_ids FROM answers WHERE id = ? AND chat_id = ?",
                (answer_id, chat_id),
            ).fetchone()
        return None if row is None else [int(i) for i in json.loads(row[0])]

    # --- Обращения к человеку ---

    def open_ticket(self, chat_id: int, reason: str, summary: str) -> Ticket:
        """Новое обращение; если у чата уже есть открытое — оно и возвращается."""
        with self._lock, self._db:
            try:
                now = self._clock()
                cursor = self._db.execute(
                    "INSERT INTO tickets (chat_id, reason, summary, status, opened_at, "
                    "last_at) VALUES (?, ?, ?, 'open', ?, ?)",
                    (chat_id, reason, summary, now, now),
                )
            except sqlite3.IntegrityError:
                found = self._db.execute(
                    f"SELECT {_TICKET_COLUMNS} FROM tickets "
                    "WHERE chat_id = ? AND status = 'open'",
                    (chat_id,),
                ).fetchone()
            else:
                found = self._db.execute(
                    f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
        ticket = _ticket(found)
        assert ticket is not None
        return ticket

    def set_thread(self, ticket_id: int, thread_id: int) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE tickets SET thread_id = ? WHERE id = ?", (thread_id, ticket_id)
            )

    def open_ticket_of(self, chat_id: int) -> Ticket | None:
        """Открытое обращение чата: есть — сообщения идут человеку."""
        with self._lock:
            row = self._db.execute(
                f"SELECT {_TICKET_COLUMNS} FROM tickets "
                "WHERE chat_id = ? AND status = 'open'",
                (chat_id,),
            ).fetchone()
        return _ticket(row)

    def ticket_of_thread(self, thread_id: int) -> Ticket | None:
        """Открытое обращение темы в группе операторов."""
        with self._lock:
            row = self._db.execute(
                f"SELECT {_TICKET_COLUMNS} FROM tickets "
                "WHERE thread_id = ? AND status = 'open'",
                (thread_id,),
            ).fetchone()
        return _ticket(row)

    def touch_ticket(self, ticket_id: int) -> None:
        """Сообщение в обращении — в любую сторону: отсчёт тишины заново."""
        with self._lock, self._db:
            self._db.execute(
                "UPDATE tickets SET last_at = ? WHERE id = ?",
                (self._clock(), ticket_id),
            )

    def idle_tickets(self, idle: float) -> list[Ticket]:
        """Открытые обращения без сообщений дольше idle секунд."""
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_TICKET_COLUMNS} FROM tickets "
                "WHERE status = 'open' AND last_at < ? ORDER BY id",
                (self._clock() - idle,),
            ).fetchall()
        return [ticket for row in rows if (ticket := _ticket(row)) is not None]

    def close_ticket(self, ticket_id: int) -> bool:
        """False — уже закрыто: /close и /bot могут прийти одновременно."""
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE tickets SET status = 'closed', closed_at = ? "
                "WHERE id = ? AND status = 'open'",
                (self._clock(), ticket_id),
            )
            return cursor.rowcount == 1

    def vote_ticket(self, ticket_id: int, chat_id: int, vote: Vote) -> bool:
        """Оценка разговора с человеком — только своего закрытого обращения."""
        with self._lock, self._db:
            cursor = self._db.execute(
                "UPDATE tickets SET vote = ? WHERE id = ? AND chat_id = ? "
                "AND status = 'closed'",
                (vote, ticket_id, chat_id),
            )
            return cursor.rowcount == 1

    # --- Склад Grocy: включение, выбор на разговор, вопрос в ожидании ---

    def grocy_enabled(self, chat_id: int) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT enabled FROM grocy WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return row is not None and bool(row[0])

    def set_grocy(self, chat_id: int, enabled: bool) -> None:
        """Включить или выключить; выбор источника в разговоре сбрасывается."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO grocy (chat_id, enabled) VALUES (?, ?) ON CONFLICT (chat_id) "
                "DO UPDATE SET enabled = excluded.enabled, source = NULL, session_id = NULL",
                (chat_id, int(enabled)),
            )

    def source(self, chat_id: int) -> Source | None:
        """Откуда искать в текущем разговоре; None — ещё не выбрано (или новый разговор)."""
        with self._lock:
            session = self._session(chat_id, create=False)
            row = self._db.execute(
                "SELECT source, session_id FROM grocy WHERE chat_id = ? AND enabled = 1",
                (chat_id,),
            ).fetchone()
        if row is None or session is None or row[1] != session or row[0] is None:
            return None
        return "stock" if row[0] == "stock" else "all"

    def set_source(self, chat_id: int, source: Source) -> None:
        """Выбор — на текущий разговор (сессию): новый разговор спросит заново."""
        with self._lock, self._db:
            session = self._session(chat_id, create=True)
            self._db.execute(
                "UPDATE grocy SET source = ?, session_id = ? WHERE chat_id = ?",
                (source, session, chat_id),
            )

    def save_pending(self, chat_id: int, question: str) -> None:
        """Вопрос ждёт выбора источника (без ПДн, как в истории)."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO pending (chat_id, question, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT (chat_id) DO UPDATE SET question = excluded.question, "
                "created_at = excluded.created_at",
                (chat_id, mask_pii(question).text, self._clock()),
            )

    def pop_pending(self, chat_id: int) -> str | None:
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT question FROM pending WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            self._db.execute("DELETE FROM pending WHERE chat_id = ?", (chat_id,))
        return None if row is None else str(row[0])

    # --- Ограничение частоты (limits.py) ---

    def ask_times(self, chat_id: int, window: float) -> list[float]:
        """Когда чат звал модель за последние window секунд."""
        with self._lock:
            rows = self._db.execute(
                "SELECT at FROM asks WHERE chat_id = ? AND at > ? ORDER BY at",
                (chat_id, self._clock() - window),
            ).fetchall()
        return [float(row[0]) for row in rows]

    def add_ask(self, chat_id: int, keep: float) -> None:
        """Запомнить обращение к модели; старше keep секунд — не нужны."""
        now = self._clock()
        with self._lock, self._db:
            self._db.execute("DELETE FROM asks WHERE at <= ?", (now - keep,))
            self._db.execute(
                "INSERT INTO asks (chat_id, at) VALUES (?, ?)", (chat_id, now)
            )

    def now(self) -> float:
        return self._clock()

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
