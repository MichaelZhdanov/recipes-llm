"""Счётчики из логов сервера: входы по SSH и баны fail2ban (этап 8, F8).

`python -m assistant.hostlog` — сервис `hostlog` в compose (профиль host,
только на сервере). Читает `/var/log/auth.log` (sshd) и
`/var/log/fail2ban.log` хоста — только на чтение — и отдаёт счётчики на
`/metrics` для Prometheus:
- `recipes_ssh_logins_total{result}`: failed — неверный пароль или ключ,
  invalid_user — вход под несуществующим пользователем, accepted — вход;
- `recipes_fail2ban_actions_total{jail, action}`: ban и unban.

Готового экспортёра логов sshd среди официальных нет; mtail — ещё один
сторонний образ. Здесь — десяток регулярных выражений и чтение файла.

В метках нет ни IP, ни имён пользователей: перебор паролей идёт с тысяч
адресов, и метрика с меткой на адрес росла бы без предела. Кто и откуда —
видно в самом auth.log.

Файл читается с конца на момент запуска: счётчики начинаются с нуля, а
Prometheus считает прирост (increase). logrotate (rsyslog) создаёт новый
файл — сменился inode или файл стал короче, значит, читаем новый с начала.
Поэтому в контейнер монтируется каталог `/var/log`, а не файл: смонтированный
файл остался бы старым inode.
Усечение на месте (copytruncate) видно, только если файл к следующему
чтению короче прочитанного; rsyslog на сервере так не ротирует.
"""

import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, start_http_server

from assistant.config import Settings
from assistant.logs import setup_logging

logger = logging.getLogger(__name__)

PORT = 9109
EVERY = 15.0
LOG_DIR = Path(os.environ.get("HOSTLOG_DIR", "/host/log"))

# sshd 9.8+ пишет от имени sshd-session, раньше — sshd.
_SSHD = r"sshd(?:-session)?\[\d+\]: "
SSH_PATTERNS = {
    "failed": re.compile(_SSHD + r"Failed (?:password|publickey|none) for "),
    "invalid_user": re.compile(_SSHD + r"Invalid user "),
    "accepted": re.compile(_SSHD + r"Accepted "),
}
# «2026-11-07 08:30:01,123 fail2ban.actions [123]: NOTICE  [sshd] Ban 203.0.113.7»
FAIL2BAN = re.compile(r"fail2ban\.actions.*\[(?P<jail>[\w-]+)\] (?P<action>Ban|Unban) ")


@dataclass
class Follower:
    """Новые строки файла с прошлого чтения; ротацию видит по inode и размеру."""

    path: Path
    _inode: int | None = None
    _offset: int = 0
    _partial: str = field(default="", repr=False)

    def start_at_end(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        self._inode, self._offset = stat.st_ino, stat.st_size

    def lines(self) -> Iterator[str]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        if stat.st_ino != self._inode or stat.st_size < self._offset:
            # Новый файл после ротации (или файл появился): читаем с начала.
            self._inode, self._offset, self._partial = stat.st_ino, 0, ""
        if stat.st_size == self._offset:
            return
        with self.path.open("rb") as file:
            file.seek(self._offset)
            chunk = file.read()
        self._offset += len(chunk)
        text = self._partial + chunk.decode("utf-8", errors="replace")
        *complete, self._partial = text.split("\n")
        yield from complete


class HostLogMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.ssh = Counter(
            "recipes_ssh_logins",
            "Входы по SSH из auth.log: failed — неверный пароль или ключ, "
            "invalid_user — несуществующий пользователь, accepted — вход",
            ["result"],
            registry=self.registry,
        )
        self.fail2ban = Counter(
            "recipes_fail2ban_actions",
            "Баны и разбаны fail2ban из fail2ban.log",
            ["jail", "action"],
            registry=self.registry,
        )
        # Нули с самого начала: панель и алерт видят метрику, даже пока событий нет.
        for result in SSH_PATTERNS:
            self.ssh.labels(result)

    def auth_line(self, line: str) -> None:
        for result, pattern in SSH_PATTERNS.items():
            if pattern.search(line):
                self.ssh.labels(result).inc()

    def fail2ban_line(self, line: str) -> None:
        match = FAIL2BAN.search(line)
        if match:
            self.fail2ban.labels(match["jail"], match["action"].lower()).inc()


def run(log_dir: Path = LOG_DIR, every: float = EVERY, port: int = PORT) -> None:
    metrics = HostLogMetrics()
    auth, bans = Follower(log_dir / "auth.log"), Follower(log_dir / "fail2ban.log")
    auth.start_at_end()
    bans.start_at_end()
    start_http_server(port, registry=metrics.registry)
    logger.info("счётчики логов сервера на порту %s", port)
    while True:
        for line in auth.lines():
            metrics.auth_line(line)
        for line in bans.lines():
            metrics.fail2ban_line(line)
        time.sleep(every)


if __name__ == "__main__":
    # JSON-логи, как у бота: секретов у сервиса нет, Settings — значения по умолчанию.
    setup_logging(Settings())
    run()
