"""Счётчики из логов сервера (этап 8): входы SSH и баны fail2ban — регулярные
выражения на строках sshd и fail2ban, чтение файла с ротацией."""

from pathlib import Path

from assistant.hostlog import Follower, HostLogMetrics

# Синтетические строки в формате Ubuntu 26.04 (адреса — TEST-NET-3).
AUTH = [
    "Nov 07 08:27:14 host sshd-session[85044]: Invalid user linda from 203.0.113.7 port 65351",
    "Nov 07 08:27:15 host sshd-session[85044]: Failed password for invalid user linda from 203.0.113.7 port 65351 ssh2",
    "Nov 07 08:27:25 host sshd-session[85108]: Failed password for root from 203.0.113.8 port 52082 ssh2",
    "Nov 07 08:30:00 host sshd[900]: Accepted publickey for root from 198.51.100.4 port 40000 ssh2",
    "Nov 07 08:30:01 host CRON[901]: pam_unix(cron:session): session opened for user root",
]
FAIL2BAN = [
    "2026-11-07 08:31:00,100 fail2ban.actions        [700]: NOTICE  [sshd] Ban 203.0.113.7",
    "2026-11-07 09:31:00,100 fail2ban.actions        [700]: NOTICE  [sshd] Unban 203.0.113.7",
    "2026-11-07 08:31:00,200 fail2ban.filter         [700]: INFO    [sshd] Found 203.0.113.7",
]


def value(metrics: HostLogMetrics, name: str, **labels: str) -> float:
    return metrics.registry.get_sample_value(name, labels) or 0.0


def test_ssh_lines_are_counted_by_result_without_ip_or_user() -> None:
    metrics = HostLogMetrics()

    for line in AUTH:
        metrics.auth_line(line)

    assert value(metrics, "recipes_ssh_logins_total", result="failed") == 2
    assert value(metrics, "recipes_ssh_logins_total", result="invalid_user") == 1
    assert value(metrics, "recipes_ssh_logins_total", result="accepted") == 1
    text = "\n".join(
        f"{s.name}{s.labels}" for m in metrics.registry.collect() for s in m.samples
    )
    assert "203.0.113" not in text and "linda" not in text


def test_fail2ban_bans_and_unbans_are_counted_per_jail() -> None:
    metrics = HostLogMetrics()

    for line in FAIL2BAN:
        metrics.fail2ban_line(line)

    assert (
        value(metrics, "recipes_fail2ban_actions_total", jail="sshd", action="ban") == 1
    )
    assert (
        value(metrics, "recipes_fail2ban_actions_total", jail="sshd", action="unban")
        == 1
    )


def test_counters_exist_at_zero_before_any_event() -> None:
    metrics = HostLogMetrics()

    assert (
        metrics.registry.get_sample_value(
            "recipes_ssh_logins_total", {"result": "failed"}
        )
        == 0.0
    )


def test_follower_reads_only_new_complete_lines(tmp_path: Path) -> None:
    log = tmp_path / "auth.log"
    log.write_text("старое\n")
    follower = Follower(log)
    follower.start_at_end()

    with log.open("a") as file:
        file.write("первая\nвторая без конца")
    assert list(follower.lines()) == ["первая"]
    with log.open("a") as file:
        file.write(" строки\n")
    assert list(follower.lines()) == ["вторая без конца строки"]
    assert list(follower.lines()) == []


def test_follower_reads_the_new_file_after_rotation_or_truncation(
    tmp_path: Path,
) -> None:
    log = tmp_path / "auth.log"
    log.write_text("до ротации\n")
    follower = Follower(log)
    follower.start_at_end()

    # logrotate (rsyslog): старый файл переименован, новый создан заново.
    log.rename(tmp_path / "auth.log.1")
    log.write_text("после ротации\n")
    assert list(follower.lines()) == ["после ротации"]

    # Усечение на месте (copytruncate): файл стал короче прочитанного.
    log.write_text("")
    with log.open("a") as file:
        file.write("усечён\n")
    assert list(follower.lines()) == ["усечён"]


def test_missing_file_is_read_from_the_start_once_it_appears(tmp_path: Path) -> None:
    """fail2ban.log появляется только после установки fail2ban."""
    log = tmp_path / "fail2ban.log"
    follower = Follower(log)
    follower.start_at_end()
    assert list(follower.lines()) == []

    log.write_text(FAIL2BAN[0] + "\n")

    assert list(follower.lines()) == [FAIL2BAN[0]]
