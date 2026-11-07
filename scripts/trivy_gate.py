"""Trivy-гейт: устранимых HIGH и CRITICAL в проде быть не должно.

    python scripts/trivy_gate.py --image <образ ассистента>

Проверяется всё, что работает в проде:
- образ ассистента (бот, hostlog и сид Grocy — тот же образ): в CI — тот, что
  собрала задача build и скачает сервер; зависимости из uv.lock Trivy видит в
  нём, в /app/.venv;
- сторонние образы из docker-compose.yml и deploy/grocy/docker-compose.yml — по
  закреплённому digest, ровно то, что скачивает сервер.

Устранимая уязвимость — та, от которой избавляет обновление:
- в образе ассистента — у неё есть исправленная версия: пакет Debian придёт при
  пересборке (apt-get upgrade в Dockerfile), пакет Python — после
  `uv lock --upgrade-package`;
- в стороннем образе — её нет в самом свежем выпуске той же мажорной версии
  (новый тег или тот же тег, перезалитый с новым digest): строку image нужно
  заменить, гейт печатает новую.
Неустранимая — патча ещё нет или он не дошёл до выпуска образа: обновлять пока
нечего. Она печатается предупреждением и деплой не останавливает.

Состояние между запусками не хранится: каждый запуск сканирует и то, что в
проде, и свежий выпуск с сегодняшней базой Trivy.

Код выхода: 0 — устранимых нет, 1 — есть, 2 — проверить не удалось (Trivy или
реестр недоступны). В GitHub Actions каждая находка — ещё и аннотация на
строке, где её устранять, а весь отчёт — в сводке задачи.

Только стандартная библиотека; Trivy — в контейнере с зафиксированной версией.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# Образ Trivy — с GHCR, а не с Docker Hub: не тратит лимит скачиваний Docker Hub.
TRIVY = "ghcr.io/aquasecurity/trivy:0.76.1"
SEVERITY = "HIGH,CRITICAL"
# Как на сервере: из индекса стороннего образа Trivy берёт linux/amd64.
PLATFORM = "linux/amd64"
COMPOSE_FILES = ("docker-compose.yml", "deploy/grocy/docker-compose.yml")
# База уязвимостей Trivy между запусками на этой машине (в CI каждый раз свежая).
CACHE = ROOT / "data" / "trivy"
# Docker Hub отвечает на запросы реестра с registry-1.docker.io.
REGISTRY_HOSTS = {"docker.io": "registry-1.docker.io"}
# Digest индекса (все платформы) — как в compose, а не одной платформы.
MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
# Сколько CVE печатать под одной находкой.
SHOWN = 12

IMAGE_LINE = re.compile(r"^\s*image:\s*(\S+)\s*$")
IMAGE_REF = re.compile(
    r"(?P<registry>[a-z0-9-]+(?:\.[a-z0-9-]+)+)/(?P<repository>[a-z0-9._/-]+)"
    r":(?P<tag>[\w.-]+)@(?P<digest>sha256:[0-9a-f]{64})"
)


class GateError(RuntimeError):
    """Проверить не удалось: Trivy или реестр недоступны, compose не тот."""


@dataclass(frozen=True)
class Finding:
    cve: str
    package: str
    installed: str
    # Пусто — патча нет.
    fixed: str
    severity: str
    # Пакет системы (Debian, Alpine), а не библиотека языка.
    system: bool
    # Файл в образе или цель Trivy: app/.venv/…/METADATA, usr/bin/caddy…
    path: str

    @property
    def key(self) -> tuple[str, str]:
        return self.cve, self.package


@dataclass(frozen=True)
class Image:
    """Строка image: стороннего образа в compose-файле."""

    file: str
    line: int
    registry: str
    repository: str
    tag: str
    digest: str

    @property
    def name(self) -> str:
        return f"{self.registry}/{self.repository}:{self.tag}"

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.digest}"


@dataclass(frozen=True)
class Problem:
    """Устранимые находки: что сделать и в какой строке."""

    file: str
    line: int
    action: str
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class Pending:
    """Неустранимые пока: что это и почему обновлять нечего."""

    file: str
    line: int
    subject: str
    reason: str
    findings: tuple[Finding, ...]


def findings(report: Mapping[str, Any]) -> list[Finding]:
    """Находки из JSON-отчёта Trivy."""
    result = []
    for target in report.get("Results") or []:
        for vuln in target.get("Vulnerabilities") or []:
            result.append(
                Finding(
                    cve=vuln["VulnerabilityID"],
                    package=vuln["PkgName"],
                    installed=vuln.get("InstalledVersion", ""),
                    fixed=vuln.get("FixedVersion", ""),
                    severity=vuln.get("Severity", ""),
                    system=target.get("Class") == "os-pkgs",
                    path=vuln.get("PkgPath") or target.get("Target", ""),
                )
            )
    return result


def unique(items: Iterable[Finding]) -> list[Finding]:
    """Одна CVE в одном пакете — одна строка (в Grafana она бывает в 15 файлах)."""
    seen: dict[tuple[str, str], Finding] = {}
    for item in items:
        seen.setdefault(item.key, item)
    return sorted(seen.values(), key=lambda f: (f.severity != "CRITICAL", f.cve))


def compose_images(root: Path, files: Sequence[str] = COMPOSE_FILES) -> list[Image]:
    """Сторонние образы с номером строки; наш образ (*app-image) — не здесь."""
    images = []
    for name in files:
        text = (root / name).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            match = IMAGE_LINE.match(line)
            if not match or match[1].startswith("*"):
                continue
            ref = IMAGE_REF.fullmatch(match[1])
            if not ref:
                raise GateError(
                    f"{name}:{number}: образ без тега и digest — {match[1]}"
                )
            images.append(Image(name, number, **ref.groupdict()))
    return images


def version(tag: str) -> tuple[str, tuple[int, ...]]:
    """Форма тега и числа в нём: v4.7.1-ls348 → ("v#.#.#-ls#", (4, 7, 1, 348))."""
    return re.sub(r"\d+", "#", tag), tuple(int(n) for n in re.findall(r"\d+", tag))


def newest(tag: str, tags: Iterable[str]) -> str:
    """Самый свежий тег той же формы и той же мажорной версии (первое число).

    Форма отсекает соседние ветки (13.3.0-ubuntu, v1.20.0-gpu, 4.0.0-rc1);
    переход на новую мажорную версию — решение человека, а не гейта.
    """
    shape, numbers = version(tag)
    same = [
        t for t in tags if version(t)[0] == shape and version(t)[1][:1] == numbers[:1]
    ]
    return max([tag, *same], key=lambda t: version(t)[1])


Response = tuple[int, dict[str, str], bytes]
Fetch = Callable[[str, str, Mapping[str, str]], Response]


def _lower(headers: Message) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def http_fetch(method: str, url: str, headers: Mapping[str, str]) -> Response:
    request = urllib.request.Request(url, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, _lower(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, _lower(error.headers), error.read()
    except urllib.error.URLError as error:
        raise GateError(f"{method} {url}: {error.reason}") from error


class Registry:
    """Чтение реестра по протоколу Docker Registry v2 без учётной записи.

    На первый запрос реестр отвечает 401 и в WWW-Authenticate называет, где
    взять токен на чтение репозитория; так устроены Docker Hub, GHCR и lscr.io.
    """

    def __init__(self, fetch: Fetch = http_fetch) -> None:
        self._fetch = fetch
        self._tokens: dict[tuple[str, str], str] = {}

    def tags(self, registry: str, repository: str) -> list[str]:
        """Все теги репозитория: реестр отдаёт их страницами, следующая — в Link."""
        tags: list[str] = []
        path = f"/v2/{repository}/tags/list?n=1000"
        while path:
            headers, body = self._get("GET", registry, repository, path)
            tags += json.loads(body).get("tags") or []
            link = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
            path = link[1] if link else ""
        return tags

    def digest(self, registry: str, repository: str, tag: str) -> str:
        """Digest, на который тег указывает сейчас. HEAD Docker Hub не считает
        скачиванием образа — лимит скачиваний он не тратит."""
        headers, _ = self._get(
            "HEAD",
            registry,
            repository,
            f"/v2/{repository}/manifests/{tag}",
            MANIFEST_TYPES,
        )
        digest = headers.get("docker-content-digest", "")
        if not digest.startswith("sha256:"):
            raise GateError(f"{registry}/{repository}:{tag}: реестр не назвал digest")
        return digest

    def _get(
        self, method: str, registry: str, repository: str, path: str, accept: str = ""
    ) -> tuple[dict[str, str], bytes]:
        host = REGISTRY_HOSTS.get(registry, registry)
        url = path if path.startswith("https://") else f"https://{host}{path}"
        key = (host, repository)
        for _ in range(2):
            headers = {"Accept": accept} if accept else {}
            if key in self._tokens:
                headers["Authorization"] = f"Bearer {self._tokens[key]}"
            status, response_headers, body = self._fetch(method, url, headers)
            if status == 401 and key not in self._tokens:
                self._tokens[key] = self._token(
                    response_headers.get("www-authenticate", "")
                )
                continue
            if status != 200:
                raise GateError(f"{host}/{repository}: {method} {path} — HTTP {status}")
            return response_headers, body
        raise GateError(f"{host}/{repository}: реестр не принял токен")

    def _token(self, challenge: str) -> str:
        params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        if not challenge.lower().startswith("bearer") or "realm" not in params:
            raise GateError(f"реестр просит неизвестный вход: {challenge[:80]!r}")
        query = urllib.parse.urlencode(
            {k: v for k, v in params.items() if k in ("service", "scope")}
        )
        status, _, body = self._fetch("GET", f"{params['realm']}?{query}", {})
        if status != 200:
            raise GateError(f"токен реестра ({params['realm']}): HTTP {status}")
        data = json.loads(body)
        token = data.get("token") or data.get("access_token")
        if not token:
            raise GateError(f"токен реестра ({params['realm']}): пустой ответ")
        return str(token)


Scan = Callable[[str, bool], Mapping[str, Any]]


def trivy(ref: str, local: bool) -> Mapping[str, Any]:
    """JSON-отчёт Trivy. local — образ из Docker этой машины (через сокет), иначе
    — из реестра, платформа сервера."""
    CACHE.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "-v", f"{CACHE}:/root/.cache"]
    if local:
        command += ["-v", "/var/run/docker.sock:/var/run/docker.sock"]
    command += [TRIVY, "image", "--quiet", "--scanners", "vuln"]
    command += ["--severity", SEVERITY, "--format", "json"]
    if local:
        command += ["--image-src", "docker"]
    else:
        command += ["--image-src", "remote", "--platform", PLATFORM]
    result = subprocess.run(
        [*command, ref], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise GateError(f"Trivy не проверил {ref}: {result.stderr.strip()[-400:]}")
    report: Mapping[str, Any] = json.loads(result.stdout)
    return report


def _line(lines: Sequence[str], needle: str) -> int:
    return next((n for n, line in enumerate(lines, 1) if needle in line), 1)


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def our_image(
    report: Mapping[str, Any], root: Path
) -> tuple[list[Problem], list[Pending]]:
    """Образ ассистента: устранимо всё, у чего Trivy знает исправленную версию."""
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8").splitlines()
    lock = (root / "uv.lock").read_text(encoding="utf-8").splitlines()
    found = findings(report)
    problems = []

    system = tuple(f for f in found if f.fixed and f.system)
    if system:
        problems.append(
            Problem(
                "Dockerfile",
                _line(dockerfile, "apt-get upgrade"),
                "пакеты системы — пересобрать образ: apt-get upgrade поставит исправления "
                "(локально — docker compose build --no-cache)",
                system,
            )
        )
    packages: dict[str, list[Finding]] = {}
    for item in found:
        if item.fixed and not item.system:
            packages.setdefault(item.package, []).append(item)
    for package, group in packages.items():
        if all("/.venv/" in f"/{f.path}" for f in group):
            problems.append(
                Problem(
                    "uv.lock",
                    _line(
                        [_normalized(line) for line in lock],
                        f'name = "{_normalized(package)}"',
                    ),
                    f"{package} — uv lock --upgrade-package {package}",
                    tuple(group),
                )
            )
        else:
            problems.append(
                Problem(
                    "Dockerfile",
                    _line(dockerfile, "FROM "),
                    f"{package} не из uv.lock ({group[0].path}) — обновить или убрать в Dockerfile",
                    tuple(group),
                )
            )

    unfixed = tuple(f for f in found if not f.fixed)
    pending = (
        [
            Pending(
                "Dockerfile",
                _line(dockerfile, "FROM "),
                "образ ассистента",
                "патча пока нет",
                unfixed,
            )
        ]
        if unfixed
        else []
    )
    return problems, pending


def third_party(
    image: Image,
    pinned: Mapping[str, Any],
    latest_ref: str,
    latest: Mapping[str, Any] | None,
) -> tuple[list[Problem], list[Pending]]:
    """Сторонний образ: устранимо то, чего нет в самом свежем выпуске."""
    found = findings(pinned)
    if latest is None:
        return [], [
            Pending(
                image.file, image.line, image.name, "новее выпуска нет", tuple(found)
            )
        ]
    newer = findings(latest)
    newer_keys = {f.key for f in newer}
    fixed = tuple(f for f in found if f.key not in newer_keys)
    still = tuple(f for f in found if f.key in newer_keys)
    tag = latest_ref.rsplit("@", 1)[0].rsplit(":", 1)[1]

    problems = []
    if fixed:
        added = unique(f for f in newer if f.key not in {g.key for g in found})
        action = f"заменить на image: {latest_ref}"
        if added:
            ids = ", ".join(f.cve for f in added[:5])
            action += (
                f" — но выпуск приносит новые: {ids}{' …' if len(added) > 5 else ''}"
            )
        problems.append(Problem(image.file, image.line, action, fixed))
    pending = (
        [
            Pending(
                image.file,
                image.line,
                image.name,
                f"есть и в свежем выпуске {tag}",
                still,
            )
        ]
        if still
        else []
    )
    return problems, pending


def check(
    app_image: str,
    root: Path,
    scan: Scan,
    registry: Registry,
    say: Callable[[str], None],
) -> tuple[list[Problem], list[Pending]]:
    problems: list[Problem] = []
    pending: list[Pending] = []

    say(f"== образ ассистента: {app_image}")
    found, waiting = our_image(scan(app_image, True), root)
    problems += found
    pending += waiting

    for image in compose_images(root):
        say(f"== {image.name} ({image.file}:{image.line})")
        pinned = scan(image.ref, False)
        if not findings(pinned):
            continue
        tag = newest(image.tag, registry.tags(image.registry, image.repository))
        digest = registry.digest(image.registry, image.repository, tag)
        latest_ref = f"{image.registry}/{image.repository}:{tag}@{digest}"
        latest = None
        if (tag, digest) != (image.tag, image.digest):
            say(f"   свежий выпуск: {tag}@{digest[:19]}…")
            latest = scan(latest_ref, False)
        found, waiting = third_party(image, pinned, latest_ref, latest)
        problems += found
        pending += waiting
    return problems, pending


def _count(items: Iterable[Finding]) -> int:
    return len({f.key for f in items})


def _listed(items: Iterable[Finding]) -> list[str]:
    rows = [
        f"    {f.cve} {f.severity} {f.package} {f.installed}"
        + (f" → {f.fixed}" if f.fixed else "")
        for f in unique(items)
    ]
    if len(rows) > SHOWN:
        rows = [*rows[:SHOWN], f"    … и ещё {len(rows) - SHOWN}"]
    return rows


def _ids(items: Iterable[Finding]) -> str:
    cves = list(dict.fromkeys(f.cve for f in unique(items)))
    return ", ".join(cves[:5]) + (" …" if len(cves) > 5 else "")


def _places(n: int) -> str:
    return f"{n} месте" if n % 10 == 1 and n % 100 != 11 else f"{n} местах"


def _escape(text: str, *, prop: bool = False) -> str:
    """Экранирование команд GitHub Actions (::error …::)."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def annotation(level: str, file: str, line: int, title: str, message: str) -> str:
    return (
        f"::{level} file={_escape(file, prop=True)},line={line},"
        f"title={_escape(title, prop=True)}::{_escape(message)}"
    )


def render(problems: Sequence[Problem], pending: Sequence[Pending]) -> list[str]:
    """Отчёт для человека: ✗ — что обновить и где, ! — что ждёт патча."""
    lines: list[str] = [""]
    for problem in problems:
        lines.append(
            f"✗ {problem.file}:{problem.line} — {problem.action}: "
            f"устраняет {_count(problem.findings)} CVE"
        )
        lines += _listed(problem.findings)
    for item in pending:
        lines.append(
            f"! {item.file}:{item.line} — {item.subject}: "
            f"{_count(item.findings)} CVE, {item.reason}"
        )
        lines += _listed(item.findings)
    waiting = sum(_count(item.findings) for item in pending)
    if problems:
        fixable = sum(_count(problem.findings) for problem in problems)
        lines.append(
            f"\nУстранимых HIGH и CRITICAL — {fixable} в {_places(len(problems))} (✗): "
            "в проде их быть не должно. Неустранимых пока — "
            f"{waiting} (!)."
        )
    else:
        lines.append(
            "\nУстранимых HIGH и CRITICAL нет. Неустранимых пока — "
            f"{waiting} (!): патча нет или он не дошёл до выпуска образа."
        )
    return lines


def annotations(problems: Sequence[Problem], pending: Sequence[Pending]) -> list[str]:
    result = [
        annotation(
            "error",
            p.file,
            p.line,
            "Trivy — устранимая уязвимость",
            f"{p.action}: устраняет {_count(p.findings)} CVE ({_ids(p.findings)})",
        )
        for p in problems
    ]
    result += [
        annotation(
            "warning",
            q.file,
            q.line,
            "Trivy — патча пока нет",
            f"{q.subject}: {_count(q.findings)} CVE, {q.reason} ({_ids(q.findings)})",
        )
        for q in pending
    ]
    return result


def _say(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    scan: Scan = trivy,
    fetch: Fetch = http_fetch,
    root: Path = ROOT,
    env: Mapping[str, str] = os.environ,
    say: Callable[[str], None] = _say,
) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/trivy_gate.py",
        description="Trivy-гейт: устранимых HIGH и CRITICAL в проде быть не должно.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="образ ассистента, уже лежащий в Docker этой машины",
    )
    args = parser.parse_args(argv)

    try:
        problems, pending = check(args.image, root, scan, Registry(fetch), say)
    except GateError as error:
        say(f"Trivy-гейт не смог проверить образы: {error}")
        return 2

    text = render(problems, pending)
    for line in text:
        say(line)
    if env.get("GITHUB_ACTIONS") == "true":
        for line in annotations(problems, pending):
            say(line)
    summary = env.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as file:
            file.write(
                "## Trivy-гейт\n\n```text\n" + "\n".join(text).strip() + "\n```\n"
            )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
