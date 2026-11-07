"""Trivy-гейт (scripts/trivy_gate.py): устранимых HIGH и CRITICAL в проде быть
не должно. Trivy и реестр подменены: проверяется решение гейта, а не сеть."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from trivy_gate import (
    GateError,
    Registry,
    annotation,
    compose_images,
    main,
    newest,
    our_image,
    third_party,
    version,
)

ROOT = Path(__file__).resolve().parent.parent
DIGEST_OLD = "sha256:" + "a" * 64
DIGEST_NEW = "sha256:" + "b" * 64

Vuln = tuple[str, str, str, str]  # CVE, пакет, версия, исправлено в («» — патча нет)


def report(
    *vulns: Vuln,
    cls: str = "os-pkgs",
    target: str = "образ (debian 13.7)",
    path: str = "",
) -> dict[str, Any]:
    return {
        "Results": [
            {
                "Target": target,
                "Class": cls,
                "Type": "debian" if cls == "os-pkgs" else "python-pkg",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": cve,
                        "PkgName": package,
                        "InstalledVersion": installed,
                        "FixedVersion": fixed,
                        "Severity": "HIGH",
                        **({"PkgPath": path} if path else {}),
                    }
                    for cve, package, installed, fixed in vulns
                ],
            }
        ]
    }


def project(tmp_path: Path, caddy_tag: str = "2.11.7") -> Path:
    """Мини-репозиторий: compose с нашим образом и одним сторонним, Dockerfile, uv.lock."""
    (tmp_path / "deploy" / "grocy").mkdir(parents=True)
    (tmp_path / "docker-compose.yml").write_text(
        "x-app-image: &app-image ${APP_IMAGE:-recipes-llm-app}\n"
        "services:\n"
        "  bot:\n"
        "    image: *app-image\n"
        "  caddy:\n"
        f"    image: docker.io/library/caddy:{caddy_tag}@{DIGEST_OLD}\n"
    )
    (tmp_path / "deploy" / "grocy" / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / "Dockerfile").write_text(
        "# комментарий\nFROM ghcr.io/astral-sh/uv:x\nRUN apt-get update && apt-get upgrade -y\n"
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "httpx2"\n\n[[package]]\nname = "pydantic-core"\n'
    )
    return tmp_path


class FakeRegistry:
    """Реестр по протоколу v2: 401 с адресом токена, страницы тегов, HEAD манифеста."""

    def __init__(self, tags: list[list[str]], digest: str = DIGEST_NEW) -> None:
        self.pages = tags
        self.digest = digest
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append((method, url, dict(headers)))
        if url.startswith("https://auth.example/token"):
            return 200, {}, json.dumps({"token": "t0ken"}).encode()
        if headers.get("Authorization") != "Bearer t0ken":
            challenge = 'Bearer realm="https://auth.example/token",service="reg",scope="repository:library/caddy:pull"'
            return 401, {"www-authenticate": challenge}, b""
        if "/tags/list" in url:
            page = 1 if "last=" in url else 0
            link = (
                {"link": '</v2/library/caddy/tags/list?last=x&n=1000>; rel="next"'}
                if page + 1 < len(self.pages)
                else {}
            )
            return 200, link, json.dumps({"tags": self.pages[page]}).encode()
        if "/manifests/" in url and method == "HEAD":
            return 200, {"docker-content-digest": self.digest}, b""
        return 404, {}, b""


def test_version_is_the_tag_shape_and_its_numbers() -> None:
    assert version("v4.7.1-ls348") == ("v#.#.#-ls#", (4, 7, 1, 348))
    assert version("13.3.0") == ("#.#.#", (13, 3, 0))


@pytest.mark.parametrize(
    ("tag", "tags", "expected"),
    [
        # Соседние ветки той же версии — другая форма тега.
        ("2.11.7", ["2.11.9", "2.11.9-alpine", "2.12.0-beta.1", "latest"], "2.11.9"),
        # Новая мажорная версия — решение человека.
        ("13.2.4", ["13.3.0", "14.0.0", "13.2.5"], "13.3.0"),
        (
            "v4.7.1-ls346",
            ["v4.7.1-ls348", "4.7.1", "amd64-v4.7.1-ls349"],
            "v4.7.1-ls348",
        ),
        # Новее нет — остаётся свой.
        ("v1.12.1", ["v1.11.0", "v1.12.0"], "v1.12.1"),
    ],
)
def test_newest_release_of_the_same_shape_and_major(
    tag: str, tags: list[str], expected: str
) -> None:
    assert newest(tag, tags) == expected


def test_compose_images_with_lines_skip_our_image(tmp_path: Path) -> None:
    images = compose_images(project(tmp_path))

    assert [(i.file, i.line, i.repository, i.tag) for i in images] == [
        ("docker-compose.yml", 6, "library/caddy", "2.11.7")
    ]


def test_image_without_digest_stops_the_gate(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "docker-compose.yml").write_text(
        "services:\n  x:\n    image: docker.io/x/y:1.0\n"
    )

    with pytest.raises(
        GateError, match=r"docker-compose.yml:3: образ без тега и digest"
    ):
        compose_images(root)


def test_registry_takes_an_anonymous_token_once_and_follows_pages() -> None:
    fake = FakeRegistry([["2.11.7", "2.11.8"], ["2.11.9"]])
    registry = Registry(fake)

    assert registry.tags("docker.io", "library/caddy") == ["2.11.7", "2.11.8", "2.11.9"]
    assert registry.digest("docker.io", "library/caddy", "2.11.9") == DIGEST_NEW
    token_requests = [c for c in fake.calls if c[1].startswith("https://auth.example")]
    assert len(token_requests) == 1
    assert "scope=repository%3Alibrary%2Fcaddy%3Apull" in token_requests[0][1]
    # Docker Hub — через registry-1.docker.io; digest — HEAD индекса образа.
    head = next(c for c in fake.calls if c[0] == "HEAD")
    assert head[1] == "https://registry-1.docker.io/v2/library/caddy/manifests/2.11.9"
    assert "application/vnd.oci.image.index.v1+json" in head[2]["Accept"]


def test_registry_error_stops_the_gate() -> None:
    def broken(
        method: str, url: str, headers: Mapping[str, str]
    ) -> tuple[int, dict[str, str], bytes]:
        return 503, {}, b""

    with pytest.raises(GateError, match="HTTP 503"):
        Registry(broken).tags("docker.io", "library/caddy")


def test_our_image_fixable_packages_point_to_where_to_fix_them(tmp_path: Path) -> None:
    """Debian — пересборка (строка apt-get upgrade), Python из .venv — uv lock,
    Python вне uv.lock — Dockerfile; без патча — предупреждение."""
    root = project(tmp_path)
    scanned = report(
        ("CVE-1", "apt", "3.0.3", "3.0.4"),
        ("CVE-2", "util-linux", "2.41", ""),
    )
    scanned["Results"] += [
        report(
            ("CVE-3", "pydantic_core", "2.46.5", "2.46.6"),
            cls="lang-pkgs",
            target="Python",
            path="app/.venv/lib/python3.14/site-packages/pydantic_core-2.46.5.dist-info/METADATA",
        )["Results"][0],
        report(
            ("CVE-4", "urllib3", "2.7.0", "2.8.0"),
            cls="lang-pkgs",
            target="Python",
            path="usr/local/lib/python3.14/site-packages/pip/_vendor/vendor.txt",
        )["Results"][0],
    ]

    problems, pending = our_image(scanned, root)

    assert [(p.file, p.line) for p in problems] == [
        ("Dockerfile", 3),
        ("uv.lock", 5),
        ("Dockerfile", 2),
    ]
    assert "apt-get upgrade" in problems[0].action
    assert "uv lock --upgrade-package pydantic_core" in problems[1].action
    assert "не из uv.lock" in problems[2].action
    assert [f.cve for f in pending[0].findings] == ["CVE-2"]


def test_third_party_fixable_is_what_the_newest_release_no_longer_has(
    tmp_path: Path,
) -> None:
    image = compose_images(project(tmp_path))[0]
    latest_ref = f"docker.io/library/caddy:2.11.9@{DIGEST_NEW}"
    pinned = report(
        ("CVE-OLD", "brotli", "1.2.0", "1.2.1"), ("CVE-ALL", "x/net", "0.59", "0.62")
    )
    latest = report(
        ("CVE-ALL", "x/net", "0.60", "0.62"), ("CVE-NEW", "musl", "1.2.6", "")
    )

    problems, pending = third_party(image, pinned, latest_ref, latest)

    assert len(problems) == 1
    assert problems[0].action.startswith(f"заменить на image: {latest_ref}")
    assert "приносит новые: CVE-NEW" in problems[0].action
    assert [f.cve for f in problems[0].findings] == ["CVE-OLD"]
    assert pending[0].reason == "есть и в свежем выпуске 2.11.9"
    assert [f.cve for f in pending[0].findings] == ["CVE-ALL"]


def test_third_party_without_a_newer_release_only_waits(tmp_path: Path) -> None:
    image = compose_images(project(tmp_path))[0]

    problems, pending = third_party(
        image, report(("CVE-1", "x/net", "0.59", "0.62")), image.ref, None
    )

    assert problems == []
    assert pending[0].reason == "новее выпуска нет"


def run_gate(
    root: Path,
    reports: dict[str, dict[str, Any]],
    fake: FakeRegistry,
    env: dict[str, str] | None = None,
) -> tuple[int, list[str], list[tuple[str, bool]]]:
    lines: list[str] = []
    scanned: list[tuple[str, bool]] = []

    def scan(ref: str, local: bool) -> dict[str, Any]:
        scanned.append((ref, local))
        return reports.get(ref, {"Results": []})

    code = main(
        ["--image", "recipes-llm-app"],
        scan=scan,
        fetch=fake,
        root=root,
        env=env or {},
        say=lines.append,
    )
    return code, lines, scanned


def test_gate_fails_with_the_line_to_replace_and_an_annotation(tmp_path: Path) -> None:
    root = project(tmp_path)
    pinned_ref = f"docker.io/library/caddy:2.11.7@{DIGEST_OLD}"
    latest_ref = f"docker.io/library/caddy:2.11.9@{DIGEST_NEW}"
    reports = {
        pinned_ref: report(("CVE-2026-111497", "brotli-libs", "1.2.0-r0", "1.2.1-r0")),
        latest_ref: report(),
    }
    summary = tmp_path / "summary.md"

    code, lines, scanned = run_gate(
        root,
        reports,
        FakeRegistry([["2.11.7", "2.11.9", "2.11.9-alpine"]]),
        {"GITHUB_ACTIONS": "true", "GITHUB_STEP_SUMMARY": str(summary)},
    )

    assert code == 1
    # Наш образ — из Docker этой машины, сторонние — из реестра.
    assert scanned == [
        ("recipes-llm-app", True),
        (pinned_ref, False),
        (latest_ref, False),
    ]
    assert (
        f"✗ docker-compose.yml:6 — заменить на image: {latest_ref}: устраняет 1 CVE"
        in lines
    )
    assert any(
        line.startswith("::error file=docker-compose.yml,line=6,")
        and "CVE-2026-111497" in line
        for line in lines
    )
    assert "Устранимых HIGH и CRITICAL — 1 в 1 месте" in "\n".join(lines)
    assert "## Trivy-гейт" in summary.read_text()


def test_gate_passes_and_warns_when_the_newest_release_has_the_same_cves(
    tmp_path: Path,
) -> None:
    root = project(tmp_path, caddy_tag="2.11.9")
    pinned_ref = f"docker.io/library/caddy:2.11.9@{DIGEST_OLD}"
    vuln = ("CVE-2026-78661", "golang.org/x/net", "v0.60.0", "0.62.0")
    # Тег перезалит с новым digest, но CVE в нём та же.
    reports = {
        pinned_ref: report(vuln),
        f"docker.io/library/caddy:2.11.9@{DIGEST_NEW}": report(vuln),
    }

    code, lines, _ = run_gate(root, reports, FakeRegistry([["2.11.9"]]))

    assert code == 0
    assert any(
        line.startswith(
            "! docker-compose.yml:6 — docker.io/library/caddy:2.11.9: 1 CVE"
        )
        for line in lines
    )
    assert "Устранимых HIGH и CRITICAL нет" in "\n".join(lines)
    # Без GITHUB_ACTIONS — без аннотаций.
    assert not any(line.startswith("::") for line in lines)


def test_clean_image_needs_no_registry(tmp_path: Path) -> None:
    fake = FakeRegistry([["2.11.9"]])

    code, _, scanned = run_gate(project(tmp_path), {}, fake)

    assert code == 0
    assert len(scanned) == 2
    assert fake.calls == []


def test_gate_that_could_not_check_says_so_with_code_2(tmp_path: Path) -> None:
    lines: list[str] = []

    def scan(ref: str, local: bool) -> dict[str, Any]:
        raise GateError("Trivy не проверил образ: нет сети")

    code = main(
        ["--image", "x"], scan=scan, root=project(tmp_path), env={}, say=lines.append
    )

    assert code == 2
    assert (
        lines[-1]
        == "Trivy-гейт не смог проверить образы: Trivy не проверил образ: нет сети"
    )


def test_annotation_escapes_what_github_treats_as_syntax() -> None:
    assert annotation("error", "a,b.yml", 3, "t: x", "50%\nдальше") == (
        "::error file=a%2Cb.yml,line=3,title=t%3A x::50%25%0Aдальше"
    )


def test_ci_scans_before_deploy_and_once_a_month() -> None:
    """scan ждёт build (сканирует отправленный образ), deploy ждёт scan; раз в
    месяц — только scan, по образу последнего коммита main."""
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    jobs = ci["jobs"]

    # yaml читает ключ on как True.
    assert ci[True]["schedule"] == [{"cron": "0 6 1 * *"}]
    assert jobs["scan"]["needs"] == "build"
    assert jobs["scan"]["permissions"] == {"contents": "read", "packages": "read"}
    assert set(jobs["deploy"]["needs"]) == {"check", "build", "scan"}
    assert "github.event_name == 'push'" in jobs["deploy"]["if"]
    assert "github.event_name != 'schedule'" in jobs["check"]["if"]
    assert "github.event_name != 'schedule'" in jobs["build"]["if"]
    steps = "\n".join(str(step.get("run", "")) for step in jobs["scan"]["steps"])
    assert "scripts/trivy_gate.py --image" in steps
    assert "ghcr.io/${GITHUB_REPOSITORY,,}:$GITHUB_SHA" in steps


def test_make_trivy_scans_a_fresh_build_of_our_image() -> None:
    makefile = (ROOT / "Makefile").read_text()

    assert "trivy: ## " in makefile
    assert "docker compose build assistant" in makefile
    assert (
        'scripts/trivy_gate.py --image "$$(docker compose config --images assistant)"'
        in makefile
    )
