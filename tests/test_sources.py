import io
import json
import tarfile
from pathlib import Path

import httpx2
import pytest

from assistant.indexing.sources import (
    SOURCE_META_FILE,
    SourceDownloadError,
    archive_url,
    ensure_sources,
)

REPO = "https://github.com/example/paperless-ngx-rag"
SHA = "d939d873e11dd14896d1bfe23f65b1baa6513ba7"


def make_archive(files: dict[str, str], commit: str | None = SHA) -> bytes:
    """Архив как у GitHub: всё лежит в верхней папке, SHA коммита в pax-заголовке."""
    buffer = io.BytesIO()
    pax = {"comment": commit} if commit else {}
    with tarfile.open(
        fileobj=buffer, mode="w:gz", format=tarfile.PAX_FORMAT, pax_headers=pax
    ) as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def make_client(archive: bytes, calls: list[str], status: int = 200) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(str(request.url))
        return httpx2.Response(status, content=archive)

    return httpx2.Client(transport=httpx2.MockTransport(handler))


TOP = "paperless-ngx-rag-3.4.2"
FILES = {
    f"{TOP}/docs/usage.md": "# Usage\n",
    f"{TOP}/src/documents/consumer.py": "def run():\n    pass\n",
    f"{TOP}/src-ui/main.ts": "console.log(1)\n",
    f"{TOP}/README.md": "readme\n",
}


def test_downloads_and_extracts_only_docs_and_src(tmp_path: Path) -> None:
    calls: list[str] = []

    tree = ensure_sources(
        REPO, "v3.4.2", tmp_path, make_client(make_archive(FILES), calls)
    )

    assert calls == [archive_url(REPO, "v3.4.2")]
    assert tree.root == tmp_path / "v3.4.2"
    assert (tree.root / "docs/usage.md").read_text() == "# Usage\n"
    assert (tree.root / "src/documents/consumer.py").exists()
    assert not (tree.root / "src-ui").exists()
    assert not (tree.root / "README.md").exists()


def test_reads_commit_sha_and_writes_meta(tmp_path: Path) -> None:
    tree = ensure_sources(
        REPO, "v3.4.2", tmp_path, make_client(make_archive(FILES), [])
    )

    assert tree.commit == SHA
    meta = json.loads((tree.root / SOURCE_META_FILE).read_text())
    assert meta["ref"] == "v3.4.2"
    assert meta["commit"] == SHA


def test_second_call_does_not_download_again(tmp_path: Path) -> None:
    calls: list[str] = []
    client = make_client(make_archive(FILES), calls)

    ensure_sources(REPO, "v3.4.2", tmp_path, client)
    tree = ensure_sources(REPO, "v3.4.2", tmp_path, client)

    assert len(calls) == 1
    assert tree.commit == SHA


def test_each_ref_gets_its_own_folder(tmp_path: Path) -> None:
    client = make_client(make_archive(FILES), [])

    old = ensure_sources(REPO, "v3.4.1", tmp_path, client)
    new = ensure_sources(REPO, "v3.4.2", tmp_path, client)

    assert old.root != new.root
    assert old.root.exists()
    assert new.root.exists()


def test_rejects_archive_that_escapes_target_dir(tmp_path: Path) -> None:
    evil = make_archive({f"{TOP}/docs/../../../evil.txt": "pwned"})
    base = tmp_path / "sources"

    with pytest.raises(tarfile.OutsideDestinationError):
        ensure_sources(REPO, "v3.4.2", base, make_client(evil, []))

    assert not (tmp_path / "evil.txt").exists()
    assert not (base / "v3.4.2").exists()


def test_leftover_folder_without_meta_is_replaced(tmp_path: Path) -> None:
    broken = tmp_path / "v3.4.2"
    (broken / "docs").mkdir(parents=True)
    (broken / "docs/half.md").write_text("прерванная загрузка")

    tree = ensure_sources(
        REPO, "v3.4.2", tmp_path, make_client(make_archive(FILES), [])
    )

    assert not (tree.root / "docs/half.md").exists()
    assert (tree.root / "docs/usage.md").exists()


def test_clear_error_when_tag_not_found(tmp_path: Path) -> None:
    with pytest.raises(SourceDownloadError, match="PAPERLESS_REF"):
        ensure_sources(REPO, "v9.9.9", tmp_path, make_client(b"", [], status=404))
