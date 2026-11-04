"""Исходники Paperless-ngx на нужном теге.

Архив тега скачивается с GitHub (без git на машине) и распаковывается в
`<base>/<ref>/`. У каждого тега своя папка: при смене PAPERLESS_REF старое
дерево не перемешивается с новым.

Распаковка безопасна: tarfile с `filter="data"` не даёт архиву записать файл
за пределы папки (`../`, абсолютные пути, опасные ссылки). Сначала всё
распаковывается во временную папку и только потом переименовывается, поэтому
прерванная загрузка не оставляет полудерево, которое выглядит готовым.
"""

import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import httpx2

from assistant.http import request_with_retries

# Для индекса нужны только документация и Python-код; фронтенд и прочее не распаковываем.
INCLUDED_DIRS = ("docs", "src")
SOURCE_META_FILE = ".source.json"


class SourceDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceTree:
    root: Path
    ref: str
    # SHA коммита из архива: git archive пишет его в pax-заголовок `comment`.
    commit: str | None


def archive_url(repo_url: str, ref: str) -> str:
    return f"{repo_url.rstrip('/')}/archive/refs/tags/{ref}.tar.gz"


def ensure_sources(
    repo_url: str, ref: str, base_dir: Path, client: httpx2.Client
) -> SourceTree:
    """Возвращает дерево исходников тега, при необходимости скачивая его."""
    target = base_dir / ref
    meta_path = target / SOURCE_META_FILE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return SourceTree(root=target, ref=ref, commit=meta.get("commit"))

    url = archive_url(repo_url, ref)
    base_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base_dir, prefix=".download-") as tmp:
        archive = Path(tmp) / "source.tar.gz"
        archive.write_bytes(_download(client, url))
        tree = Path(tmp) / "tree"
        commit = _extract(archive, tree)
        meta = {
            "ref": ref,
            "commit": commit,
            "archive_url": url,
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (tree / SOURCE_META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        if target.exists():
            # Папка без .source.json — след прерванной загрузки.
            shutil.rmtree(target)
        tree.rename(target)
    return SourceTree(root=target, ref=ref, commit=commit)


def _download(client: httpx2.Client, url: str) -> bytes:
    response = request_with_retries(lambda: client.get(url, follow_redirects=True))
    if response.status_code != 200:
        raise SourceDownloadError(
            f"не удалось скачать {url}: HTTP {response.status_code}. "
            "Проверьте PAPERLESS_REPO_URL и PAPERLESS_REF."
        )
    return response.content


def _extract(archive: Path, dest: Path) -> str | None:
    dest.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            # GitHub кладёт всё в верхнюю папку вида paperless-ngx-rag-3.4.2/ — срезаем её.
            parts = PurePosixPath(member.name).parts[1:]
            if parts and parts[0] in INCLUDED_DIRS:
                members.append(
                    member.replace(name=str(PurePosixPath(*parts)), deep=False)
                )
        commit = tar.pax_headers.get("comment")
        tar.extractall(dest, members=members, filter="data")
    return commit
