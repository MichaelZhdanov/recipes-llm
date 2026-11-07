"""Снимок индекса в релизе GitHub: make index-download и make index-upload.

Полная сборка индекса — ≈ 200 ₽ эмбеддингов и около часа. Тот же индекс
снимком Qdrant лежит в релизе `index` репозитория и скачивается бесплатно,
поэтому скачивание — обычный способ получить индекс; сборка — только когда
снимка для нужных настроек нет.

Релиз один на все настройки: у снимка ключ из паспорта индекса — ревизия
датасета, модель эмбеддингов, выборка (`snapshot_key`). В релизе два файла:
`<ключ>.snapshot` (снимок Qdrant) и `<ключ>.json` (коллекция, sha256, размер,
паспорт). Адрес вычисляется из настроек: после пересборки код не меняется,
файлы с тем же ключом заменяются.

Оба действия идемпотентны:
- скачать (`download_index`): индекс в Qdrant уже подходит к настройкам —
  ничего; в релизе нет снимка для этих настроек — стоп, нужна сборка; файл с
  верным sha256 уже в index/ — не качать; иначе — скачать во временный .part,
  сверить sha256, восстановить коллекцию и переключить на неё алиас;
- выложить (`upload_index`): индексы выборок не выкладываются (решение
  автора), в релизе уже эта коллекция — ничего; иначе снимок снимается с
  Qdrant (если его ещё нет в index/), рядом пишется <ключ>.json, оба файла
  уходят в релиз через gh (`gh release upload --clobber`).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx2

from assistant.indexing.dataset import file_sha256
from assistant.retrieval.store import IndexManifest, VectorStore, manifest_reason

REPOSITORY = "MichaelZhdanov/recipes-llm"
RELEASE = "index"
DIRECTORY = Path("index")
# Деплой с пересборкой индекса пишет сюда ключ выгруженного снимка: раннер CI
# забирает снимок по scp и выкладывает сам (токен записи на сервер не попадает).
EXPORTED = "exported"
# Прогресс скачивания — каждые столько процентов.
PROGRESS_STEP = 5
CHUNK = 8 << 20

Say = Callable[[str], None]
Run = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class SnapshotError(RuntimeError):
    """Снимок не скачался, не сошёлся sha256 или Qdrant не принял его."""


@dataclass(frozen=True)
class SnapshotInfo:
    """<ключ>.json в релизе: какая коллекция в снимке и как его проверить."""

    key: str
    collection: str
    sha256: str
    size: int
    manifest: IndexManifest

    def to_json(self) -> str:
        data = {
            "key": self.key,
            "collection": self.collection,
            "sha256": self.sha256,
            "size": self.size,
            "manifest": self.manifest.model_dump(),
        }
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> SnapshotInfo:
        data = json.loads(text)
        return cls(
            key=data["key"],
            collection=data["collection"],
            sha256=data["sha256"],
            size=int(data["size"]),
            manifest=IndexManifest.model_validate(data["manifest"]),
        )


def snapshot_key(ref: str, model: str, sample: int | None, seed: int | None) -> str:
    """Ключ снимка по паспорту: ffaf49ee-giga-embeddings-instruct-3b-full."""
    name = re.sub(r"[^a-z0-9]+", "-", model.rsplit("/", 1)[-1].lower()).strip("-")
    size = "full" if sample is None else f"sample{sample}-seed{seed}"
    return f"{ref[:8]}-{name}-{size}"


def asset_url(name: str) -> str:
    return f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}/{name}"


def fetch_info(key: str, http: httpx2.Client) -> SnapshotInfo | None:
    """<ключ>.json из релиза; None — снимка для этих настроек там нет."""
    response = http.get(asset_url(f"{key}.json"), follow_redirects=True)
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise SnapshotError(
            f"релиз {RELEASE}: {key}.json — HTTP {response.status_code}"
        )
    return SnapshotInfo.from_json(response.text)


def ensure_file(
    info: SnapshotInfo, http: httpx2.Client, directory: Path, say: Say
) -> Path:
    """Файл снимка в index/: уже лежит с верным sha256 — не качается заново.

    Скачивается во временный .part и переименовывается, только когда размер и
    sha256 сошлись: оборванная загрузка не выглядит готовым файлом.
    """
    target = directory / f"{info.collection}.snapshot"
    if target.exists() and target.stat().st_size == info.size:
        say(f"снимок уже скачан: {target} — проверяю sha256")
        if file_sha256(target) == info.sha256:
            return target
        say("sha256 не тот — качаю заново")
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    url = asset_url(f"{info.key}.snapshot")
    say(f"скачиваю {url} ({info.size / 1e9:.2f} ГБ)")
    digest, done, shown = hashlib.sha256(), 0, 0
    try:
        with (
            http.stream("GET", url, follow_redirects=True) as response,
            partial.open("wb") as file,
        ):
            if response.status_code != 200:
                raise SnapshotError(f"{url}: HTTP {response.status_code}")
            for block in response.iter_bytes(CHUNK):
                file.write(block)
                digest.update(block)
                done += len(block)
                percent = done * 100 // max(info.size, 1)
                if percent >= shown + PROGRESS_STEP:
                    shown = percent - percent % PROGRESS_STEP
                    say(f"  {shown}% — {done / 1e9:.2f} из {info.size / 1e9:.2f} ГБ")
        if done != info.size or digest.hexdigest() != info.sha256:
            raise SnapshotError(
                f"скачано {done} байт, sha256 {digest.hexdigest()[:12]}…, а в релизе — "
                f"{info.size} байт и {info.sha256[:12]}…: файл не целиком или не тот"
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def restore(
    path: Path, info: SnapshotInfo, qdrant_url: str, http: httpx2.Client, say: Say
) -> None:
    """Коллекция из файла снимка: Qdrant сам сверяет sha256 (checksum)."""
    url = f"{qdrant_url.rstrip('/')}/collections/{info.collection}/snapshots/upload"
    say(f"восстанавливаю коллекцию {info.collection} из {path}")
    with path.open("rb") as file:
        response = http.post(
            url,
            params={"priority": "snapshot", "wait": "true", "checksum": info.sha256},
            files={"snapshot": (path.name, file, "application/octet-stream")},
            timeout=None,
        )
    if response.status_code != 200:
        raise SnapshotError(
            f"Qdrant не восстановил {info.collection}: HTTP {response.status_code} "
            f"{response.text[:200]}"
        )


def download_index(
    store: VectorStore,
    *,
    ref: str,
    model: str,
    sample: int | None,
    seed: int,
    http: httpx2.Client,
    qdrant_url: str,
    directory: Path = DIRECTORY,
    say: Say,
) -> bool:
    """Индекс из релиза. False — в релизе нет снимка для этих настроек."""
    key = snapshot_key(ref, model, sample, seed if sample is not None else None)
    info = fetch_info(key, http)
    if info is None:
        say(f"в релизе {RELEASE} нет снимка {key}")
        return False
    if why := manifest_reason(
        info.manifest, ref=ref, model=model, sample=sample, seed=seed
    ):
        say(f"снимок {key} в релизе не подходит к настройкам: {why}")
        return False
    path = ensure_file(info, http, directory, say)
    restore(path, info, qdrant_url, http, say)
    store.activate(info.collection)
    return True


def export_snapshot(
    store: VectorStore,
    *,
    qdrant_url: str,
    http: httpx2.Client,
    directory: Path = DIRECTORY,
    say: Say,
) -> SnapshotInfo:
    """Снимок активной коллекции в index/ и <ключ>.json рядом.

    Файл уже есть (сняли раньше или скачали) — Qdrant не трогается. Иначе Qdrant
    пишет снимок у себя, он скачивается в index/ и удаляется в Qdrant.
    """
    manifest = store.manifest()
    collection = store.active()
    assert collection is not None  # manifest() без коллекции уже упал бы
    key = snapshot_key(
        manifest.ref, manifest.embedding_model, manifest.sample, manifest.seed
    )
    path = directory / f"{collection}.snapshot"
    if not path.exists():
        base = f"{qdrant_url.rstrip('/')}/collections/{collection}/snapshots"
        say(f"снимаю снимок {collection} в Qdrant")
        created = http.post(base, params={"wait": "true"}, timeout=None)
        if created.status_code != 200:
            raise SnapshotError(f"Qdrant не снял снимок: HTTP {created.status_code}")
        name = created.json()["result"]["name"]
        directory.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        try:
            with (
                http.stream("GET", f"{base}/{name}", timeout=None) as response,
                partial.open("wb") as file,
            ):
                if response.status_code != 200:
                    raise SnapshotError(f"снимок {name}: HTTP {response.status_code}")
                for block in response.iter_bytes(CHUNK):
                    file.write(block)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
            http.delete(f"{base}/{name}", params={"wait": "true"})
    say(f"снимок {path} — считаю sha256")
    info = SnapshotInfo(
        key, collection, file_sha256(path), path.stat().st_size, manifest
    )
    (directory / f"{key}.json").write_text(info.to_json(), encoding="utf-8")
    return info


def publish(info: SnapshotInfo, directory: Path, run: Run, say: Say) -> None:
    """Оба файла — в релиз через gh: сначала снимок, потом json (json ссылается
    на снимок). Файл снимка в релизе зовётся по ключу: ссылка, а не копия."""
    if shutil.which("gh") is None:
        raise SnapshotError("нет gh (GitHub CLI): выложить снимок может только автор")
    snapshot = directory / f"{info.collection}.snapshot"
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        named = Path(tmp) / f"{info.key}.snapshot"
        os.link(snapshot, named)
        meta = Path(tmp) / f"{info.key}.json"
        shutil.copyfile(directory / f"{info.key}.json", meta)
        for asset in (named, meta):
            say(
                f"выкладываю {asset.name} ({asset.stat().st_size / 1e9:.2f} ГБ) в релиз {RELEASE}"
            )
            result = run(
                [
                    "gh",
                    "release",
                    "upload",
                    RELEASE,
                    str(asset),
                    "--repo",
                    REPOSITORY,
                    "--clobber",
                ]
            )
            if result.returncode != 0:
                raise SnapshotError(
                    f"gh release upload: {result.stderr.strip()[-300:] or result.returncode}"
                )


def upload_index(
    store: VectorStore,
    *,
    qdrant_url: str,
    http: httpx2.Client,
    run: Run,
    directory: Path = DIRECTORY,
    publish_now: bool = True,
    say: Say,
) -> str:
    """Активный индекс — в релиз. Что сделано — строкой для человека.

    publish_now=False (деплой на сервере): только снять снимок и записать ключ в
    index/exported — выложит раннер CI.
    """
    manifest = store.manifest()
    if manifest.sample is not None:
        return "индекс — выборка: в релиз выкладываются только индексы всего датасета"
    collection = store.active()
    key = snapshot_key(manifest.ref, manifest.embedding_model, None, None)
    published = fetch_info(key, http)
    if published is not None and published.collection == collection:
        return f"в релизе уже этот индекс ({collection}) — выкладывать нечего"
    info = export_snapshot(
        store, qdrant_url=qdrant_url, http=http, directory=directory, say=say
    )
    if not publish_now:
        (directory / EXPORTED).write_text(info.key + "\n", encoding="utf-8")
        return f"снимок готов к выкладке: {directory}/{info.collection}.snapshot, ключ {info.key}"
    publish(info, directory, run, say)
    return f"в релизе {RELEASE}: {info.key} — коллекция {info.collection}"
