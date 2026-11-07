"""Снимок индекса в релизе GitHub (assistant/indexing/snapshot.py): идемпотентно
скачать и выложить. Сеть и gh подменены: GitHub, Qdrant — MockTransport."""

import hashlib
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import httpx2
import pytest
from qdrant_client import QdrantClient, models

from assistant.cli import INDEX_OUTDATED, index_download
from assistant.config import Settings
from assistant.indexing.snapshot import (
    EXPORTED,
    SnapshotError,
    SnapshotInfo,
    asset_url,
    download_index,
    ensure_file,
    fetch_info,
    restore,
    snapshot_key,
    upload_index,
)
from assistant.retrieval.store import IndexManifest, VectorStore

MANIFEST = IndexManifest(
    dataset="rogozinushka/povarenok-recipes",
    ref="ffaf49ee7439afe6dbf0653714ceacb9b57ff830",
    file_sha256="0" * 64,
    recipes=3,
    skipped={},
    sample=None,
    seed=None,
    embedding_model="ai-sage/Giga-Embeddings-instruct-3B",
    dim=4,
    quantization=None,
    built_at="2026-11-06T06:47:31+00:00",
    assistant_version="0.1.0",
)
KEY = "ffaf49ee-giga-embeddings-instruct-3b-full"
DATA = b"qdrant snapshot bytes" * 1000
QDRANT = "http://qdrant.test:6333"


def info(collection: str = "recipes-abc", data: bytes = DATA) -> SnapshotInfo:
    return SnapshotInfo(
        KEY, collection, hashlib.sha256(data).hexdigest(), len(data), MANIFEST
    )


class Fake:
    """GitHub (релиз) и Qdrant: что отдавать и какие запросы пришли."""

    def __init__(self, release: dict[str, bytes] | None = None) -> None:
        self.release = release or {}
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        url = str(request.url)
        for name, body in self.release.items():
            if url == asset_url(name):
                return httpx2.Response(200, content=body)
        if url.startswith(QDRANT):
            if request.method == "POST" and url.split("?")[0].endswith("/snapshots"):
                return httpx2.Response(200, json={"result": {"name": "made.snapshot"}})
            if request.method == "GET":
                return httpx2.Response(200, content=DATA)
            return httpx2.Response(200, json={"result": True})
        return httpx2.Response(404)

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self))


def store_with(collection: str, manifest: IndexManifest = MANIFEST) -> VectorStore:
    """In-memory Qdrant с коллекцией и паспортом; алиас — на неё."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        metadata=manifest.model_dump(),
    )
    store = VectorStore(client)
    store.activate(collection)
    return store


def test_key_comes_from_the_index_passport() -> None:
    assert snapshot_key(MANIFEST.ref, MANIFEST.embedding_model, None, None) == KEY
    assert (
        snapshot_key("abc12345xyz", "BAAI/bge-m3", 10000, 42)
        == "abc12345-bge-m3-sample10000-seed42"
    )


def test_info_round_trips_through_json() -> None:
    assert SnapshotInfo.from_json(info().to_json()) == info()


def test_missing_snapshot_in_the_release_is_none() -> None:
    assert fetch_info(KEY, Fake().client()) is None
    with pytest.raises(SnapshotError, match="HTTP 500"):
        fetch_info(
            KEY,
            httpx2.Client(
                transport=httpx2.MockTransport(lambda r: httpx2.Response(500))
            ),
        )


def test_file_with_the_right_sha256_is_not_downloaded_again(tmp_path: Path) -> None:
    (tmp_path / "recipes-abc.snapshot").write_bytes(DATA)
    fake = Fake()

    path = ensure_file(info(), fake.client(), tmp_path, say=lambda _: None)

    assert path == tmp_path / "recipes-abc.snapshot"
    assert fake.requests == []


def test_download_checks_sha256_before_the_file_appears(tmp_path: Path) -> None:
    good = Fake({f"{KEY}.snapshot": DATA})
    said: list[str] = []

    path = ensure_file(info(), good.client(), tmp_path, said.append)

    assert path.read_bytes() == DATA
    assert any("100%" in line for line in said)

    broken = Fake({f"{KEY}.snapshot": DATA[:-1] + b"!"})
    with pytest.raises(SnapshotError, match="не целиком или не тот"):
        ensure_file(
            info(collection="recipes-new"), broken.client(), tmp_path, said.append
        )
    assert not (tmp_path / "recipes-new.snapshot").exists()
    assert not list(tmp_path.glob("*.part"))


def test_restore_uploads_the_file_to_qdrant_with_its_checksum(tmp_path: Path) -> None:
    file = tmp_path / "recipes-abc.snapshot"
    file.write_bytes(DATA)
    fake = Fake()

    restore(file, info(), QDRANT, fake.client(), say=lambda _: None)

    (request,) = fake.requests
    assert request.url.path == "/collections/recipes-abc/snapshots/upload"
    assert request.url.params["priority"] == "snapshot"
    assert request.url.params["checksum"] == info().sha256
    assert b'name="snapshot"' in request.read()


def test_download_restores_and_switches_the_alias(tmp_path: Path) -> None:
    store = store_with("recipes-old")
    # Коллекция «как после восстановления»: Qdrant здесь подменён.
    store._client.create_collection(
        "recipes-abc",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        metadata=MANIFEST.model_dump(),
    )
    fake = Fake({f"{KEY}.json": info().to_json().encode(), f"{KEY}.snapshot": DATA})

    ok = download_index(
        store,
        ref=MANIFEST.ref,
        model=MANIFEST.embedding_model,
        sample=None,
        seed=1,
        http=fake.client(),
        qdrant_url=QDRANT,
        directory=tmp_path,
        say=lambda _: None,
    )

    assert ok
    assert store.active() == "recipes-abc"
    assert not store._client.collection_exists("recipes-old")


def test_no_snapshot_for_these_settings_means_a_build_is_needed(tmp_path: Path) -> None:
    store = VectorStore(QdrantClient(":memory:"))
    fake = Fake()

    ok = download_index(
        store,
        ref=MANIFEST.ref,
        model=MANIFEST.embedding_model,
        sample=10000,
        seed=42,
        http=fake.client(),
        qdrant_url=QDRANT,
        directory=tmp_path,
        say=lambda _: None,
    )

    assert not ok
    assert [str(r.url) for r in fake.requests] == [
        asset_url("ffaf49ee-giga-embeddings-instruct-3b-sample10000-seed42.json")
    ]


def test_first_download_on_a_machine_with_the_index_does_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Индекс в Qdrant уже подходит к настройкам — ни одного запроса в сеть."""
    store = store_with("recipes-abc")
    settings = Settings(
        dataset_revision=MANIFEST.ref, embedding_model=MANIFEST.embedding_model
    )
    fake = Fake()

    code = index_download(store, settings, sample=None, seed=1, http=fake.client())

    assert code == 0
    assert fake.requests == []
    assert "скачивать нечего" in capsys.readouterr().out


def test_index_download_says_a_build_is_needed_with_code_3() -> None:
    settings = Settings(dataset_revision=MANIFEST.ref, embedding_model="other/model")

    code = index_download(
        VectorStore(QdrantClient(":memory:")),
        settings,
        sample=None,
        seed=1,
        http=Fake().client(),
    )

    assert code == INDEX_OUTDATED


class Gh:
    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        return subprocess.CompletedProcess(list(args), self.code, "", "нет прав")


def test_upload_skips_samples_and_what_is_already_released(tmp_path: Path) -> None:
    gh = Gh()
    sample = store_with(
        "recipes-s", MANIFEST.model_copy(update={"sample": 10, "seed": 1})
    )

    said = upload_index(
        sample,
        qdrant_url=QDRANT,
        http=Fake().client(),
        run=gh,
        directory=tmp_path,
        say=print,
    )
    assert "только индексы всего датасета" in said

    released = Fake({f"{KEY}.json": info().to_json().encode()})
    said = upload_index(
        store_with("recipes-abc"),
        qdrant_url=QDRANT,
        http=released.client(),
        run=gh,
        directory=tmp_path,
        say=print,
    )
    assert "уже этот индекс" in said
    assert gh.calls == []
    assert list(tmp_path.iterdir()) == []


def test_upload_takes_a_snapshot_and_publishes_snapshot_then_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    gh = Gh()
    qdrant = Fake()

    said = upload_index(
        store_with("recipes-new"),
        qdrant_url=QDRANT,
        http=qdrant.client(),
        run=gh,
        directory=tmp_path,
        say=lambda _: None,
    )

    # Qdrant: снять снимок, скачать его, удалить у себя.
    methods = [
        (r.method, r.url.path) for r in qdrant.requests if r.url.host == "qdrant.test"
    ]
    assert methods == [
        ("POST", "/collections/recipes-new/snapshots"),
        ("GET", "/collections/recipes-new/snapshots/made.snapshot"),
        ("DELETE", "/collections/recipes-new/snapshots/made.snapshot"),
    ]
    assert (tmp_path / "recipes-new.snapshot").read_bytes() == DATA
    written = SnapshotInfo.from_json((tmp_path / f"{KEY}.json").read_text())
    assert (written.collection, written.sha256) == (
        "recipes-new",
        hashlib.sha256(DATA).hexdigest(),
    )
    assert [Path(call[4]).name for call in gh.calls] == [
        f"{KEY}.snapshot",
        f"{KEY}.json",
    ]
    assert all("--clobber" in call for call in gh.calls)
    assert "recipes-new" in said


def test_export_for_ci_writes_the_key_and_publishes_nothing(tmp_path: Path) -> None:
    (tmp_path / "recipes-new.snapshot").write_bytes(DATA)
    gh = Gh()

    upload_index(
        store_with("recipes-new"),
        qdrant_url=QDRANT,
        http=Fake().client(),
        run=gh,
        directory=tmp_path,
        publish_now=False,
        say=lambda _: None,
    )

    assert (tmp_path / EXPORTED).read_text() == f"{KEY}\n"
    assert (tmp_path / f"{KEY}.json").exists()
    assert gh.calls == []


def test_failed_gh_upload_is_an_error_with_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    (tmp_path / "recipes-new.snapshot").write_bytes(DATA)

    with pytest.raises(SnapshotError, match="нет прав"):
        upload_index(
            store_with("recipes-new"),
            qdrant_url=QDRANT,
            http=Fake().client(),
            run=Gh(code=1),
            directory=tmp_path,
            say=lambda _: None,
        )
