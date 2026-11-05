"""Векторное хранилище: коллекция Qdrant с кусками и паспортом индекса.

В метаданных коллекции лежит паспорт: версия данных, модель эмбеддингов,
размерность. Поиск сверяет паспорт с настройками. Если версия данных или
модель поменялись, а индекс старый, будет понятная ошибка, а не тихие
ответы по чужой версии.
"""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from assistant.indexing.chunks import Chunk

COLLECTION = "recipes"
# Пространство имён для uuid5: id точки детерминирован, повторная сборка даёт те же id.
_POINT_NAMESPACE = uuid.UUID("5b0c6c4e-8a35-4f3e-9d43-3c2f6f1f2a10")


class IndexMissingError(RuntimeError):
    pass


class IndexMismatchError(RuntimeError):
    pass


class IndexManifest(BaseModel):
    ref: str
    commit: str | None
    embedding_model: str
    dim: int
    built_at: str
    chunks: dict[str, int]
    assistant_version: str


def mismatch(manifest: IndexManifest, *, ref: str, model: str) -> str | None:
    """Почему индекс не подходит к настройкам, или None, если подходит."""
    if manifest.ref == ref and manifest.embedding_model == model:
        return None
    return (
        f"индекс построен для {manifest.ref} и модели {manifest.embedding_model}, "
        f"а в настройках {ref} и {model}"
    )


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def point_id(chunk: Chunk) -> str:
    return str(
        uuid.uuid5(
            _POINT_NAMESPACE,
            f"{chunk.ref}:{chunk.path}:{chunk.start_line}-{chunk.end_line}",
        )
    )


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION) -> None:
        self._client = client
        self._collection = collection

    def recreate(self, manifest: IndexManifest) -> None:
        """Удаляет старую коллекцию и создаёт пустую с паспортом."""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            # Векторы Voyage нормированы, поэтому косинус и скалярное произведение совпадают.
            vectors_config=models.VectorParams(
                size=manifest.dim, distance=models.Distance.COSINE
            ),
            metadata=manifest.model_dump(),
        )
        for field in ("kind", "source_id"):
            self._client.create_payload_index(
                self._collection, field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[list[float]], batch: int = 256
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} кусков, но {len(vectors)} векторов")
        for start in range(0, len(chunks), batch):
            points = [
                models.PointStruct(
                    id=point_id(chunk), vector=vector, payload=chunk.model_dump()
                )
                for chunk, vector in zip(
                    chunks[start : start + batch],
                    vectors[start : start + batch],
                    strict=True,
                )
            ]
            self._client.upsert(self._collection, points=points, wait=True)

    def manifest(self) -> IndexManifest:
        if not self._client.collection_exists(self._collection):
            raise IndexMissingError("индекс не построен: выполните make index")
        metadata = self._client.get_collection(self._collection).config.metadata
        if not metadata:
            raise IndexMissingError("у коллекции нет паспорта: выполните make index")
        return IndexManifest.model_validate(metadata)

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def has_source(self, source_id: str) -> bool:
        source_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id", match=models.MatchValue(value=source_id)
                )
            ]
        )
        return (
            self._client.count(
                self._collection, count_filter=source_filter, exact=True
            ).count
            > 0
        )

    def verify(self) -> int:
        """Читает все точки и проверяет, что payload — полный кусок. Возвращает число точек."""
        total = 0
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                Chunk.model_validate(point.payload)
            total += len(points)
            if offset is None:
                return total

    def search(
        self, vector: list[float], limit: int, kinds: Collection[str] | None = None
    ) -> list[Hit]:
        query_filter = None
        if kinds:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="kind", match=models.MatchAny(any=list(kinds))
                    )
                ]
            )
        response = self._client.query_points(
            self._collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            Hit(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]
