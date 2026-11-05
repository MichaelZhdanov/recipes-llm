"""Векторное хранилище: коллекция Qdrant с рецептами и паспортом индекса.

В метаданных коллекции лежит паспорт: версия данных, модель эмбеддингов,
размерность. Поиск сверяет паспорт с настройками. Если версия данных или
модель поменялись, а индекс старый, будет понятная ошибка, а не тихие
ответы по чужой версии.

**Алиас вместо одной коллекции.** Поиск ходит в алиас `recipes`, а каждая
сборка индекса пишет в свою новую коллекцию `recipes-<id>`. Когда все точки
на месте и проверены, алиас одной операцией переключается на новую
коллекцию, а старая удаляется. Если сборка упала посередине, удаляется
только недостроенная коллекция: поиск всё это время работает по старой.

**Векторы на диске, в памяти — сжатые копии.** На сервере 2 ГБ RAM, а
полный индекс в float32 — 0,6–1,2 ГБ. Поэтому точные векторы лежат на
диске (`on_disk`), а в памяти держится их копия в int8 (1 байт на число
вместо 4). Поиск идёт по сжатым векторам, а лучшие кандидаты
переранжируются по точным (`rescore`, с запасом `oversampling`).
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from assistant.indexing.chunks import Chunk
from assistant.retrieval.filters import FIELD, IngredientFilter

ALIAS = "recipes"
# Пространство имён для uuid5: id точки детерминирован, повторная сборка даёт те же id.
_POINT_NAMESPACE = uuid.UUID("5b0c6c4e-8a35-4f3e-9d43-3c2f6f1f2a10")
# Сколько кандидатов взять по сжатым векторам на одно место в выдаче.
OVERSAMPLING = 2.0
UPSERT_BATCH = 256
# Сколько разных имён ингредиентов читать из индекса; в полном датасете их 1 097.
VOCABULARY_LIMIT = 5000


class IndexMissingError(RuntimeError):
    pass


class IndexMismatchError(RuntimeError):
    pass


class IndexManifest(BaseModel):
    """Паспорт индекса: из каких данных и какой моделью он построен."""

    dataset: str
    # Ревизия датасета: по ней поиск сверяет индекс с настройками.
    ref: str
    file_sha256: str
    recipes: int
    # Сколько строк датасета пропущено и почему.
    skipped: dict[str, int]
    # Размер выборки и её сид; None — весь датасет.
    sample: int | None
    seed: int | None
    embedding_model: str
    dim: int
    # Сжатие векторов в памяти: int8 или None — без сжатия.
    quantization: Literal["int8"] | None
    built_at: str
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
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{chunk.ref}:{chunk.recipe_id}"))


def _quantization(manifest: IndexManifest) -> models.ScalarQuantization | None:
    if manifest.quantization is None:
        return None
    return models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            # Крайний 1 % значений не растягивает шкалу int8.
            quantile=0.99,
            always_ram=True,
        )
    )


class VectorStore:
    def __init__(
        self, client: QdrantClient, alias: str = ALIAS, *, quantization: bool = True
    ) -> None:
        self._client = client
        self._alias = alias
        # False — искать по точным векторам, не трогая сжатые: замер цены сжатия
        # на том же индексе, без пересборки.
        self._quantization = quantization

    def active(self) -> str | None:
        """Коллекция, на которую сейчас указывает алиас, или None."""
        for item in self._client.get_aliases().aliases:
            if item.alias_name == self._alias:
                return item.collection_name
        return None

    def create(self, manifest: IndexManifest) -> str:
        """Новая пустая коллекция с паспортом. Поиск её не видит до activate()."""
        name = f"{self._alias}-{uuid.uuid4().hex[:12]}"
        self._client.create_collection(
            name,
            # Векторы моделей нормированы, поэтому косинус и скалярное произведение совпадают.
            vectors_config=models.VectorParams(
                size=manifest.dim, distance=models.Distance.COSINE, on_disk=True
            ),
            quantization_config=_quantization(manifest),
            metadata=manifest.model_dump(),
        )
        # Фильтр «с чем и без чего» (этап 2) идёт по именам ингредиентов.
        self._client.create_payload_index(
            name, FIELD, field_schema=models.PayloadSchemaType.KEYWORD
        )
        return name

    def activate(self, name: str) -> None:
        """Переключает алиас на коллекцию name одной операцией и удаляет прежнюю."""
        previous = self.active()
        operations: list[models.AliasOperations] = []
        if previous is not None:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self._alias)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=name, alias_name=self._alias
                )
            )
        )
        self._client.update_collection_aliases(change_aliases_operations=operations)
        if previous is not None and previous != name:
            self._client.delete_collection(previous)

    def drop(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    def recreate(self, manifest: IndexManifest) -> str:
        """Пустой индекс с паспортом сразу под алиасом: для маленьких индексов и тестов."""
        name = self.create(manifest)
        self.activate(name)
        return name

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
        *,
        collection: str | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} рецептов, но {len(vectors)} векторов")
        target = collection or self._alias
        for start in range(0, len(chunks), UPSERT_BATCH):
            points = [
                models.PointStruct(
                    id=point_id(chunk), vector=vector, payload=chunk.model_dump()
                )
                for chunk, vector in zip(
                    chunks[start : start + UPSERT_BATCH],
                    vectors[start : start + UPSERT_BATCH],
                    strict=True,
                )
            ]
            self._client.upsert(target, points=points, wait=True)

    def manifest(self) -> IndexManifest:
        name = self.active()
        if name is None:
            raise IndexMissingError("индекс не построен: выполните make index")
        metadata = self._client.get_collection(name).config.metadata
        if not metadata:
            raise IndexMissingError("у коллекции нет паспорта: выполните make index")
        return IndexManifest.model_validate(metadata)

    def vocabulary(self) -> dict[str, int]:
        """Имена ингредиентов в индексе и в скольких рецептах каждое, от частых
        к редким. Считает Qdrant по payload-индексу (facet), поэтому словарь
        всегда совпадает с тем, что лежит в индексе, в том числе у выборки."""
        if self.active() is None:
            raise IndexMissingError("индекс не построен: выполните make index")
        response = self._client.facet(
            self._alias, key=FIELD, limit=VOCABULARY_LIMIT, exact=True
        )
        if len(response.hits) >= VOCABULARY_LIMIT:
            raise RuntimeError(f"в индексе больше {VOCABULARY_LIMIT} имён ингредиентов")
        return {str(hit.value): hit.count for hit in response.hits}

    def count(self, collection: str | None = None) -> int:
        return self._client.count(collection or self._alias, exact=True).count

    def verify(self, collection: str | None = None) -> int:
        """Читает все точки и проверяет, что payload — полный рецепт. Возвращает число точек."""
        total = 0
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection or self._alias,
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

    def search_params(self) -> models.SearchParams:
        # Без сжатия в коллекции Qdrant эти параметры просто не использует.
        return models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=not self._quantization, rescore=True, oversampling=OVERSAMPLING
            )
        )

    def search(
        self,
        vector: list[float],
        limit: int,
        conditions: IngredientFilter | None = None,
    ) -> list[Hit]:
        """Ближайшие рецепты среди тех, что прошли условия (фильтр Qdrant
        отбрасывает остальные до ранжирования векторами)."""
        response = self._client.query_points(
            self._alias,
            query=vector,
            query_filter=conditions.to_qdrant() if conditions else None,
            limit=limit,
            with_payload=True,
            search_params=self.search_params(),
        )
        return [
            Hit(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]
