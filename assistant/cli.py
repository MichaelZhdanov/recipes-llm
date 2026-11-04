"""Командная строка: `python -m assistant.cli index` и `… ask "вопрос"`.

Запускается на хосте через `uv run --env-file .env` (см. Makefile). Ход
работы пишется в stderr, результат — в stdout.
"""

import argparse
import json
import sys
from collections.abc import Sequence

import httpx2
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from assistant.config import Settings
from assistant.indexing.build import build_index
from assistant.indexing.sources import SourceDownloadError
from assistant.retrieval.embeddings import EmbeddingError, make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import (
    Hit,
    IndexMismatchError,
    IndexMissingError,
    VectorStore,
)

# Ошибки, о которых пользователю достаточно одной строки без трассировки.
_EXPECTED_ERRORS = (
    EmbeddingError,
    SourceDownloadError,
    IndexMissingError,
    IndexMismatchError,
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _index(settings: Settings) -> int:
    with httpx2.Client(timeout=60) as http:
        # Эмбеддер первым: без ключа ошибка сразу, ещё до сетевых вызовов.
        embedder = make_embedder(settings, http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=60))
        report = build_index(
            settings, http=http, store=store, embedder=embedder, log=_log
        )
    manifest = report.manifest
    cost = (
        f"${report.cost_usd:.4f}"
        if report.cost_usd is not None
        else "цена модели неизвестна"
    )
    masked = ", ".join(f"{k} {v}" for k, v in sorted(report.masked.items())) or "нет"
    print(
        f"Индекс построен: {manifest.ref} ({manifest.commit}), модель {manifest.embedding_model}, "
        f"размерность {manifest.dim}\n"
        f"кусков: {sum(manifest.chunks.values())} {manifest.chunks}\n"
        f"токенов: {report.tokens}, оценка стоимости: {cost}\n"
        f"маскер сработал: {masked}\n"
        f"время: {report.seconds:.0f} с"
    )
    return 0


def format_hits(hits: Sequence[Hit]) -> str:
    """Человекочитаемая выдача: что нашлось, где лежит и как открыть."""
    if not hits:
        return "ничего не найдено"
    blocks = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        preview = " ".join(chunk.text.split())[:200]
        lines = [
            f"{number}. [{hit.score:.3f}] {chunk.kind}: {chunk.breadcrumb}",
            f"   {chunk.path}:{chunk.start_line}-{chunk.end_line}",
            f"   {chunk.url}",
        ]
        if chunk.site_url:
            lines.append(f"   сайт: {chunk.site_url}")
        lines.append(f"   «{preview}»")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _ask(settings: Settings, question: str, k: int, as_json: bool) -> int:
    with httpx2.Client(timeout=30) as http:
        embedder = make_embedder(settings, http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        hits = Retriever(store, embedder, ref=settings.paperless_ref).search(
            question, k=k
        )
    if as_json:
        payload = [{"score": hit.score, **hit.chunk.model_dump()} for hit in hits]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_hits(hits))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m assistant.cli", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="пересобрать индекс по PAPERLESS_REF")
    ask = commands.add_parser("ask", help="найти куски документации и кода по вопросу")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5, help="сколько источников показать")
    ask.add_argument("--json", action="store_true", help="вывод в JSON")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        if args.command == "index":
            return _index(settings)
        if args.command == "ask":
            return _ask(settings, args.question, args.k, args.json)
    except _EXPECTED_ERRORS as error:
        _log(f"ошибка: {error}")
        return 1
    except ResponseHandlingException as error:
        _log(
            f"ошибка: Qdrant недоступен по {settings.qdrant_url} ({error}). "
            "Запустите его: docker compose up -d --wait qdrant"
        )
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
