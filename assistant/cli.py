"""Командная строка: `python -m assistant.cli ask "вопрос" | search "вопрос"`.

Запускается на хосте через `uv run --env-file .env` (см. Makefile). Ход
работы пишется в stderr, результат — в stdout. Команда `index` вернётся
на этапе 1 вместе с загрузчиком рецептов.
"""

import argparse
import json
import sys
from collections.abc import Sequence

import httpx2
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from assistant.config import Settings
from assistant.generation.answer import make_answerer, render_markdown
from assistant.generation.gigachat import LlmError, make_http_client
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
    LlmError,
    IndexMissingError,
    IndexMismatchError,
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def format_hits(hits: Sequence[Hit]) -> str:
    """Человекочитаемая выдача: что нашлось, где лежит и как открыть."""
    if not hits:
        return "ничего не найдено"
    blocks = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        blocks.append(
            f"{number}. [{hit.score:.3f}] {chunk.name}\n"
            f"   {chunk.url}\n"
            f"   ингредиенты: {', '.join(chunk.ingredients)}"
        )
    return "\n\n".join(blocks)


def _search(settings: Settings, question: str, k: int, as_json: bool) -> int:
    with httpx2.Client(timeout=30) as http:
        embedder = make_embedder(settings, http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        hits = Retriever(store, embedder, ref=settings.dataset_revision).search(
            question, k=k
        )
    if as_json:
        payload = [{"score": hit.score, **hit.chunk.model_dump()} for hit in hits]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_hits(hits))
    return 0


def _ask(settings: Settings, question: str, as_json: bool) -> int:
    with (
        httpx2.Client(timeout=30) as embeddings_http,
        make_http_client() as llm_http,
    ):
        answerer = make_answerer(
            settings,
            embeddings_http=embeddings_http,
            llm_http=llm_http,
            qdrant=QdrantClient(url=settings.qdrant_url, timeout=30),
        )
        answer = answerer.answer(question).answer
    print(answer.model_dump_json(indent=2) if as_json else render_markdown(answer))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m assistant.cli", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="ответ со ссылками на документацию и код")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="вывод в JSON")
    search = commands.add_parser(
        "search", help="найти куски документации и кода по вопросу, без LLM"
    )
    search.add_argument("question")
    search.add_argument("-k", type=int, default=5, help="сколько источников показать")
    search.add_argument("--json", action="store_true", help="вывод в JSON")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        if args.command == "ask":
            return _ask(settings, args.question, args.json)
        if args.command == "search":
            return _search(settings, args.question, args.k, args.json)
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
