"""Командная строка: `python -m assistant.cli data | index … | ask "вопрос" | search "вопрос"`.

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
from assistant.generation.answer import make_answerer, render_markdown
from assistant.generation.gigachat import LlmError
from assistant.generation.llm import llm_model, make_chat_model, make_http_client
from assistant.indexing.build import (
    SAMPLE_SEED,
    IndexSource,
    build_index,
    estimate,
    pick_sample,
)
from assistant.indexing.dataset import (
    DatasetError,
    ensure_dataset,
    known_sha256,
    read_dataset,
)
from assistant.retrieval.embeddings import EmbeddingError, make_embedder
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import (
    Hit,
    IndexMismatchError,
    IndexMissingError,
    VectorStore,
    mismatch,
)
from assistant.tools.search_recipes import SearchArgs, SearchPlanner

# Ошибки, о которых пользователю достаточно одной строки без трассировки.
_EXPECTED_ERRORS = (
    DatasetError,
    EmbeddingError,
    LlmError,
    IndexMissingError,
    IndexMismatchError,
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def rebuild_reason(
    store: VectorStore,
    *,
    ref: str,
    model: str,
    sample: int | None = None,
    seed: int = SAMPLE_SEED,
) -> str | None:
    """Зачем строить индекс заново, или None, если он уже подходит к настройкам:
    та же ревизия, модель и размер выборки (у выборки — и сид)."""
    try:
        manifest = store.manifest()
    except IndexMissingError as error:
        return str(error)
    if reason := mismatch(manifest, ref=ref, model=model):
        return reason
    if manifest.sample != sample or (sample is not None and manifest.seed != seed):
        have = (
            f"выборке из {manifest.sample} (сид {manifest.seed})"
            if manifest.sample
            else "всём датасете"
        )
        want = (
            f"нужна выборка из {sample} (сид {seed})"
            if sample
            else "нужен весь датасет"
        )
        return f"индекс построен на {have}, а {want}"
    return None


def _data(settings: Settings) -> int:
    with httpx2.Client() as http:
        path = ensure_dataset(
            settings.dataset_revision,
            settings.data_path,
            http,
            sha256=known_sha256(settings.dataset_revision),
        )
    print(
        f"датасет {settings.dataset_revision[:8]}: {path} ({path.stat().st_size} байт)"
    )
    return 0


def _index(
    settings: Settings,
    *,
    if_needed: bool,
    sample: int | None,
    seed: int,
    yes: bool,
    quantization: bool,
) -> int:
    with httpx2.Client(timeout=60) as http:
        # Эмбеддер первым: без ключа ошибка сразу, ещё до сетевых вызовов.
        embedder = make_embedder(settings, http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=60))
        if if_needed:
            reason = rebuild_reason(
                store,
                ref=settings.dataset_revision,
                model=embedder.model,
                sample=sample,
                seed=seed,
            )
            if reason is None:
                print(
                    "индекс уже построен для этой ревизии и модели, пересборка не нужна"
                )
                return 0
            _log(f"строю индекс: {reason}")
        path = ensure_dataset(
            settings.dataset_revision,
            settings.data_path,
            http,
            sha256=known_sha256(settings.dataset_revision),
        )
        dataset = read_dataset(path)
        skipped = ", ".join(f"{k} {v}" for k, v in sorted(dataset.skipped.items()))
        _log(
            f"рецептов в датасете: {len(dataset.recipes)}, пропущено: {skipped or 'нет'}"
        )
        recipes = (
            pick_sample(dataset.recipes, sample, seed) if sample else dataset.recipes
        )
        plan = estimate(recipes, embedder.model, embedder.max_batch_texts)
        _log(f"оценка для {embedder.model}: {plan}")
        if plan.needs_confirmation and not yes:
            _log(
                "оценка дороже порога подтверждения: если всё верно, запустите "
                "снова с YES=1 (make index YES=1)"
            )
            return 2
        report = build_index(
            recipes,
            source=IndexSource(
                ref=settings.dataset_revision,
                file_sha256=dataset.sha256,
                skipped=dataset.skipped,
                sample=len(recipes) if sample else None,
                seed=seed if sample else None,
            ),
            store=store,
            embedder=embedder,
            log=_log,
            quantization=quantization,
        )
    manifest = report.manifest
    masked = ", ".join(f"{k} {v}" for k, v in sorted(report.masked.items())) or "нет"
    print(
        f"Индекс построен: {manifest.recipes} рецептов, ревизия {manifest.ref[:8]}, "
        f"модель {manifest.embedding_model}, размерность {manifest.dim}, "
        f"сжатие {manifest.quantization or 'нет'}\n"
        f"токенов: {report.tokens}, стоимость: {report.cost or 'цена модели неизвестна'}\n"
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
        blocks.append(
            f"{number}. [{hit.score:.3f}] {chunk.name}\n"
            f"   {chunk.url}\n"
            f"   ингредиенты: {', '.join(chunk.ingredients)}"
        )
    return "\n\n".join(blocks)


def format_plan(args: SearchArgs) -> str:
    """Во что LLM разобрала вопрос: запрос для векторов и условия фильтра."""
    lines = [f"запрос: {args.query}"]
    if args.conditions.include:
        groups = (" или ".join(group) for group in args.conditions.include)
        lines.append(f"с: {'; '.join(groups)}")
    if args.conditions.exclude:
        lines.append(f"без: {', '.join(args.conditions.exclude)}")
    if args.dropped:
        lines.append(f"нет в словаре, отброшено: {', '.join(args.dropped)}")
    return "\n".join(lines)


def _search(
    settings: Settings, question: str, k: int, as_json: bool, filters: bool
) -> int:
    with httpx2.Client(timeout=30) as http, make_http_client(settings) as llm_http:
        # Ключи проверяются до Qdrant: ошибка настройки видна сразу.
        embedder = make_embedder(settings, http)
        llm = make_chat_model(settings, llm_http) if filters else None
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        retriever = Retriever(store, embedder, ref=settings.dataset_revision)
        if llm is None:
            args = SearchArgs(query=question, conditions=IngredientFilter())
        else:
            planner = SearchPlanner(
                llm, model=llm_model(settings), vocabulary=store.vocabulary()
            )
            args = planner.plan(question)
        hits = retriever.search(args.query, k=k, conditions=args.conditions)
    if as_json:
        payload = {
            "query": args.query,
            "include": [list(group) for group in args.conditions.include],
            "exclude": list(args.conditions.exclude),
            "dropped": list(args.dropped),
            "hits": [{"score": hit.score, **hit.chunk.model_dump()} for hit in hits],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{format_plan(args)}\n\n{format_hits(hits)}")
    return 0


def _ask(settings: Settings, question: str, as_json: bool) -> int:
    with (
        httpx2.Client(timeout=30) as embeddings_http,
        make_http_client(settings) as llm_http,
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
    commands.add_parser("data", help="скачать датасет на ревизии DATASET_REVISION")
    index = commands.add_parser("index", help="построить индекс рецептов заново")
    index.add_argument(
        "--if-needed",
        action="store_true",
        help="только если индекса нет, он для другой ревизии или модели или это выборка",
    )
    index.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="только N рецептов; без флага — INDEX_SAMPLE из окружения, пусто — весь датасет",
    )
    index.add_argument("--seed", type=int, default=SAMPLE_SEED, help="сид выборки")
    index.add_argument(
        "--yes", action="store_true", help="не спрашивать подтверждения дорогой сборки"
    )
    index.add_argument(
        "--no-quantization",
        action="store_true",
        help="без сжатия int8 в памяти: для замера, сколько оно стоит по качеству",
    )
    ask = commands.add_parser("ask", help="ответ со ссылками на рецепты")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="вывод в JSON")
    search = commands.add_parser(
        "search",
        help="найти рецепты: разбор вопроса LLM, фильтр и векторы, без ответа",
    )
    search.add_argument("question")
    search.add_argument("-k", type=int, default=5, help="сколько рецептов показать")
    search.add_argument("--json", action="store_true", help="вывод в JSON")
    search.add_argument(
        "--no-filters",
        action="store_true",
        help="только векторы, без разбора вопроса и фильтра (как на этапе 1)",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        if args.command == "data":
            return _data(settings)
        if args.command == "index":
            return _index(
                settings,
                if_needed=args.if_needed,
                sample=args.sample
                if args.sample is not None
                else settings.index_sample,
                seed=args.seed,
                yes=args.yes,
                quantization=not args.no_quantization,
            )
        if args.command == "ask":
            return _ask(settings, args.question, args.json)
        if args.command == "search":
            return _search(
                settings, args.question, args.k, args.json, filters=not args.no_filters
            )
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
