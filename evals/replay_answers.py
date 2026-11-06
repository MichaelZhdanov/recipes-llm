"""Повтор шага ответа по сохранённому разбору: `python -m evals.replay_answers RAW`.

Kimi K2.6 хорошо понимает запрос (разбор — 14 с по медиане), но долго
пишет ответ (77 с). Идея — разбор оставить ему, а ответ отдать быстрой
модели. Чтобы сравнить быстрые модели честно и дёшево, разбор не
перезапускается: из сырых данных прогона (`make eval`) берутся запросы,
«с чем» и «без чего», поиск повторяется по ним без LLM — можно шире
(`--k`), — и ответ пишет заданная модель тем же кодом, что в боте
(`Answerer` с сохранённым разбором вместо `SearchPlanner`).

Без проверки перед отправкой и без нового поиска на «не нашёл»: меряется
сам писатель ответа. Отчёт и сырые данные — как у `make eval`, судья —
`evals/subagent_judge.py`. Время в отчёте: поиск — без разбора, ответ —
только эта модель.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx2
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.generation.answer import Answerer
from assistant.generation.gigachat import DETERMINISTIC_TEMPERATURE, Message
from assistant.generation.llm import llm_model, make_chat_model, make_http_client
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from assistant.tools.search_recipes import SearchArgs
from evals.dataset import load_cases
from evals.report import RunMeta
from evals.run import _finish, load_answer_run, mark_answerable, run_answers


class SavedPlanner:
    """Разбор вопроса из сырых данных прогона: тот же, что сделала модель тогда."""

    def __init__(self, plans: Mapping[str, SearchArgs]) -> None:
        self._plans = plans

    def plan(
        self, question: str, history: Sequence[Message] = (), note: str = ""
    ) -> SearchArgs:
        return self._plans[question]


def saved_plans(raw: dict[str, object]) -> dict[str, SearchArgs]:
    """Вопрос кейса → разбор первого прогона; без разбора (сбой) — пропускается."""
    cases = {c.id: c for c in load_cases()}
    plans: dict[str, SearchArgs] = {}
    for record in raw["records"]:  # type: ignore[attr-defined]
        case = cases.get(record["id"])
        if record["run"] != 0 or case is None:
            continue
        searches = load_answer_run(record).searches
        if searches:
            plans[case.question] = searches[0]
    return plans


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.replay_answers", description=__doc__
    )
    parser.add_argument("raw", type=Path, help="сырые данные прогона с разбором")
    parser.add_argument("--provider", default="cloudru")
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="")
    parser.add_argument(
        "--k", type=int, default=8, help="сколько рецептов найти для ответа"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", required=True, help="суффикс имени отчёта")
    args = parser.parse_args(argv)

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    plans = saved_plans(raw)
    cases = [c for c in load_cases() if c.question in plans]
    settings = Settings().model_copy(
        update={
            "llm_provider": args.provider,
            "llm_model": args.model,
            "llm_effort": args.effort,
            "answer_review": False,
        }
    )
    with httpx2.Client(timeout=30) as http, make_http_client(settings) as llm_http:
        embedder = make_embedder(settings, http)
        retriever = Retriever(
            VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30)),
            embedder,
            ref=settings.dataset_revision,
        )
        llm = make_chat_model(settings, llm_http)
        model = llm_model(settings)
        answerer = Answerer(
            retriever,
            llm,
            model=model,
            ref=settings.dataset_revision,
            k=args.k,
            planner=SavedPlanner(plans),
        )
        print(
            f"повтор ответа: {model}, effort {args.effort or 'по умолчанию'}, k={args.k}, "
            f"{len(cases)} кейсов, разбор из {args.raw.name}",
            file=sys.stderr,
            flush=True,
        )
        results = run_answers(cases, answerer, runs=1, workers=args.workers)
        results = mark_answerable(results, retriever)
    meta = RunMeta(
        manifest=retriever.manifest,
        revision=raw["meta"]["revision"],
        provider=embedder.provider.name,
        query_template=embedder.query_template,
        embedding_stats=embedder.stats,
        llm_stats=llm.stats,
        llm_model=model,
        temperature=DETERMINISTIC_TEMPERATURE,
        prompt_version=answerer.prompt_version,
        rescored_from=f"повтор ответа по разбору из {args.raw}",
        llm_provider=args.provider,
    )
    return _finish(results, meta, None, args.label)


if __name__ == "__main__":
    sys.exit(main())
