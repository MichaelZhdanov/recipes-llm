"""Прогон эталонного набора: `python -m evals.run` (через `make eval`).

Здесь только порядок действий: загрузить кейсы, прогнать, записать отчёт
и сырые данные, вернуть код выхода. Метрики считают
`evals/retrieval_metrics.py` и `evals/answer_metrics.py`, отчёт
оформляет `evals/report.py`.

Режимы (`--stage`):
- `answer` (по умолчанию) — ассистент целиком: поиск, GigaChat, проверка
  ссылок (промпт под рецепты — этап 3);
- `retrieval` — только поиск: GigaChat разбирает вопрос в фильтры
  (search_recipes), Qdrant фильтрует, векторы ранжируют; метрики считает
  код. `--no-filters` — только векторы, как на этапе 1 (без GigaChat).

После генерации модель-судья (`evals/judge.py`) оценивает groundedness
и проверяется на канарейках; `--no-judge` её выключает. `--rescore RAW`
пересчитывает отчёт по сохранённым ответам модели без новой генерации:
ссылки проверяются текущим кодом, судья запускается заново.

Каждый кейс прогоняется RUNS раз (правило из CLAUDE.md), в отчёте среднее
и разброс.

Быстрый прогон при работе над промптом: `--sample N --runs 1` — N кейсов,
поровну из каждой категории, с фиксированным сидом. Одинаковый сид даёт ту же
выборку, так что быстрые прогоны разных версий сравнимы между собой. Цифры
такого прогона ориентировочные. В отчёт этапа идёт только полный прогон.

Отчёт: evals/reports/<дата>[-метка].md, сырые данные — в evals/reports/raw/
(эта папка в .gitignore), одна запись на каждую пару «кейс, прогон».

Пока идёт генерация, каждый готовый кейс сразу дописывается в журнал
`raw/<отчёт>.partial.jsonl` (`evals/progress.py`). Если прогон прервался
(429 у GigaChat, Ctrl+C), `--resume ЖУРНАЛ` с теми же флагами продолжает его:
готовые кейсы не прогоняются заново.
Код выхода 1, если метрика ниже порога.
"""

import argparse
import dataclasses
import json
import random
import re
import shlex
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import httpx2
from qdrant_client import QdrantClient

from assistant.config import AnswerMode, LlmProvider, Settings
from assistant.generation import llm as assistant_llm
from assistant.generation.answer import Answerer, assemble
from assistant.generation.citations import SourceText
from assistant.generation.gigachat import (
    DETERMINISTIC_TEMPERATURE,
    ChatModel,
    Completion,
    LlmStats,
    Usage,
)
from assistant.indexing.build import pick_sample
from assistant.indexing.chunks import Chunk
from assistant.indexing.dataset import (
    Recipe,
    ensure_dataset,
    known_sha256,
    read_dataset,
    recipe_ids_in,
)
from assistant.retrieval.embeddings import EmbeddingStats, make_embedder
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import Hit, IndexManifest, VectorStore
from assistant.tools.search_recipes import PROMPT_VERSION as SEARCH_PROMPT
from assistant.tools.search_recipes import SearchArgs, SearchPlanner
from evals import progress
from evals.answer_metrics import AnswerCase, AnswerRun, needs_earlier_recipes
from evals.dataset import (
    CANARIES_FILE,
    CLARITY_CANARIES_FILE,
    DATASET_DIR,
    HOLDOUT_DIR,
    RELEVANCE_CANARIES_FILE,
    Case,
    load_canaries,
    load_cases,
    load_clarity_canaries,
    load_relevance_canaries,
)
from evals.judge import JUDGE_EFFORT, JUDGE_MODELS, Judge, Judging, run_judge
from evals.progress import add_embedding_stats, add_llm_stats, journal_path
from evals.relevance import Relevance, RelevanceJudge, judge_relevance
from evals.report import (
    Parsing,
    RunMeta,
    answer_gates,
    git_revision,
    render_answer_report,
    render_report,
    report_path,
    retrieval_gates,
)
from evals.retrieval_metrics import (
    CaseResult,
    conditions,
    filtered_search,
    has_conditions,
    history,
    mark_available,
    run_cases,
    run_score,
    unknown_ingredients,
    vector_search,
)

RUNS = 3
SAMPLE_SEED = 20261104
# Категории каждого режима.
# У by_name, include, exclude и follow_up поиск проверяет код, у semantic — судья.
# Из follow_up в поиск идут только кейсы с условиями: вопросы о количестве
# в уже найденном рецепте проверяет ответ (этап 3).
RETRIEVAL_CATEGORIES = frozenset(
    {"by_name", "include", "exclude", "semantic", "follow_up"}
)
# Эскалация и безопасность — этапы 5 и 7.
ANSWER_CATEGORIES = frozenset(
    {
        "include",
        "exclude",
        "by_name",
        "semantic",
        "how_to_cook",
        "out_of_scope",
        "chit_chat",
        "follow_up",
    }
)
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass(frozen=True)
class JudgeChoice:
    """Какая модель судит: `--judge-provider`, `--judge-model`, `--judge-effort`."""

    provider: LlmProvider = "anthropic"
    model: str = ""
    # Только у Claude. Задан явно, а не «по умолчанию модели»: если вендор сменит
    # умолчание, судья не станет тихо мягче или строже.
    effort: str = JUDGE_EFFORT

    @property
    def model_name(self) -> str:
        return self.model or JUDGE_MODELS[self.provider]

    @property
    def label(self) -> str:
        """Как судья подписан в отчёте: модель и, у Claude, effort."""
        if self.provider == "anthropic" and self.effort:
            return f"{self.model_name}, effort {self.effort}"
        return self.model_name


DEFAULT_JUDGE = JudgeChoice()


def raw_records(
    results: Sequence[CaseResult], relevance: Relevance | None = None
) -> list[dict[str, object]]:
    """Сырые данные поиска: одна запись на пару «кейс, прогон»."""
    marks = relevance.marks if relevance else {}
    return [
        {
            "id": r.case.id,
            "run": number,
            "score": run_score(r.case, run),
            "available": r.available,
            "top": [dataclasses.asdict(found) for found in run.top],
            "latency": run.latency,
            "args": dataclasses.asdict(run.args) if run.args else None,
            "error": run.error,
            # Вердикты судьи по рецептам топа: судится только первый прогон.
            "relevance": marks.get(r.case.id) if number == 0 else None,
        }
        for r in results
        for number, run in enumerate(r.runs)
    ]


def indexed_recipes(
    settings: Settings, manifest: IndexManifest, http: httpx2.Client
) -> tuple[list[Recipe], set[str]]:
    """Рецепты, которые сейчас в индексе, и словарь имён ингредиентов датасета.

    Выборку восстанавливаем по паспорту (размер и сид): так метрика знает,
    есть ли в индексе вообще подходящий рецепт.
    """
    path = ensure_dataset(
        manifest.ref,
        settings.data_path,
        http,
        sha256=known_sha256(manifest.ref),
    )
    recipes = read_dataset(path).recipes
    vocabulary = {i.name for r in recipes for i in r.ingredients}
    if manifest.sample is not None and manifest.seed is not None:
        recipes = pick_sample(recipes, manifest.sample, manifest.seed)
    return recipes, vocabulary


def answer_records(
    results: Sequence[AnswerCase], judging: Judging | None = None
) -> list[dict[str, object]]:
    """Сырые данные ответов: всё, чтобы разобрать прогон и пересчитать метрики без LLM."""
    verdicts = judging.verdicts if judging else {}
    return [
        {
            "id": r.case.id,
            "run": number,
            "answer": run.answer.model_dump(),
            "content": run.content,
            "finish_reason": run.finish_reason,
            "usage": dataclasses.asdict(run.usage) if run.usage else None,
            "retrieval_seconds": run.retrieval_seconds,
            "generation_seconds": run.generation_seconds,
            "top": run.top,
            "sources": [
                {"number": s.number, "chunk": s.chunk.model_dump(), "text": s.text}
                for s in run.sources
            ],
            "checked": [dataclasses.asdict(s) for s in run.checked],
            "searches": [dataclasses.asdict(args) for args in run.searches],
            "available": r.available,
            "judge": (
                dataclasses.asdict(verdicts[(r.case.id, number)])
                if (r.case.id, number) in verdicts
                else None
            ),
        }
        for r in results
        for number, run in enumerate(r.runs)
    ]


def judging_record(judging: Judging) -> dict[str, object]:
    """Судья и его вердикты по канарейкам: по ним сравнивают двух судей
    (`evals/compare_judges.py`)."""
    return {
        "model": judging.model,
        "prompt_version": judging.prompt_version,
        "canaries": [
            {"id": c.id, "expected": c.supported, "mark": m}
            for c, m in zip(judging.canaries, judging.canary_marks, strict=True)
        ],
        "clarity_canaries": [
            {"id": c.id, "expected": c.clear, "mark": m}
            for c, m in zip(
                judging.clarity_canaries, judging.clarity_marks, strict=True
            )
        ],
    }


def meta_record(meta: RunMeta) -> dict[str, object]:
    return {
        "manifest": meta.manifest.model_dump(),
        "revision": meta.revision,
        "provider": meta.provider,
        "query_template": meta.query_template,
        "embedding_stats": meta.embedding_stats.as_dict(),
        "llm_stats": meta.llm_stats.as_dict(),
        "llm_model": meta.llm_model,
        "temperature": meta.temperature,
        "prompt_version": meta.prompt_version,
        "split": meta.split,
        "sample": meta.sample,
        "mode": meta.mode,
        "llm_provider": meta.llm_provider,
    }


def load_meta(data: dict[str, Any], *, revision: str, rescored_from: str) -> RunMeta:
    embedding = data["embedding_stats"]
    llm = data["llm_stats"]
    return RunMeta(
        manifest=IndexManifest.model_validate(data["manifest"]),
        revision=revision,
        provider=data["provider"],
        query_template=data["query_template"],
        embedding_stats=EmbeddingStats(
            requests=embedding["requests"],
            tokens=embedding["tokens"],
            masked=Counter(embedding["masked"]),
        ),
        llm_stats=LlmStats(**{**llm, "masked": Counter(llm["masked"])}),
        llm_model=data["llm_model"],
        temperature=data["temperature"],
        prompt_version=data["prompt_version"],
        split=data.get("split", "dev"),
        sample=data.get("sample"),
        mode=data.get("mode", "always"),
        llm_provider=data.get("llm_provider", "gigachat"),
        rescored_from=f"{rescored_from}, прогон на {data['revision']}",
    )


def load_answer_run(record: dict[str, Any]) -> AnswerRun:
    """Прогон из сырых данных. Ссылки проверяются заново текущим кодом (assemble)."""
    sources = [
        SourceText(s["number"], Chunk.model_validate(s["chunk"]), s["text"])
        for s in record["sources"]
    ]
    # Среди источников есть и рецепты прошлых ходов: найденные — по id из топа.
    chunks = {source.chunk.source_id: source.chunk for source in sources}
    hits = [
        Hit(chunk=chunks[source_id], score=score)
        for source_id, score in record["top"]
        if source_id in chunks
    ]
    searches = [
        SearchArgs(
            query=args["query"],
            conditions=IngredientFilter(
                include=tuple(tuple(g) for g in args["conditions"]["include"]),
                exclude=tuple(args["conditions"]["exclude"]),
            ),
            dropped=tuple(args["dropped"]),
        )
        for args in record.get("searches", [])
    ]
    completion = None
    if record["finish_reason"] is not None:
        completion = Completion(
            content=record["content"],
            finish_reason=record["finish_reason"],
            model=record["answer"]["model"],
            usage=Usage(**record["usage"]),
        )
    result = assemble(
        hits,
        sources,
        completion,
        prompt_version=record["answer"]["prompt_version"],
        ref=record["answer"]["dataset_revision"],
        retrieval_seconds=record["retrieval_seconds"],
        generation_seconds=record["generation_seconds"],
        args=searches[0] if searches else None,
        searches=searches,
    )
    return AnswerRun.from_result(result)


def load_answer_cases(
    records: Sequence[dict[str, Any]], cases: dict[str, Case]
) -> list[AnswerCase]:
    runs: dict[str, list[AnswerRun]] = {}
    available: dict[str, bool | None] = {}
    for record in records:
        runs.setdefault(record["id"], []).append(load_answer_run(record))
        available[record["id"]] = record.get("available")
    return [
        AnswerCase(cases[case_id], case_runs, available[case_id])
        for case_id, case_runs in runs.items()
    ]


def pick_cases(cases: Sequence[Case], size: int, seed: int = SAMPLE_SEED) -> list[Case]:
    """size кейсов: по кругу из каждой категории, порядок набора сохраняется.

    По кругу, а не просто случайно: в маленькой выборке иначе может не
    оказаться out_of_scope, и abstention не посчитать.
    """
    if size >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    groups: dict[str, list[Case]] = {}
    for case in cases:
        groups.setdefault(case.category, []).append(case)
    queues = [rng.sample(group, len(group)) for _, group in sorted(groups.items())]
    picked: list[Case] = []
    while len(picked) < size:
        for queue in queues:
            if queue and len(picked) < size:
                picked.append(queue.pop())
    order = {case.id: index for index, case in enumerate(cases)}
    return sorted(picked, key=lambda case: order[case.id])


def _choose(
    cases: list[Case], sample: int | None, seed: int
) -> tuple[list[Case], str | None]:
    if sample is None or sample >= len(cases):
        return cases, None
    return pick_cases(cases, sample, seed), f"{sample} из {len(cases)}, сид {seed}"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _no_cases(directory: Path, categories: frozenset[str]) -> int:
    """Прогонять нечего: понятное сообщение и код 2, а не пустой отчёт."""
    _log(f"нет кейсов в {directory} для категорий {', '.join(sorted(categories))}")
    return 2


class RecipeIndex(Protocol):
    """Откуда берутся рецепты по id: поиск ассистента или само хранилище."""

    def recipes(self, ids: Sequence[int]) -> list[Chunk]: ...


def mark_answerable(
    results: Sequence[AnswerCase], index: RecipeIndex
) -> list[AnswerCase]:
    """Отмечает уточнения, рецептов прошлого ответа которых нет в индексе.

    На выборке из датасета их нет почти никогда: ассистент честно говорит
    «не нашёл», и это не ложный отказ. Рецепты не подставляются из датасета:
    тогда проверялась бы не та система, что работает у пользователя.
    """
    marked = []
    for r in results:
        if not needs_earlier_recipes(r.case):
            marked.append(r)
            continue
        ids = recipe_ids_in(
            "\n".join(turn.text for turn in r.case.turns if turn.role == "assistant")
        )
        found = {chunk.recipe_id for chunk in index.recipes(ids)}
        marked.append(dataclasses.replace(r, available=set(ids) <= found))
    return marked


def run_answers(
    cases: Sequence[Case],
    answerer: Answerer,
    *,
    runs: int,
    done: Mapping[str, AnswerCase] | None = None,
    on_case: Callable[[AnswerCase], None] | None = None,
) -> list[AnswerCase]:
    """Прогоняет кейсы по порядку. Готовые (done) берутся из журнала, а не заново.

    on_case вызывается после каждого нового кейса: так он сразу попадает в журнал.
    """
    results = []
    for index, case in enumerate(cases, start=1):
        if done is not None and case.id in done:
            results.append(done[case.id])
            _log(f"[{index}/{len(cases)}] {case.id}: из журнала")
            continue
        case_runs = [
            AnswerRun.from_result(answerer.answer(case.question, history(case)))
            for _ in range(runs)
        ]
        outcome = ", ".join(run.answer.reason or "ответ" for run in case_runs)
        _log(f"[{index}/{len(cases)}] {case.id}: {outcome}")
        result = AnswerCase(case=case, runs=case_runs)
        if on_case is not None:
            on_case(result)
        results.append(result)
    return results


def _write_raw(path: Path, raw: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = REPORTS_DIR / "raw" / f"{path.stem}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def _write(path: Path, report: str, raw: object) -> None:
    _write_raw(path, raw)
    path.write_text(report, encoding="utf-8")


def _retrieval_stage(
    settings: Settings,
    runs: int,
    label: str | None,
    sample: int | None,
    seed: int,
    categories: frozenset[str] = RETRIEVAL_CATEGORIES,
    judge: bool = True,
    quantization: bool = True,
    filters: bool = True,
    judge_choice: JudgeChoice = DEFAULT_JUDGE,
) -> int:
    # Коммит запоминается до прогона: правки во время долгого прогона
    # не должны попасть в его отчёт.
    revision = git_revision()
    selected = [
        c
        for c in load_cases()
        if c.category in categories and (c.category != "follow_up" or has_conditions(c))
    ]
    cases, sampled = _choose(selected, sample, seed)
    if not cases:
        return _no_cases(DATASET_DIR, categories)
    with (
        httpx2.Client(timeout=30) as http,
        assistant_llm.make_http_client(settings) as llm_http,
    ):
        embedder = make_embedder(settings, http)
        llm = assistant_llm.make_chat_model(settings, llm_http) if filters else None
        store = VectorStore(
            QdrantClient(url=settings.qdrant_url, timeout=30), quantization=quantization
        )
        retriever = Retriever(store, embedder, ref=settings.dataset_revision)
        recipes, vocabulary = indexed_recipes(settings, retriever.manifest, http)
        if unknown := unknown_ingredients(cases, vocabulary):
            print(
                "в кейсах есть ингредиенты не из словаря датасета:",
                *unknown,
                sep="\n  ",
            )
            return 2
        parsing = None
        if llm is None:
            search = vector_search(retriever)
        else:
            index_vocabulary = store.vocabulary()
            planner = SearchPlanner(
                llm,
                model=assistant_llm.llm_model(settings),
                vocabulary=index_vocabulary,
            )
            search = filtered_search(planner, retriever)
        results = mark_available(run_cases(cases, search, runs=runs), recipes)
        if llm is not None:
            parsing = Parsing(
                model=assistant_llm.llm_model(settings),
                served=tuple(sorted(planner.served)),
                prompt_version=SEARCH_PROMPT,
                vocabulary=len(index_vocabulary),
                stats=llm.stats,
                retries=planner.retries,
                provider=settings.llm_provider,
                effort=settings.llm_effort,
            )
    relevance = None
    if judge and any(r.case.category == "semantic" for r in results):
        relevance = _judge_relevance(settings, results, judge_choice)

    today = date.today()
    report = render_report(
        results,
        manifest=retriever.manifest,
        revision=revision,
        provider=embedder.provider.name,
        query_template=embedder.query_template,
        stats=embedder.stats,
        today=today,
        sample=sampled,
        relevance=relevance,
        quantization=quantization,
        parsing=parsing,
    )
    path = report_path(REPORTS_DIR, today, label)
    _write(path, report, raw_records(results, relevance))
    value = conditions(results)
    gates = retrieval_gates(results, relevance)
    failed = [name for name, passed in gates.items() if not passed]
    print(
        f"conditions@5 = {'—' if value is None else f'{value:.2f}'}, "
        f"пороги: {'не пройдены — ' + ', '.join(failed) if failed else 'пройдены'}, "
        f"отчёт: {path}"
    )
    return 1 if failed else 0


@contextmanager
def _judge_llm(settings: Settings, choice: JudgeChoice) -> Iterator[ChatModel]:
    """Свой клиент и свой экземпляр модели: токены судьи считаются отдельно от
    ассистента, а у GigaChat свой TLS-корень. Тот же выбор провайдера, что
    у ассистента, но со своим провайдером и без LLM_EFFORT ассистента."""
    effort = choice.effort if choice.provider == "anthropic" else ""
    judge_settings = settings.model_copy(
        update={"llm_provider": choice.provider, "llm_effort": effort}
    )
    with assistant_llm.make_http_client(judge_settings) as http:
        yield assistant_llm.make_chat_model(judge_settings, http)


def _judge_relevance(
    settings: Settings, results: Sequence[CaseResult], choice: JudgeChoice
) -> Relevance:
    """Судья оценивает выдачу по вопросам «по смыслу»."""
    canaries = load_relevance_canaries() if RELEVANCE_CANARIES_FILE.exists() else []
    with _judge_llm(settings, choice) as llm:
        judge = RelevanceJudge(llm, model=choice.model_name)
        _log(f"судья {judge.model}: выдача по смыслу и {len(canaries)} канареек")
        return dataclasses.replace(
            judge_relevance(results, judge, canaries), model=choice.label
        )


def _judge(
    settings: Settings, results: Sequence[AnswerCase], choice: JudgeChoice
) -> Judging:
    with _judge_llm(settings, choice) as llm:
        judge = Judge(llm, model=choice.model_name)
        _log(
            f"судья {judge.model}: {sum(len(r.runs) for r in results)} ответов и канарейки"
        )
        # Канарейки пишутся вместе с промптом судьи: пока их нет, судья работает без них.
        canaries = load_canaries() if CANARIES_FILE.exists() else []
        clarity = load_clarity_canaries() if CLARITY_CANARIES_FILE.exists() else []
        return dataclasses.replace(
            run_judge(results, judge, canaries, clarity), model=choice.label
        )


def _finish(
    results: Sequence[AnswerCase],
    meta: RunMeta,
    judging: Judging | None,
    label: str | None,
) -> int:
    today = date.today()
    path = report_path(REPORTS_DIR, today, label)
    # Сначала сырые данные: если отчёт упадёт, дорогая генерация не пропадёт,
    # и отчёт можно собрать заново через --rescore.
    raw: dict[str, object] = {
        "meta": meta_record(meta),
        "records": answer_records(results, judging),
    }
    if judging is not None:
        raw["judging"] = judging_record(judging)
    _write_raw(path, raw)
    report = render_answer_report(results, meta=meta, judging=judging, today=today)
    path.write_text(report, encoding="utf-8")
    gates = answer_gates(results, judging)
    failed = [name for name, passed in gates.items() if not passed]
    print(f"не прошли порог: {', '.join(failed)}" if failed else "все пороги пройдены")
    print(f"отчёт: {path}")
    return 1 if failed else 0


def _answer_stage(
    settings: Settings,
    runs: int,
    label: str | None,
    judge: bool,
    split: str,
    sample: int | None = None,
    seed: int = SAMPLE_SEED,
    categories: frozenset[str] = ANSWER_CATEGORIES,
    resume: Path | None = None,
    command: Sequence[str] = (),
    mode: AnswerMode | None = None,
    judge_choice: JudgeChoice = DEFAULT_JUDGE,
) -> int:
    revision = git_revision()
    mode = mode or settings.answer_mode
    directory = HOLDOUT_DIR if split == "holdout" else DATASET_DIR
    cases, sampled = _choose(
        [c for c in load_cases(directory) if c.category in categories],
        sample,
        seed,
    )
    if not cases:
        return _no_cases(directory, categories)
    journal_file = resume or journal_path(
        REPORTS_DIR / "raw", report_path(REPORTS_DIR, date.today(), label)
    )
    model = assistant_llm.llm_model(settings)
    with (
        httpx2.Client(timeout=30) as embeddings_http,
        assistant_llm.make_http_client(settings) as llm_http,
    ):
        embedder = make_embedder(settings, embeddings_http)
        llm = assistant_llm.make_chat_model(settings, llm_http)
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        retriever = Retriever(store, embedder, ref=settings.dataset_revision)
        journal = _open_journal(journal_file, resume is not None, revision, cases, runs)
        if journal is None:
            return 2
        done = _resumed_cases(journal, cases)
        answerer = Answerer(
            retriever, llm, model=model, ref=settings.dataset_revision, mode=mode
        )
        _log(
            f"модель {model}, режим {mode}, {len(cases)} кейсов × {runs} прогона, "
            f"журнал {journal_file}"
        )

        def save(result: AnswerCase) -> None:
            progress.append(
                journal_file,
                answer_records([result]),
                add_llm_stats(journal.llm_stats, llm.stats),
                add_embedding_stats(journal.embedding_stats, embedder.stats),
            )

        try:
            results = run_answers(cases, answerer, runs=runs, done=done, on_case=save)
            results = mark_answerable(results, retriever)
            judging = _judge(settings, results, judge_choice) if judge else None
        except BaseException:
            # Готовые кейсы уже в журнале: подсказываем, как продолжить, и падаем как раньше.
            args = shlex.join(
                [*_without_resume(command), "--resume", str(journal_file)]
            )
            _log(f"прогон прерван, готовые кейсы — в {journal_file}")
            _log(f"продолжить: make eval ARGS={shlex.quote(args)}")
            raise

    meta = RunMeta(
        manifest=retriever.manifest,
        revision=journal.revision,
        provider=embedder.provider.name,
        query_template=embedder.query_template,
        embedding_stats=add_embedding_stats(journal.embedding_stats, embedder.stats),
        llm_stats=add_llm_stats(journal.llm_stats, llm.stats),
        llm_model=model,
        temperature=DETERMINISTIC_TEMPERATURE,
        prompt_version=answerer.prompt_version,
        split=split,
        sample=sampled,
        mode=mode,
        llm_provider=settings.llm_provider,
    )
    code = _finish(results, meta, judging, label)
    # Всё из журнала теперь в raw JSON, а оставленный журнал помешал бы новому прогону.
    journal_file.unlink()
    return code


def _open_journal(
    path: Path, resume: bool, revision: str, cases: Sequence[Case], runs: int
) -> progress.Journal | None:
    """Новый журнал или продолжение старого; None и сообщение, если так нельзя."""
    ids = [c.id for c in cases]
    if not resume:
        try:
            return progress.start(path, revision, ids, runs)
        except FileExistsError:
            print(
                f"есть журнал незаконченного прогона {path}: продолжите его с "
                f"--resume {path} или уберите файл, чтобы начать заново"
            )
            return None
    journal = progress.load(path)
    if journal.case_ids != ids or journal.runs != runs:
        print(
            f"журнал {path} — от другого прогона: {len(journal.case_ids)} кейсов × "
            f"{journal.runs}, а сейчас {len(ids)} × {runs}. Запустите с теми же флагами."
        )
        return None
    return journal


def _resumed_cases(
    journal: progress.Journal, cases: Sequence[Case]
) -> dict[str, AnswerCase]:
    """Готовые кейсы из журнала. Ссылки проверяются заново текущим кодом, как в --rescore."""
    records = [
        record for case_records in journal.done.values() for record in case_records
    ]
    by_id = {c.id: c for c in cases}
    return {r.case.id: r for r in load_answer_cases(records, by_id)}


def _without_resume(command: Sequence[str]) -> list[str]:
    """Флаги прогона без старого --resume: для подсказки, как продолжить."""
    result: list[str] = []
    skip = False
    for arg in command:
        if skip:
            skip = False
        elif arg == "--resume":
            skip = True
        elif not arg.startswith("--resume="):
            result.append(arg)
    return result


def rescore_label(raw_path: Path) -> str:
    """2026-11-04-gigachat-2-pro.json → gigachat-2-pro-rescore: дату добавит report_path."""
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}-?", "", raw_path.stem)
    return f"{stem}-rescore" if stem else "rescore"


def _rescore(
    settings: Settings,
    raw_path: Path,
    label: str | None,
    judge: bool,
    judge_choice: JudgeChoice = DEFAULT_JUDGE,
) -> int:
    """Метрики заново по сохранённым ответам модели: без генерации, ссылки — текущим кодом."""
    revision = git_revision()
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    cases = {c.id: c for c in [*load_cases(), *load_cases(HOLDOUT_DIR)]}
    results = load_answer_cases(data["records"], cases)
    if any(needs_earlier_recipes(r.case) and r.available is None for r in results):
        # Старые сырые данные без отметки: проверяем по текущему индексу.
        store = VectorStore(QdrantClient(url=settings.qdrant_url, timeout=30))
        results = mark_answerable(results, store)
    meta = load_meta(data["meta"], revision=revision, rescored_from=str(raw_path))
    judging = _judge(settings, results, judge_choice) if judge else None
    return _finish(results, meta, judging, label or rescore_label(raw_path))


def _categories(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="быстрый прогон: N кейсов поровну из категорий (с --runs 1 — за минуты)",
    )
    parser.add_argument(
        "--seed", type=int, default=SAMPLE_SEED, help="сид для --sample"
    )
    parser.add_argument(
        "--categories",
        type=_categories,
        help="только эти категории через запятую, например by_name,include",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "holdout"),
        default="dev",
        help="holdout — отложенная выборка, её прогоняют один раз в конце",
    )
    parser.add_argument("--label", help="суффикс имени отчёта, например имя модели")
    parser.add_argument(
        "--stage",
        choices=("answer", "retrieval"),
        default="answer",
        help="answer — ассистент целиком; retrieval — только поиск (разбор вопроса, фильтр, векторы)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="без модели-судьи (groundedness и релевантность по смыслу не считаются)",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="STAGE=retrieval: искать по точным векторам, не используя сжатые (замер цены int8)",
    )
    parser.add_argument(
        "--judge-provider",
        choices=("gigachat", "anthropic"),
        default="anthropic",
        help="кто судит: anthropic — Claude Sonnet 5.5 (по умолчанию), gigachat — GigaChat-3-Ultra",
    )
    parser.add_argument(
        "--judge-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=JUDGE_EFFORT,
        help=f"сколько думает судья Claude (по умолчанию {JUDGE_EFFORT}); у GigaChat не действует",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="модель судьи; пусто — по умолчанию провайдера",
    )
    parser.add_argument(
        "--mode",
        choices=("always", "auto"),
        help="ответы: always — сначала поиск; auto — модель решает (по умолчанию ANSWER_MODE)",
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="STAGE=retrieval: только векторы, без разбора вопроса и фильтра (базовая линия)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="JOURNAL",
        help="продолжить прерванный прогон по журналу evals/reports/raw/*.partial.jsonl",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        metavar="RAW_JSON",
        help="пересчитать метрики по сохранённым ответам модели, без генерации",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    judge_choice = JudgeChoice(args.judge_provider, args.judge_model, args.judge_effort)
    if args.rescore:
        return _rescore(
            settings, args.rescore, args.label, not args.no_judge, judge_choice
        )
    if args.stage == "retrieval":
        return _retrieval_stage(
            settings,
            args.runs,
            args.label,
            args.sample,
            args.seed,
            args.categories or RETRIEVAL_CATEGORIES,
            not args.no_judge,
            not args.no_quantization,
            not args.no_filters,
            judge_choice,
        )
    label = args.label or ("holdout" if args.split == "holdout" else None)
    return _answer_stage(
        settings,
        args.runs,
        label,
        not args.no_judge,
        args.split,
        args.sample,
        args.seed,
        args.categories or ANSWER_CATEGORIES,
        args.resume,
        list(argv) if argv is not None else sys.argv[1:],
        args.mode,
        judge_choice,
    )


if __name__ == "__main__":
    sys.exit(main())
