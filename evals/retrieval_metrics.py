"""Метрики поиска: recall@k, MRR и задержка.

- recall@k — доля кейсов, где хотя бы один ожидаемый источник попал в топ-k;
- MRR — среднее 1/ранг первого ожидаемого источника (0, если его нет в топ-k);
- задержка поиска — эмбеддинг вопроса плюс запрос к Qdrant.

Сеть нужна только `run_cases`, остальное — чистые функции над результатами.
"""

import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from assistant.indexing.chunks import Chunk
from assistant.retrieval.search import Retriever
from evals.dataset import Case, parse_line

K = 5


@dataclass(frozen=True)
class RetrievalRun:
    """Один прогон поиска по одному кейсу."""

    # Источники топ-K и их score: для разбора промахов.
    top: list[tuple[str, float]]
    # Ранг первого ожидаемого источника (с 1), None — его нет в топ-K.
    rank: int | None
    latency: float
    # Сколько кусков в топе нашёл точный поиск по тексту ошибки.
    exact: int = 0


@dataclass(frozen=True)
class CaseResult:
    case: Case
    runs: list[RetrievalRun]

    @property
    def ranks(self) -> list[int | None]:
        return [run.rank for run in self.runs]

    @property
    def latencies(self) -> list[float]:
        return [run.latency for run in self.runs]

    @property
    def top(self) -> list[tuple[str, float]]:
        """Топ последнего прогона: поиск детерминирован, прогоны совпадают."""
        return self.runs[-1].top


def first_hit_rank(expected: Collection[str], retrieved: Sequence[str]) -> int | None:
    for rank, source_id in enumerate(retrieved, start=1):
        if source_id in expected:
            return rank
    return None


def recall_per_run(results: Sequence[CaseResult], k: int = K) -> list[float]:
    runs = len(results[0].ranks)
    return [
        sum(1 for r in results if (rank := r.ranks[run]) is not None and rank <= k)
        / len(results)
        for run in range(runs)
    ]


def mrr_per_run(results: Sequence[CaseResult]) -> list[float]:
    runs = len(results[0].ranks)
    return [
        sum(1 / rank if (rank := r.ranks[run]) else 0.0 for r in results) / len(results)
        for run in range(runs)
    ]


def percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def hit_rank(case: Case, chunks: Sequence[Chunk]) -> int | None:
    """Ранг первого подходящего куска (с 1) или None.

    У ошибки подходящий кусок — тот, где лежит строка expected_line: место
    ошибки известно до строки, а не до раздела. У остальных — по expected_sources.
    """
    if case.expected_line is not None:
        path, line = parse_line(case.expected_line)
        for rank, chunk in enumerate(chunks, start=1):
            if chunk.path == path and chunk.start_line <= line <= chunk.end_line:
                return rank
        return None
    return first_hit_rank(case.expected_sources, [c.source_id for c in chunks])


def missing_sources(
    cases: Sequence[Case],
    has_source: Callable[[str], bool],
    has_line: Callable[[str, int], bool] = lambda _path, _line: True,
) -> list[str]:
    """Кейсы, ссылающиеся на источник, которого нет в индексе: такой кейс не засчитается никогда."""
    missing = [
        f"{c.id}: {s}" for c in cases for s in c.expected_sources if not has_source(s)
    ]
    missing += [
        f"{c.id}: {c.expected_line}"
        for c in cases
        if c.expected_line is not None and not has_line(*parse_line(c.expected_line))
    ]
    return missing


def run_cases(
    cases: Sequence[Case],
    retriever: Retriever,
    *,
    runs: int,
    k: int = K,
    clock: Callable[[], float] = time.perf_counter,
) -> list[CaseResult]:
    results = []
    for case in cases:
        case_runs = []
        for _ in range(runs):
            started = clock()
            hits = retriever.search(case.question, k=k)
            latency = clock() - started
            top = [(h.chunk.source_id, h.score) for h in hits]
            rank = hit_rank(case, [h.chunk for h in hits])
            exact = sum(h.exact for h in hits)
            case_runs.append(
                RetrievalRun(top=top, rank=rank, latency=latency, exact=exact)
            )
        results.append(CaseResult(case=case, runs=case_runs))
    return results


def has_expectation(case: Case) -> bool:
    """Есть ли у кейса, что искать: у out_of_scope ожидаемого источника нет."""
    return bool(case.expected_sources) or case.expected_line is not None


def exact_share(results: Sequence[CaseResult]) -> float:
    """Доля прогонов, где сработал точный поиск. На howto и config — ложные срабатывания."""
    runs = [run for r in results for run in r.runs]
    return sum(run.exact > 0 for run in runs) / len(runs) if runs else 0.0
