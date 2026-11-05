"""Метрики поиска рецептов. Считает код, без LLM.

- `name_hit@5` (кейсы by_name) — есть ли в топ-5 рецепт, в названии
  которого есть одно из слов `expected_name`;
- `include_precision@5` (кейсы include) — доля рецептов топ-5, где есть
  все группы `expected_include` (из группы хватает одного имени);
- `exclude_violations@5` (кейсы exclude) — доля рецептов топ-5 с
  исключённым ингредиентом. Векторный поиск «без» не понимает, это
  базовая линия, которую исправят фильтры этапа 2;
- задержка поиска — эмбеддинг вопроса плюс запрос к Qdrant.

Кейс, для которого в индексе нет ни одного подходящего рецепта, в метрику
не идёт: модель тут ни при чём (так бывает на выборке из датасета).

Сеть нужна только `run_cases`, остальное — чистые функции над результатами.
"""

import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from assistant.indexing.chunks import recipe_chunk
from assistant.indexing.dataset import Recipe
from assistant.retrieval.search import Retriever
from evals.dataset import Case

K = 5
# Категории, у которых метрику поиска считает код.
CODE_CHECKED = frozenset({"by_name", "include", "exclude"})


@dataclass(frozen=True)
class Found:
    """Рецепт из выдачи поиска: ровно то, что нужно метрикам и отчёту."""

    recipe_id: int
    name: str
    ingredients: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class RetrievalRun:
    top: list[Found]
    latency: float


@dataclass(frozen=True)
class CaseResult:
    case: Case
    runs: list[RetrievalRun]
    # Есть ли в индексе хоть один подходящий рецепт; None — не проверялось.
    available: bool | None = None

    @property
    def latencies(self) -> list[float]:
        return [run.latency for run in self.runs]


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е").strip()


def name_matches(name: str, words: Iterable[str]) -> bool:
    lowered = _norm(name)
    return any(_norm(word) in lowered for word in words)


def include_matches(
    ingredients: Iterable[str], groups: Sequence[Sequence[str]]
) -> bool:
    have = {_norm(item) for item in ingredients}
    return all(any(_norm(name) in have for name in group) for group in groups)


def exclude_violated(ingredients: Iterable[str], exclude: Iterable[str]) -> bool:
    have = {_norm(item) for item in ingredients}
    return any(_norm(name) in have for name in exclude)


def fits(case: Case, name: str, ingredients: Sequence[str]) -> bool:
    """Подходит ли рецепт под все условия кейса: название, «с чем» и «без чего»."""
    if case.expected_name and not name_matches(name, case.expected_name):
        return False
    if not include_matches(ingredients, case.expected_include):
        return False
    return not exclude_violated(ingredients, case.expected_exclude)


def run_score(case: Case, run: RetrievalRun) -> float | None:
    """Метрика одного прогона кейса или None, если для категории её нет."""
    top = run.top[:K]
    if not top:
        return 0.0 if case.category in CODE_CHECKED else None
    if case.category == "by_name":
        return float(any(name_matches(f.name, case.expected_name) for f in top))
    if case.category == "include":
        return statistics.mean(
            include_matches(f.ingredients, case.expected_include) for f in top
        )
    if case.category == "exclude":
        return statistics.mean(
            exclude_violated(f.ingredients, case.expected_exclude) for f in top
        )
    return None


def category_mean(results: Sequence[CaseResult], category: str) -> float | None:
    """Среднее по прогонам кейсов категории, у которых в индексе есть ответ."""
    scores = [
        score
        for r in results
        if r.case.category == category and r.available is not False
        for run in r.runs
        if (score := run_score(r.case, run)) is not None
    ]
    return statistics.mean(scores) if scores else None


def name_hit(results: Sequence[CaseResult]) -> float | None:
    return category_mean(results, "by_name")


def include_precision(results: Sequence[CaseResult]) -> float | None:
    return category_mean(results, "include")


def exclude_violations(results: Sequence[CaseResult]) -> float | None:
    return category_mean(results, "exclude")


def primary(results: Sequence[CaseResult]) -> float | None:
    """Основная метрика выбора модели: среднее name_hit@5 и include_precision@5."""
    parts = [
        m for m in (name_hit(results), include_precision(results)) if m is not None
    ]
    return statistics.mean(parts) if parts else None


def percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)] if ordered else 0.0


def mark_available(
    results: Sequence[CaseResult], recipes: Sequence[Recipe]
) -> list[CaseResult]:
    """Отмечает кейсы, для которых в проиндексированных рецептах есть подходящий."""
    chunks = [recipe_chunk(r, "-") for r in recipes]
    marked = []
    for r in results:
        if r.case.category not in CODE_CHECKED:
            marked.append(r)
            continue
        available = any(fits(r.case, c.name, c.ingredients) for c in chunks)
        marked.append(replace(r, available=available))
    return marked


def unknown_ingredients(cases: Sequence[Case], vocabulary: set[str]) -> list[str]:
    """Имена из кейсов, которых нет в словаре датасета: такой кейс не засчитается никогда."""
    known = {_norm(name) for name in vocabulary}
    return [
        f"{c.id}: {name}"
        for c in cases
        for name in [*(n for g in c.expected_include for n in g), *c.expected_exclude]
        if _norm(name) not in known
    ]


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
            top = [
                Found(
                    recipe_id=h.chunk.recipe_id,
                    name=h.chunk.name,
                    ingredients=tuple(h.chunk.ingredients),
                    score=h.score,
                )
                for h in hits
            ]
            case_runs.append(RetrievalRun(top=top, latency=latency))
        results.append(CaseResult(case=case, runs=case_runs))
    return results
