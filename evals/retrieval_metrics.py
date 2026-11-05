"""Метрики поиска рецептов. Считает код, без LLM.

- `name_hit@5` (кейсы by_name) — есть ли в топ-5 рецепт, в названии
  которого есть одно из слов `expected_name`;
- `include_precision@5` (кейсы include) — доля рецептов топ-5, где есть
  все группы `expected_include` (из группы хватает одного имени);
- `exclude_violations@5` (кейсы exclude) — доля рецептов топ-5 с
  исключённым ингредиентом. Векторный поиск «без» не понимает: без
  фильтров (`--no-filters`) это базовая линия этапа 1;
- `conditions@5` (кейсы include, exclude и follow_up с условиями) — доля
  рецептов топ-5, где выполнены **все** условия кейса: название, «с чем»
  и «без чего». Главная метрика этапа 2: проверяет путь целиком — разбор
  вопроса, фильтр и векторы;
- задержка поиска — разбор вопроса GigaChat, эмбеддинг запроса и Qdrant.

Кейс, для которого в индексе нет ни одного подходящего рецепта, в метрику
не идёт: модель тут ни при чём (так бывает на выборке из датасета).

Сеть нужна только функции поиска, которую получает `run_cases`, остальное —
чистые функции над результатами.
"""

import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from assistant.generation.gigachat import LlmError, Message
from assistant.indexing.chunks import recipe_chunk
from assistant.indexing.dataset import Recipe
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import Hit
from assistant.tools.search_recipes import SearchArgs, SearchPlanner
from evals.dataset import Case

K = 5
# Категории, у которых метрику поиска считает код.
CODE_CHECKED = frozenset({"by_name", "include", "exclude", "follow_up"})
# Категории, где у кейса бывают условия для conditions@5.
CONDITION_CATEGORIES = frozenset({"include", "exclude", "follow_up"})

# Поиск по кейсу: во что разобран вопрос (None — без разбора) и выдача.
Search = Callable[[Case], tuple[SearchArgs | None, list[Hit]]]


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
    # Во что GigaChat разобрал вопрос; None — поиск без разбора (--no-filters).
    args: SearchArgs | None = None
    # Сбой разбора (GigaChat не вызвал функцию, недоступен): выдача пустая.
    error: str | None = None


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


def has_conditions(case: Case) -> bool:
    return case.category in CONDITION_CATEGORIES and bool(
        case.expected_include or case.expected_exclude or case.expected_name
    )


def conditions_score(case: Case, run: RetrievalRun) -> float | None:
    """Доля топа, где выполнены все условия кейса; None — у кейса условий нет."""
    if not has_conditions(case):
        return None
    top = run.top[:K]
    return (
        statistics.mean(fits(case, f.name, f.ingredients) for f in top) if top else 0.0
    )


def run_score(case: Case, run: RetrievalRun) -> float | None:
    """Метрика одного прогона кейса или None, если для категории её нет."""
    if case.category == "follow_up":
        return conditions_score(case, run)
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


def conditions(results: Sequence[CaseResult]) -> float | None:
    """conditions@5 по всем прогонам кейсов с условиями, у которых в индексе есть ответ."""
    scores = [
        score
        for r in results
        if r.available is not False
        for run in r.runs
        if (score := conditions_score(r.case, run)) is not None
    ]
    return statistics.mean(scores) if scores else None


def dropped_names(results: Sequence[CaseResult]) -> list[tuple[str, str]]:
    """(кейс, имя): имена из разбора, которых нет в словаре индекса."""
    pairs = (
        (r.case.id, name)
        for r in results
        for run in r.runs
        if run.args is not None
        for name in run.args.dropped
    )
    return list(dict.fromkeys(pairs))


def errors(results: Sequence[CaseResult]) -> int:
    return sum(run.error is not None for r in results for run in r.runs)


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
        if r.case.category not in CODE_CHECKED or (
            r.case.category == "follow_up" and not has_conditions(r.case)
        ):
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


def history(case: Case) -> list[Message]:
    """Реплики диалога из кейса в формате сообщений модели."""
    return [{"role": turn.role, "content": turn.text} for turn in case.turns]


def filtered_search(planner: SearchPlanner, retriever: Retriever) -> Search:
    """Поиск этапа 2: разбор вопроса с историей диалога, фильтр и векторы по query."""

    def search(case: Case) -> tuple[SearchArgs | None, list[Hit]]:
        args = planner.plan(case.question, history(case))
        return args, retriever.search(args.query, k=K, conditions=args.conditions)

    return search


def vector_search(retriever: Retriever) -> Search:
    """Базовая линия этапа 1: только векторы по вопросу, без разбора и истории."""

    def search(case: Case) -> tuple[SearchArgs | None, list[Hit]]:
        return None, retriever.search(case.question, k=K)

    return search


def run_cases(
    cases: Sequence[Case],
    search: Search,
    *,
    runs: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[CaseResult]:
    """Прогоняет поиск по кейсам. Сбой разбора у одного кейса не роняет прогон:
    его выдача пустая (метрика 0), а причина попадает в отчёт."""
    results = []
    for case in cases:
        case_runs = []
        for _ in range(runs):
            started = clock()
            try:
                args, hits = search(case)
            except LlmError as error:
                case_runs.append(
                    RetrievalRun(top=[], latency=clock() - started, error=str(error))
                )
                continue
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
            case_runs.append(RetrievalRun(top=top, latency=latency, args=args))
        results.append(CaseResult(case=case, runs=case_runs))
    return results
