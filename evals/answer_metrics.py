"""Метрики ответов: точность ссылок, отказы, язык, задержка и стоимость.

Все функции чистые: на входе результаты прогона, на выходе доля «k из n».
Кейсов немного (32 × 3 прогона), поэтому у каждой доли есть 95% интервал
Уилсона: он честно показывает, насколько цифре можно верить. Например,
11 из 12 — это 0.92, но интервал 0.65–0.99.

Метрики и пороги — из CLAUDE.md («Метрики и пороги»):
- tool correctness — верно ли решено, искать ли: на вопросах про рецепты
  поиск был и с непустым запросом, на болтовне поиска не было. В режиме
  always поиск есть всегда, поэтому болтовня там — промах по построению:
  так режимы и сравниваются;
- citation accuracy — доля подтверждённых цитат среди всех цитат модели
  до фильтра. Считается по ответам, где модель сказала «ответ есть»;
- abstention — доля честных отказов на out_of_scope: модель сама не нашла
  ответ, код отбросил все утверждения или модель ответила как на болтовню
  («я подбираю только рецепты»). Поломки (format_error, blacklist, length)
  честным отказом не считаются;
- false refusal — доля отказов на вопросах из области (любая причина).

Остальное показывается без порога: покрытие утверждений, негативный
контроль, цитирование ожидаемого источника, must_include для config,
must_not_include, язык ответа.
"""

import math
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from assistant.generation.answer import Answer, AnswerResult
from assistant.generation.citations import (
    CheckedCitation,
    CheckedStatement,
    Citation,
    SourceText,
    check_citation,
)
from assistant.generation.gigachat import LlmStats, Usage
from assistant.generation.review import Review
from assistant.tools.search_recipes import SearchArgs
from evals.dataset import Case

# Вопросы из области: на них ассистент должен отвечать, а не отказывать.
IN_SCOPE = frozenset(
    {"include", "exclude", "by_name", "semantic", "follow_up", "how_to_cook"}
)
# Где без поиска не ответить и где он не нужен (tool correctness). follow_up
# и out_of_scope не считаются: там верны оба решения.
SEARCH_NEEDED = frozenset({"include", "exclude", "by_name", "semantic", "how_to_cook"})
NO_SEARCH = frozenset({"chit_chat"})
HONEST_REFUSALS = frozenset({"model_not_found", "filtered_out"})
# Цена Claude, $ за 1M токенов: вход, выход, чтение кэша (официальные цены API,
# ноябрь 2026; кэш — 0,1 входа). Запись в кэш (1,25 входа) не учтена: она один
# раз на прогон.
PRICE_USD_PER_MTOK = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
}
# Цена платных тарифов GigaChat, ₽ за 1M токенов (ноябрь 2026). На Freemium
# бесплатно, цена нужна, чтобы оценить ответ в продакшене.
PRICE_RUB_PER_MTOK = {
    "GigaChat-2": 65.0,
    "GigaChat-2-Pro": 500.0,
    "GigaChat-2-Max": 650.0,
}


@dataclass(frozen=True)
class Rate:
    """Доля «k из n» с 95% интервалом Уилсона."""

    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else math.nan

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        if not self.n:
            return math.nan, math.nan
        p = self.value
        denominator = 1 + z * z / self.n
        center = (p + z * z / (2 * self.n)) / denominator
        spread = (
            z * math.sqrt(p * (1 - p) / self.n + z * z / (4 * self.n**2)) / denominator
        )
        return max(0.0, center - spread), min(1.0, center + spread)

    def __str__(self) -> str:
        if not self.n:
            return "—"
        low, high = self.wilson()
        return f"{self.value:.2f} ({self.k}/{self.n}, ДИ {low:.2f}–{high:.2f})"


@dataclass(frozen=True)
class FirstTry:
    """Первая попытка ответа, если её рецепты оказались не по вопросу и
    ассистент искал заново: запрос, найденное и что сказала проверка."""

    query: str
    top: list[tuple[str, float]]
    review: Review | None


@dataclass(frozen=True)
class AnswerRun:
    """Один прогон ассистента по одному кейсу."""

    answer: Answer
    # Все утверждения модели с результатом проверки, до фильтра.
    checked: list[CheckedStatement]
    # Что видела модель: номер, кусок, текст после маскирования.
    sources: list[SourceText]
    top: list[tuple[str, float]]
    retrieval_seconds: float
    generation_seconds: float
    usage: Usage | None
    finish_reason: str | None
    # Сырой ответ модели: для пересчёта и разбора.
    content: str
    # Поиски: в always — один разбор, в auto — сколько модель решила (может, ноль).
    searches: tuple[SearchArgs, ...] = ()
    # Проверка Sonnet перед отправкой; None — её не было. Итоговые утверждения — в final.
    review: Review | None = None
    # Первая попытка, если был новый поиск («рецепты не по вопросу»); иначе None.
    first_try: FirstTry | None = None

    @classmethod
    def from_result(cls, result: AnswerResult) -> AnswerRun:
        trace = result.trace
        completion = trace.completion
        return cls(
            answer=result.answer,
            checked=trace.checked,
            sources=trace.sources,
            top=[(h.chunk.source_id, h.score) for h in trace.hits],
            retrieval_seconds=trace.retrieval_seconds,
            generation_seconds=trace.generation_seconds,
            usage=completion.usage if completion else None,
            finish_reason=completion.finish_reason if completion else None,
            content=completion.content if completion else "",
            searches=trace.searches,
            review=trace.review,
            first_try=(
                FirstTry(
                    query=trace.first.args.query if trace.first.args else "",
                    top=[(h.chunk.source_id, h.score) for h in trace.first.hits],
                    review=trace.first.review,
                )
                if trace.first is not None
                else None
            ),
        )

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.answer.statements)

    def cited_source_ids(self) -> set[str]:
        """source_id кусков, на которые ссылается итоговый ответ."""
        by_number = {s.number: s.chunk.source_id for s in self.sources}
        return {
            by_number[c.source]
            for s in self.checked
            if s.kept
            for c in s.citations
            if c.valid
        }


@dataclass(frozen=True)
class AnswerCase:
    case: Case
    runs: list[AnswerRun]
    # Есть ли в индексе рецепты, на которые ссылается уточнение («сколько муки
    # во втором?»); False — на этом индексе кейс не проверить (выборка из
    # датасета), и в метрики он не идёт. None — не проверялось.
    available: bool | None = None


def needs_earlier_recipes(case: Case) -> bool:
    """Уточнение, ответ на которое — в рецепте из прошлого ответа."""
    return case.category == "follow_up" and bool(case.must_include)


def only_run(results: Sequence[AnswerCase], run: int) -> list[AnswerCase]:
    """Те же кейсы, но только с одним прогоном: для разброса по прогонам."""
    return [AnswerCase(r.case, [r.runs[run]], r.available) for r in results]


def per_run(
    results: Sequence[AnswerCase], metric: Callable[[Sequence[AnswerCase]], Rate]
) -> list[float]:
    runs = len(results[0].runs) if results else 0
    return [metric(only_run(results, i)).value for i in range(runs)]


def _runs(results: Iterable[AnswerCase], categories: Iterable[str]) -> list[AnswerRun]:
    """Прогоны кейсов этих категорий, кроме кейсов, которые на индексе не проверить."""
    wanted = set(categories)
    return [
        run
        for r in results
        if r.case.category in wanted and r.available is not False
        for run in r.runs
    ]


def _citations(results: Sequence[AnswerCase]) -> list[CheckedCitation]:
    return [
        c
        for run in _runs(results, {r.case.category for r in results})
        for s in run.checked
        for c in s.citations
    ]


def citation_accuracy(results: Sequence[AnswerCase]) -> Rate:
    citations = _citations(results)
    return Rate(sum(c.valid for c in citations), len(citations))


def statement_coverage(results: Sequence[AnswerCase]) -> Rate:
    """Доля утверждений модели хотя бы с одной подтверждённой цитатой."""
    statements = [
        s
        for run in _runs(results, {r.case.category for r in results})
        for s in run.checked
    ]
    return Rate(
        sum(any(c.valid for c in s.citations) for s in statements), len(statements)
    )


def negative_control(results: Sequence[AnswerCase]) -> Rate:
    """Те же цитаты, но приписанные соседнему источнику. Проверку должны пройти ≈ 0.

    Если проходят многие, проверка слишком мягкая: фрагмент находится где угодно.
    """
    k = n = 0
    for run in _runs(results, {r.case.category for r in results}):
        count = len(run.sources)
        if count < 2:
            continue
        for statement in run.checked:
            for c in statement.citations:
                if not 1 <= c.source <= count:
                    continue
                shifted = Citation(source=c.source % count + 1, quote=c.quote)
                k += check_citation(shifted, run.sources).valid
                n += 1
    return Rate(k, n)


def abstention(results: Sequence[AnswerCase]) -> Rate:
    runs = _runs(results, {"out_of_scope"})
    honest = sum(
        r.answer.status == "chat" or r.answer.reason in HONEST_REFUSALS for r in runs
    )
    return Rate(honest, len(runs))


def filtered_out_share(results: Sequence[AnswerCase]) -> Rate:
    """Доля out_of_scope, где отказ дал фильтр, а не сама модель."""
    runs = _runs(results, {"out_of_scope"})
    return Rate(sum(r.answer.reason == "filtered_out" for r in runs), len(runs))


def false_refusal(results: Sequence[AnswerCase]) -> Rate:
    """Не ответ на вопрос из области: «не нашёл» или ответ как на болтовню."""
    runs = _runs(results, IN_SCOPE)
    return Rate(sum(r.answer.status != "answer" for r in runs), len(runs))


def searched(run: AnswerRun) -> bool:
    return any(args.query.strip() for args in run.searches)


def tool_correctness(results: Sequence[AnswerCase]) -> Rate:
    """Решение «искать или нет» верно: поиск с запросом там, где нужен, и без
    поиска на болтовне."""
    needed = _runs(results, SEARCH_NEEDED)
    unneeded = _runs(results, NO_SEARCH)
    right = sum(searched(r) for r in needed) + sum(not r.searches for r in unneeded)
    return Rate(right, len(needed) + len(unneeded))


def chat_handled(results: Sequence[AnswerCase]) -> Rate:
    """На болтовню — короткий ответ без утверждений о рецептах (status chat)."""
    runs = _runs(results, NO_SEARCH)
    return Rate(sum(r.answer.status == "chat" for r in runs), len(runs))


def must_include_found(results: Sequence[AnswerCase]) -> Rate:
    """Факты из must_include в тексте ответа, подстрокой с учётом регистра.

    Это точные слова (название, ингредиент), их можно проверить кодом.
    Факты «своими словами» у semantic проверяет судья.
    """
    k = n = 0
    for r in results:
        if r.available is False:
            continue
        for run in r.runs:
            for fact in r.case.must_include:
                n += 1
                k += fact in run.text
    return Rate(k, n)


def must_not_include_violations(results: Sequence[AnswerCase]) -> Rate:
    """Прогоны, где в ответе есть запрещённое (например, выдуманный параметр)."""
    k = n = 0
    for r in results:
        if not r.case.must_not_include:
            continue
        for run in r.runs:
            n += 1
            text = run.text.lower()
            k += any(item.lower() in text for item in r.case.must_not_include)
    return Rate(k, n)


_CODE_SPAN = re.compile(r"`[^`]*`")


def cyrillic_share(text: str) -> float | None:
    """Доля кириллицы среди букв вне `кода`: ответ должен быть по-русски."""
    letters = [ch for ch in _CODE_SPAN.sub("", text) if ch.isalpha()]
    if not letters:
        return None
    return sum("а" <= ch.lower() <= "я" or ch in "ёЁ" for ch in letters) / len(letters)


def language(results: Sequence[AnswerCase]) -> float | None:
    shares = [
        share
        for run in _runs(results, {r.case.category for r in results})
        if (share := cyrillic_share(run.text)) is not None
    ]
    return statistics.mean(shares) if shares else None


def outcomes(results: Sequence[AnswerCase], categories: Iterable[str]) -> Counter[str]:
    """Чем закончились прогоны: answer или причина отказа."""
    return Counter(run.answer.reason or "answer" for run in _runs(results, categories))


def top1_scores(
    results: Sequence[AnswerCase], categories: Iterable[str]
) -> list[float]:
    return [run.top[0][1] for run in _runs(results, categories) if run.top]


def cost_usd(model: str, stats: LlmStats) -> float | None:
    """Стоимость прогона у Claude по токенам: вход без кэша, кэш и выход по своим ценам."""
    price = PRICE_USD_PER_MTOK.get(model)
    if price is None:
        return None
    uncached = stats.prompt_tokens
    return (
        uncached * price[0]
        + stats.completion_tokens * price[1]
        + stats.precached_prompt_tokens * price[2]
    ) / 1_000_000


def cost_rub(model: str, tokens: int) -> float | None:
    """Стоимость по платному тарифу. Модель из ответа API: «GigaChat-2-Max:2.0.30.01»."""
    price = PRICE_RUB_PER_MTOK.get(model.split(":")[0])
    return None if price is None else price * tokens / 1_000_000
