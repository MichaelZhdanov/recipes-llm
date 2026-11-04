"""Метрики ответов: точность ссылок, отказы, язык, задержка и стоимость.

Все функции чистые: на входе результаты прогона, на выходе доля «k из n».
Кейсов немного (32 × 3 прогона), поэтому у каждой доли есть 95% интервал
Уилсона: он честно показывает, насколько цифре можно верить. Например,
11 из 12 — это 0.92, но интервал 0.65–0.99.

Метрики и пороги — из CLAUDE.md («Метрики и пороги»):
- citation accuracy — доля подтверждённых цитат среди всех цитат модели
  до фильтра. Считается по ответам, где модель сказала «ответ есть»;
- abstention — доля честных отказов на out_of_scope: модель сама не нашла
  ответ или код отбросил все утверждения. Поломки (format_error, blacklist,
  length) честным отказом не считаются;
- false refusal — доля отказов на howto и config (любая причина).

Остальное показывается без порога: покрытие утверждений, негативный
контроль, цитирование ожидаемого источника, must_include для config,
must_not_include, язык ответа.
"""

import math
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from assistant.generation.answer import Answer, AnswerResult
from assistant.generation.citations import (
    CheckedCitation,
    CheckedStatement,
    Citation,
    SourceText,
    check_citation,
)
from assistant.generation.gigachat import Usage
from evals.dataset import Case

IN_SCOPE = frozenset({"howto", "config"})
HONEST_REFUSALS = frozenset({"model_not_found", "filtered_out"})
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
    # Неверные переводы терминов, исправленные кодом.
    term_fixes: list[tuple[str, str]] = field(default_factory=list)

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
            term_fixes=trace.term_fixes,
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


def only_run(results: Sequence[AnswerCase], run: int) -> list[AnswerCase]:
    """Те же кейсы, но только с одним прогоном: для разброса по прогонам."""
    return [AnswerCase(r.case, [r.runs[run]]) for r in results]


def per_run(
    results: Sequence[AnswerCase], metric: Callable[[Sequence[AnswerCase]], Rate]
) -> list[float]:
    runs = len(results[0].runs) if results else 0
    return [metric(only_run(results, i)).value for i in range(runs)]


def _runs(results: Iterable[AnswerCase], categories: Iterable[str]) -> list[AnswerRun]:
    wanted = set(categories)
    return [run for r in results if r.case.category in wanted for run in r.runs]


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
    return Rate(sum(r.answer.reason in HONEST_REFUSALS for r in runs), len(runs))


def filtered_out_share(results: Sequence[AnswerCase]) -> Rate:
    """Доля out_of_scope, где отказ дал фильтр, а не сама модель."""
    runs = _runs(results, {"out_of_scope"})
    return Rate(sum(r.answer.reason == "filtered_out" for r in runs), len(runs))


def false_refusal(results: Sequence[AnswerCase]) -> Rate:
    runs = _runs(results, IN_SCOPE)
    return Rate(sum(r.answer.status == "not_found" for r in runs), len(runs))


def expected_source_cited(
    results: Sequence[AnswerCase], *, only_retrieved: bool = False
) -> Rate:
    """Доля прогонов howto/config, где ответ ссылается на ожидаемый источник.

    only_retrieved — только прогоны, где ожидаемый источник был в топе
    поиска: так видно, что теряет генерация, а что — поиск.
    """
    k = n = 0
    for r in results:
        if r.case.category not in IN_SCOPE:
            continue
        expected = set(r.case.expected_sources)
        for run in r.runs:
            if only_retrieved and not expected & {s for s, _ in run.top}:
                continue
            n += 1
            k += bool(expected & run.cited_source_ids())
    return Rate(k, n)


def must_include_found(results: Sequence[AnswerCase], category: str = "config") -> Rate:
    """Факты из must_include в тексте ответа, подстрокой с учётом регистра.

    Для config это точные имена параметров, их можно проверить кодом.
    У howto факты сформулированы по-русски, их проверяет судья.
    """
    k = n = 0
    for r in results:
        if r.case.category != category:
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


def term_fixes(results: Sequence[AnswerCase]) -> Counter[str]:
    """Сколько раз код исправил неверный перевод термина: «объёма → тома»."""
    return Counter(
        f"{wrong} → {fixed}"
        for run in _runs(results, {r.case.category for r in results})
        for wrong, fixed in run.term_fixes
    )


def cost_rub(model: str, tokens: int) -> float | None:
    """Стоимость по платному тарифу. Модель из ответа API: «GigaChat-2-Max:2.0.30.01»."""
    price = PRICE_RUB_PER_MTOK.get(model.split(":")[0])
    return None if price is None else price * tokens / 1_000_000
