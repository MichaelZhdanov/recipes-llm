"""Отчёт eval в markdown: параметры прогона, метрики с порогами, разбор кейсов.

Здесь только оформление: метрики считают `retrieval_metrics` и
`answer_metrics`, а запись файлов и код выхода — `run`. Отчёт поиска
(`render_report`) — для `--stage retrieval`.
"""

import math
import statistics
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from assistant.generation.gigachat import LlmStats
from assistant.indexing.build import estimate_cost
from assistant.retrieval.embeddings import RETRIEVAL_TASK, EmbeddingStats
from assistant.retrieval.store import IndexManifest
from evals.answer_metrics import (
    IN_SCOPE,
    AnswerCase,
    AnswerRun,
    abstention,
    chat_handled,
    citation_accuracy,
    cost_rub,
    cost_usd,
    false_refusal,
    filtered_out_share,
    language,
    must_include_found,
    must_not_include_violations,
    negative_control,
    outcomes,
    per_run,
    statement_coverage,
    tool_correctness,
    top1_scores,
)
from evals.judge import (
    Judging,
    canary_scores,
    clarity,
    clarity_canary_scores,
    groundedness,
    must_include_judged,
    unjudged,
)
from evals.relevance import Relevance, relevance_canary_scores, semantic_relevance
from evals.retrieval_metrics import (
    CODE_CHECKED,
    CaseResult,
    K,
    RetrievalRun,
    category_mean,
    conditions,
    conditions_score,
    dropped_names,
    errors,
    exclude_violations,
    has_conditions,
    include_precision,
    name_hit,
    percentile,
    primary,
    run_score,
)


def _sample_row(sample: str | None) -> list[str]:
    if sample is None:
        return []
    return [f"| Выборка кейсов | {sample} — быстрый прогон, цифры ориентировочные |"]


def _mean_std(values: Sequence[float]) -> str:
    return f"{statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}"


def git_revision(cwd: Path | None = None) -> str:
    """Короткий SHA и пометка, если отслеживаемые файлы изменены.

    Неотслеживаемые файлы (черновики, заметки) на код не влияют, поэтому
    отчёт ими не помечается.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "неизвестен"
    return f"{sha} (есть незакоммиченные изменения)" if dirty else sha


def _index_cell(manifest: IndexManifest) -> str:
    size = (
        f"выборка {manifest.sample} из всего датасета, сид {manifest.seed}"
        if manifest.sample
        else "весь датасет"
    )
    quantization = manifest.quantization or "без сжатия"
    return (
        f"{manifest.recipes} рецептов ({size}), векторы {quantization}; "
        f"построен {manifest.built_at}"
    )


def _instruction_cell(template: str | None) -> str:
    if not template:
        return "нет, вопрос уходит как есть"
    shown = template.format(task=RETRIEVAL_TASK, query="<вопрос>")
    # Перевод строки сломал бы таблицу markdown.
    return "`" + shown.replace("\n", "\\n") + "`"


def _value(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _mark(value: bool | None) -> str:
    return {True: "✅ ", False: "❌ ", None: "? "}[value]


# Пороги поиска (этап 2): записаны до первого прогона с фильтрами,
# менять только с записью в отчёте. Код выхода 1, если какой-то не пройден.
RETRIEVAL_THRESHOLDS = {
    "conditions": 0.90,
    # Не больше: доля топа с исключённым ингредиентом.
    "exclude_violations": 0.05,
    "name_hit": 0.85,
    "semantic_relevance": 0.80,
}


def retrieval_gates(
    results: Sequence[CaseResult], relevance: Relevance | None = None
) -> dict[str, bool]:
    """Метрики поиска с порогом и прошли ли они его. Метрики без кейсов порога не имеют."""
    t = RETRIEVAL_THRESHOLDS
    gates: dict[str, bool] = {}
    if (value := conditions(results)) is not None:
        gates["conditions"] = value >= t["conditions"]
    if (value := exclude_violations(results)) is not None:
        gates["exclude_violations"] = value <= t["exclude_violations"]
    if (value := name_hit(results)) is not None:
        gates["name_hit"] = value >= t["name_hit"]
    if relevance is not None and (rate := semantic_relevance(relevance)).n:
        gates["semantic_relevance"] = rate.value >= t["semantic_relevance"]
    return gates


def _gate(gates: dict[str, bool], name: str, sign: str) -> str:
    threshold = f"{sign} {RETRIEVAL_THRESHOLDS[name]:.2f}"
    return f"{threshold} {_status(gates[name])}" if name in gates else threshold


@dataclass(frozen=True)
class Parsing:
    """Как разбирались вопросы: для шапки отчёта поиска."""

    # Модель из настроек и модели с версией из ответов API.
    model: str
    served: tuple[str, ...]
    prompt_version: str
    vocabulary: int
    stats: LlmStats
    # Повторы запроса после зацикливания (finish_reason: length).
    retries: int = 0
    provider: str = "gigachat"
    # Сколько думает Claude (LLM_EFFORT); пусто — по умолчанию модели.
    effort: str = ""


def _parsing_cell(parsing: Parsing | None) -> str:
    if parsing is None:
        return "нет (--no-filters): только векторы по вопросу, как на этапе 1"
    served = ", ".join(parsing.served) or "—"
    if parsing.provider == "anthropic":
        effort = parsing.effort or "по умолчанию модели"
        call = (
            f"вызов search_recipes с tool_choice auto, effort {effort}, "
            "temperature не задаётся (API её не принимает)"
        )
    else:
        call = "принудительный вызов search_recipes, temperature 1e-4"
    return (
        f"{parsing.model} (из ответа API: {served}), промпт `{parsing.prompt_version}`, "
        f"{call}; словарь индекса — {parsing.vocabulary} имён"
    )


def _plan_line(run: RetrievalRun) -> str:
    if run.error is not None:
        return f"сбой разбора: {run.error}"
    if run.args is None:
        return "без разбора"
    parts = [f"запрос «{run.args.query}»"]
    if run.args.conditions.include:
        groups = (" или ".join(g) for g in run.args.conditions.include)
        parts.append(f"с: {'; '.join(groups)}")
    if run.args.conditions.exclude:
        parts.append(f"без: {', '.join(run.args.conditions.exclude)}")
    if run.args.dropped:
        parts.append(f"нет в словаре: {', '.join(run.args.dropped)}")
    return " · ".join(parts)


def _relevance_rows(relevance: Relevance | None, gates: dict[str, bool]) -> list[str]:
    if relevance is None:
        return []
    caught, accepted = relevance_canary_scores(relevance)
    return [
        f"| semantic_relevance@{K}: доля подходящих рецептов в топе (semantic, судья) "
        f"| {semantic_relevance(relevance)} | {_gate(gates, 'semantic_relevance', '≥')} |",
        f"| канарейки: судья отверг неподходящие / принял подходящие | {caught} / {accepted} | — |",
        f"| неразобранных ответов судьи, токены судьи | {relevance.failures}, {relevance.tokens} | — |",
    ]


def _found_lines(r: CaseResult) -> list[str]:
    top = r.runs[-1].top if r.runs else []
    return [
        f"{n}. {f.name} ({f.score:.3f}) — {', '.join(f.ingredients)}"
        for n, f in enumerate(top, start=1)
    ]


def render_report(
    results: Sequence[CaseResult],
    *,
    manifest: IndexManifest,
    revision: str,
    provider: str,
    query_template: str | None,
    stats: EmbeddingStats,
    today: date,
    sample: str | None = None,
    relevance: Relevance | None = None,
    quantization: bool = True,
    parsing: Parsing | None = None,
) -> str:
    latencies = [lat for r in results for lat in r.latencies]
    cost = estimate_cost(manifest.embedding_model, stats.tokens)
    categories = Counter(r.case.category for r in results)
    unavailable = [r for r in results if r.available is False]
    gates = retrieval_gates(results, relevance)
    dropped = dropped_names(results)
    masked = sum(stats.masked.values()) + (
        sum(parsing.stats.masked.values()) if parsing else 0
    )

    lines = [
        f"# Отчёт eval — {today.isoformat()}",
        "",
        "Проверяется только поиск, ответа модели нет; как разбирался вопрос — строка",
        "«Разбор вопроса». Наборы — `evals/dataset/`, кто и как их составлял —",
        "`evals/dataset/README.md`.",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Коммит ассистента | {revision} |",
        f"| Данные | {manifest.dataset}, ревизия {manifest.ref[:8]} |",
        f"| Модель эмбеддингов | {provider}: {manifest.embedding_model}, размерность {manifest.dim} |",
        f"| Инструкция к вопросу | {_instruction_cell(query_template)} |",
        f"| Индекс | {_index_cell(manifest)} |",
        *(
            []
            if quantization
            else [
                "| Поиск | по точным векторам, сжатые не используются (--no-quantization) |"
            ]
        ),
        f"| Разбор вопроса | {_parsing_cell(parsing)} |",
        f"| Кейсов | {len(results)} ({', '.join(f'{k} {v}' for k, v in sorted(categories.items()))}), "
        f"прогонов {len(results[0].runs) if results else 0} |",
        f"| Судья | {f'{relevance.model}, рубрика `{relevance.prompt_version}`' if relevance else 'не запускался'} |",
        *_sample_row(sample),
        "",
        "## Метрики",
        "",
        "Считает код по топ-5 поиска, `semantic_relevance` — судья. Кейсы без подходящего",
        f"рецепта в индексе в метрику не идут (их {len(unavailable)}, список ниже).",
        "Пороги записаны до первого прогона с фильтрами (`RETRIEVAL_THRESHOLDS`).",
        "",
        "| Метрика | Значение | Порог |",
        "|---|---|---|",
        f"| **conditions@{K}: доля топа, где выполнены все условия кейса (include, exclude, follow_up)** "
        f"| **{_value(conditions(results))}** | {_gate(gates, 'conditions', '≥')} |",
        f"| exclude_violations@{K}: доля топа с исключённым ингредиентом (exclude, меньше — лучше) "
        f"| {_value(exclude_violations(results))} | {_gate(gates, 'exclude_violations', '≤')} |",
        f"| name_hit@{K}: в топе есть рецепт с нужным словом в названии (by_name) "
        f"| {_value(name_hit(results))} | {_gate(gates, 'name_hit', '≥')} |",
        f"| include_precision@{K}: доля топа, где есть все нужные ингредиенты (include) "
        f"| {_value(include_precision(results))} | — |",
        f"| M этапа 1: среднее name_hit@{K} и include_precision@{K} | {_value(primary(results))} | — |",
        *_relevance_rows(relevance, gates),
        f"| имена из разбора не из словаря (отброшены) | {len(dropped)} | — |",
        f"| сбои разбора (выдача пустая) | {errors(results)} | — |",
        *(
            [f"| повторы разбора после зацикливания (length) | {parsing.retries} | — |"]
            if parsing
            else []
        ),
        f"| задержка поиска p50 / p95 | {percentile(latencies, 0.5):.2f} / "
        f"{percentile(latencies, 0.95):.2f} с | — |",
        *(
            [
                f"| токены LLM на разбор: вход и выход без кэша / вход из кэша / выход "
                f"| {parsing.stats.total_tokens} / {parsing.stats.precached_prompt_tokens} / "
                f"{parsing.stats.completion_tokens} | — |"
            ]
            if parsing
            else []
        ),
        f"| токены запросов к API эмбеддингов | {stats.tokens}{f' ({cost})' if cost else ''} | — |",
        f"| срабатывания маскера ПДн | {masked} | — |",
        "",
        "## По категориям",
        "",
        "Метрика категории: by_name — name_hit, include — include_precision,",
        "exclude — exclude_violations (меньше — лучше), follow_up — conditions.",
        "",
        "| Категория | Кейсов | Метрика |",
        "|---|---|---|",
    ]
    for category in sorted(categories):
        value = category_mean(results, category) if category in CODE_CHECKED else None
        lines.append(f"| {category} | {categories[category]} | {_value(value)} |")

    if unavailable:
        lines += ["", "## Кейсы без подходящего рецепта в индексе", ""]
        lines += [f"- {r.case.id} — «{r.case.question}»" for r in unavailable]

    lines += [
        "",
        "## По кейсам",
        "",
        "| id | категория | стиль | метрика по прогонам | conditions по прогонам | топ-1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        scores = ", ".join(_value(run_score(r.case, run)) for run in r.runs)
        met = ", ".join(_value(conditions_score(r.case, run)) for run in r.runs)
        top1 = r.runs[-1].top[0].name if r.runs and r.runs[-1].top else "—"
        lines.append(
            f"| {r.case.id} | {r.case.category} | {r.case.style} | {scores} | {met} | {top1} |"
        )

    if parsing is not None:
        lines += ["", "## Разбор вопросов (последний прогон)", ""]
        lines += [
            f"- {r.case.id} «{r.case.question}» → {_plan_line(r.runs[-1])}"
            for r in results
            if r.runs
        ]

    # Промах: у кейса с условиями в топе есть рецепт, который их не выполняет;
    # у by_name в топе нет ни одного рецепта с нужным названием.
    def missed(r: CaseResult) -> bool:
        if r.available is False or not r.runs:
            return False
        if has_conditions(r.case):
            met = conditions_score(r.case, r.runs[-1])
            return met is not None and met < 1.0
        return run_score(r.case, r.runs[-1]) == 0.0

    misses = [r for r in results if missed(r)]
    lines += ["", f"## Промахи ({len(misses)})", ""]
    if not misses:
        lines.append("Нет.")
    for r in misses:
        expected = []
        if r.case.expected_name:
            expected.append(f"название: {', '.join(r.case.expected_name)}")
        expected += [f"с: {' или '.join(group)}" for group in r.case.expected_include]
        if r.case.expected_exclude:
            expected.append(f"без: {', '.join(r.case.expected_exclude)}")
        lines += [
            f"### {r.case.id} — «{r.case.question}»",
            "",
            "Ожидалось: " + "; ".join(expected),
            "",
            f"Разбор: {_plan_line(r.runs[-1])}",
            "",
            "Найдено:",
            *_found_lines(r),
            "",
        ]

    semantic = [r for r in results if r.case.category == "semantic"]
    if semantic:
        lines += ["", "## Вопросы по смыслу: что нашлось", ""]
        if relevance:
            lines += [
                "✅ — судья счёл рецепт подходящим, ❌ — нет, ? — без вердикта.",
                "",
            ]
        for r in semantic:
            marks = relevance.marks.get(r.case.id, []) if relevance else []
            found = [
                f"{_mark(marks[n]) if n < len(marks) else ''}{line}"
                for n, line in enumerate(_found_lines(r))
            ]
            lines += [f"### {r.case.id} — «{r.case.question}»", "", *found, ""]
    if relevance and relevance.canaries:
        lines += ["", "## Канарейки судьи релевантности", ""]
        lines += [
            f"- {c.id}: «{c.question}» → {c.name}: ожидалось "
            f"{'подходит' if c.relevant else 'не подходит'}, судья — {_mark(m)}"
            for c, m in zip(relevance.canaries, relevance.canary_marks, strict=True)
        ]
    return "\n".join(lines).rstrip() + "\n"


def report_path(directory: Path, today: date, label: str | None) -> Path:
    return directory / f"{today.isoformat()}{f'-{label}' if label else ''}.md"


# --- Этап 2: ответы со ссылками ---

ANSWER_THRESHOLDS = {
    "tool_correctness": 0.90,
    "citation_accuracy": 0.90,
    "abstention": 0.90,
    # Не больше: доля отказов на вопросах из области.
    "false_refusal": 0.10,
    "groundedness": 0.85,
}


def answer_gates(
    results: Sequence[AnswerCase], judging: Judging | None = None
) -> dict[str, bool]:
    """Метрики с порогом и прошли ли они его: по ним считается код выхода."""
    gates: dict[str, bool] = {}
    if (tools := tool_correctness(results)).n:
        gates["tool_correctness"] = tools.value >= ANSWER_THRESHOLDS["tool_correctness"]
    gates["citation_accuracy"] = (
        citation_accuracy(results).value >= ANSWER_THRESHOLDS["citation_accuracy"]
    )
    # Прогон по части категорий (--categories): у метрики без кейсов порога нет.
    if (abstain := abstention(results)).n:
        gates["abstention"] = abstain.value >= ANSWER_THRESHOLDS["abstention"]
    if (refusal := false_refusal(results)).n:
        gates["false_refusal"] = refusal.value <= ANSWER_THRESHOLDS["false_refusal"]
    if judging is not None:
        gates["groundedness"] = (
            groundedness(judging.verdicts).value >= ANSWER_THRESHOLDS["groundedness"]
        )
    return gates


@dataclass(frozen=True)
class RunMeta:
    """Параметры прогона для шапки отчёта. Пишутся и в сырые данные — для --rescore."""

    manifest: IndexManifest
    revision: str
    provider: str
    query_template: str | None
    embedding_stats: EmbeddingStats
    llm_stats: LlmStats
    llm_model: str
    temperature: float
    prompt_version: str
    # Пересчёт по сохранённому прогону: откуда взяты ответы модели.
    rescored_from: str | None = None
    # dev — основной набор, holdout — отложенная выборка.
    split: str = "dev"
    # Быстрый прогон на части кейсов: «8 из 32, сид …»; None — все кейсы.
    sample: str | None = None
    # always — сначала поиск; auto — модель решает (решение 7).
    mode: str = "always"
    llm_provider: str = "gigachat"


def _spread(values: Sequence[float]) -> str:
    finite = [v for v in values if not math.isnan(v)]
    if len(finite) < 2:
        return ""
    return f"по прогонам {min(finite):.2f}–{max(finite):.2f}"


def _status(passed: bool) -> str:
    return "✅" if passed else "❌"


def _p(values: Sequence[float]) -> str:
    if not values:
        return "—"
    return f"{percentile(values, 0.5):.2f} / {percentile(values, 0.95):.2f} с"


def _outcome_cell(run: AnswerRun) -> str:
    return "ответ" if run.answer.reason is None else run.answer.reason


def _dataset_lines(split: str, categories: Iterable[str]) -> list[str]:
    names = ", ".join(sorted(categories))
    if split == "holdout":
        return [
            f"Ответы со ссылками на **отложенной выборке** (`evals/dataset/holdout/`): {names}.",
            "Её прогоняют один раз, когда модель, промпт и проверка уже выбраны на dev.",
        ]
    return [
        f"Ответы со ссылками. Категории: {names}; наборы в `evals/dataset/`,",
        "кто и как их составлял — `evals/dataset/README.md`.",
    ]


def _prompts_cell(runs: Sequence[AnswerRun], default: str) -> str:
    """Версии промптов из самих ответов: прогон по сохранённым ответам может их смешать."""
    used = Counter(
        f"`{run.answer.prompt_version}`" for run in runs if run.answer.prompt_version
    )
    if not used:
        return f"`{default}`"
    return ", ".join(f"{name} — {n}" for name, n in sorted(used.items()))


_MODES = {
    "always": "always — «всегда ищем»: разбор вопроса, поиск, ответ",
    "auto": "auto — «модель решает»: модель сама вызывает search_recipes",
}

# Группы для таблиц исходов и score: только те, где есть кейсы.
_GROUPS = (
    ("вопросы из области", IN_SCOPE),
    ("out_of_scope", frozenset({"out_of_scope"})),
)


def render_answer_report(
    results: Sequence[AnswerCase],
    *,
    meta: RunMeta,
    judging: Judging | None,
    today: date,
) -> str:
    manifest, llm_stats, llm_model = meta.manifest, meta.llm_stats, meta.llm_model
    gates = answer_gates(results, judging)
    runs = [run for r in results for run in r.runs]
    api_models = sorted({run.answer.model for run in runs if run.answer.model})
    categories = Counter(r.case.category for r in results)
    cost = cost_rub(api_models[0] if api_models else llm_model, llm_stats.total_tokens)
    dollars = cost_usd(api_models[0] if api_models else llm_model, llm_stats)
    tools = tool_correctness(results)
    accuracy = citation_accuracy(results)
    abstain = abstention(results)
    refusal = false_refusal(results)
    lang = language(results)

    lines = [
        f"# Отчёт eval — {today.isoformat()}",
        "",
        *_dataset_lines(meta.split, categories),
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Коммит ассистента | {meta.revision} |",
        *(
            [
                f"| Пересчёт | ответы модели из `{meta.rescored_from}`, ссылки и судья — заново |"
            ]
            if meta.rescored_from
            else []
        ),
        f"| Данные | {manifest.dataset}, ревизия {manifest.ref[:8]} |",
        f"| Модель эмбеддингов | {meta.provider}: {manifest.embedding_model}, размерность {manifest.dim} |",
        f"| Инструкция к вопросу | {_instruction_cell(meta.query_template)} |",
        f"| Индекс | {_index_cell(manifest)} |",
        f"| LLM | {llm_model}; по ответам API: {', '.join(api_models) or '—'} |",
        f"| Режим | {_MODES.get(meta.mode, meta.mode)} |",
        (
            "| Температура | не задаётся: API Anthropic на версии 2026-10-21 её не принимает |"
            if meta.llm_provider == "anthropic"
            else f"| Температура | {meta.temperature} (GigaChat не принимает 0) |"
        ),
        f"| Промпт | {_prompts_cell(runs, meta.prompt_version)} |",
        f"| Судья | {f'{judging.model}, рубрика `{judging.prompt_version}`' if judging else 'не запускался'} |",
        *_sample_row(meta.sample),
        f"| Кейсов | {len(results)} ({', '.join(f'{k} {v}' for k, v in sorted(categories.items()))}), "
        f"прогонов {len(results[0].runs) if results else 0} |",
        "",
        "## Метрики",
        "",
        "Доли — «k из n» по всем прогонам, ДИ — 95% интервал Уилсона.",
        "",
        "| Метрика | Значение | Порог | Статус |",
        "|---|---|---|---|",
        *(
            [
                f"| tool correctness: искал, где нужно, и не искал на болтовне | {tools} "
                f"{_spread(per_run(results, tool_correctness))} | "
                f"≥ {ANSWER_THRESHOLDS['tool_correctness']:.2f} | {_status(gates['tool_correctness'])} |"
            ]
            if "tool_correctness" in gates
            else []
        ),
        f"| citation accuracy | {accuracy} {_spread(per_run(results, citation_accuracy))} | "
        f"≥ {ANSWER_THRESHOLDS['citation_accuracy']:.2f} | {_status(gates['citation_accuracy'])} |",
        *(
            [
                f"| abstention (out_of_scope) | {abstain} {_spread(per_run(results, abstention))} | "
                f"≥ {ANSWER_THRESHOLDS['abstention']:.2f} | {_status(gates['abstention'])} |"
            ]
            if "abstention" in gates
            else []
        ),
        *(
            [
                f"| false refusal (вопросы из области) | {refusal} {_spread(per_run(results, false_refusal))} | "
                f"≤ {ANSWER_THRESHOLDS['false_refusal']:.2f} | {_status(gates['false_refusal'])} |"
            ]
            if "false_refusal" in gates
            else []
        ),
        f"| болтовня: короткий ответ без утверждений (chit_chat) | {chat_handled(results)} | показать | |",
        f"| покрытие утверждений цитатами | {statement_coverage(results)} | показать | |",
        f"| негативный контроль (цитата у соседнего источника) | {negative_control(results)} | ≈ 0 | |",
        f"| out_of_scope: отказ дал фильтр, а не модель | {filtered_out_share(results)} | показать | |",
        f"| must_include (подстрокой) | {must_include_found(results)} | показать | |",
        f"| must_not_include нарушен | {must_not_include_violations(results)} | 0 | |",
        f"| доля кириллицы в ответе вне `кода` | {'—' if lang is None else f'{lang:.2f}'} "
        "| показать | |",
        f"| поиск p50 / p95 | {_p([run.retrieval_seconds for run in runs])} | показать | |",
        f"| генерация p50 / p95 | {_p([run.generation_seconds for run in runs if run.usage])} "
        "| показать | |",
        f"| токены LLM: промпт / ответ / из кэша | {llm_stats.prompt_tokens} / "
        f"{llm_stats.completion_tokens} / {llm_stats.precached_prompt_tokens} | показать | |",
        (
            f"| стоимость (цены API Anthropic) | ≈${dollars:.3f} за прогон, "
            f"≈${dollars / max(len(runs), 1):.5f} за ответ | показать | |"
            if dollars is not None
            else f"| стоимость по платному тарифу | "
            f"{'—' if cost is None else f'≈{cost:.2f} ₽ за прогон, ≈{cost / max(len(runs), 1):.3f} ₽ за ответ'}"
            " (на Freemium бесплатно) | показать | |"
        ),
        f"| срабатывания маскера ПДн (эмбеддинги / LLM) | {sum(meta.embedding_stats.masked.values())} / "
        f"{sum(llm_stats.masked.values())} | — | |",
        *_judge_rows(judging),
        "",
        "## Чем закончились прогоны",
        "",
        "| Группа | ответ | model_not_found | filtered_out | format_error | blacklist | length |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, group in _GROUPS:
        counts = outcomes(results, group)
        if not counts:
            continue
        lines.append(
            f"| {label} | "
            + " | ".join(
                str(counts[key])
                for key in (
                    "answer",
                    "model_not_found",
                    "filtered_out",
                    "format_error",
                    "blacklist",
                    "length",
                )
            )
            + " |"
        )

    lines += [
        "",
        "## Score топ-1 поиска",
        "",
        "Порога по score нет. Распределение — чтобы решать о нём осознанно.",
        "",
        "| Группа | min | медиана | max |",
        "|---|---|---|---|",
    ]
    for label, group in _GROUPS:
        scores = top1_scores(results, group)
        if scores:
            lines.append(
                f"| {label} | {min(scores):.3f} | {statistics.median(scores):.3f} | {max(scores):.3f} |"
            )

    lines += [
        "",
        "## По кейсам",
        "",
        "| id | категория | итог по прогонам | цитаты: подтверждено / всего |",
        "|---|---|---|---|",
    ]
    for r in results:
        cited = [c for run in r.runs for s in run.checked for c in s.citations]
        lines.append(
            f"| {r.case.id} | {r.case.category} | {', '.join(_outcome_cell(run) for run in r.runs)} | "
            f"{sum(c.valid for c in cited)} / {len(cited)} |"
        )

    lines += _problem_sections(results)
    lines += _judge_sections(results, judging)
    return "\n".join(lines).rstrip() + "\n"


def _problem_sections(results: Sequence[AnswerCase]) -> list[str]:
    """Разбор того, что пошло не так: отказы в области, ответы вне области, плохие цитаты."""
    refused = [
        (r, run)
        for r in results
        if r.case.category in IN_SCOPE
        for run in r.runs
        if run.answer.status == "not_found"
    ]
    answered = [
        (r, run)
        for r in results
        if r.case.category == "out_of_scope"
        for run in r.runs
        if run.answer.status == "answer"
    ]
    bad = [
        (r.case.id, c)
        for r in results
        for run in r.runs
        for s in run.checked
        for c in s.citations
        if not c.valid
    ]
    lines = ["", f"## Отказы на вопросах из области ({len(refused)})", ""]
    if not refused:
        lines.append("Нет.")
    for r, run in refused:
        lines.append(f"- {r.case.id} ({run.answer.reason}) — «{r.case.question}»")
        for s in run.checked:
            errors = "; ".join(c.error or "ок" for c in s.citations) or "без цитат"
            names = (
                f"; чисел нет в цитатах: {', '.join(s.unknown_numbers)}"
                if s.unknown_numbers
                else ""
            )
            lines.append(f"  - «{s.text}» — {errors}{names}")
    lines += ["", f"## Ответы вне области ({len(answered)})", ""]
    if not answered:
        lines.append("Нет.")
    for r, run in answered:
        lines.append(f"- {r.case.id} — «{r.case.question}»")
        lines += [f"  - «{s.text}»" for s in run.answer.statements]
    broken = [
        (r.case.id, run)
        for r in results
        for run in r.runs
        if run.answer.reason in ("format_error", "blacklist", "length")
    ]
    lines += ["", f"## Поломки: format_error, blacklist, length ({len(broken)})", ""]
    if not broken:
        lines.append("Нет.")
    for case_id, run in broken:
        preview = " ".join(run.content.split())[:120]
        lines.append(
            f"- {case_id}: {run.answer.reason}, finish_reason={run.finish_reason}, "
            f"{len(run.content)} символов: «{preview}»"
        )
    lines += ["", f"## Неподтверждённые цитаты ({len(bad)})", ""]
    if not bad:
        lines.append("Нет.")
    reasons = Counter(c.error for _, c in bad)
    lines += [f"- {error}: {count}" for error, count in reasons.most_common()]
    if bad:
        lines += ["", "Первые 10:", ""]
        lines += [
            f"- {case_id}, источник {c.source}: «{c.quote}» — {c.error}"
            for case_id, c in bad[:10]
        ]
    return lines


def _judge_rows(judging: Judging | None) -> list[str]:
    if judging is None:
        return []
    caught, accepted = canary_scores(judging.canaries, judging.canary_marks)
    grounded = groundedness(judging.verdicts)
    passed = grounded.value >= ANSWER_THRESHOLDS["groundedness"]
    return [
        f"| groundedness (судья) | {grounded} | ≥ {ANSWER_THRESHOLDS['groundedness']:.2f} "
        f"| {_status(passed)} |",
        f"| утверждения грамотны и понятны (судья) | {clarity(judging.verdicts)} | показать | |",
        f"| must_include в semantic (судья) | {must_include_judged(judging.verdicts)} | показать | |",
        f"| канарейки: судья поймал неверные | {caught} | все | |",
        f"| канарейки: судья принял верные | {accepted} | все | |",
        *_clarity_canary_rows(judging),
        f"| пункты без вердикта судьи | {unjudged(judging.verdicts)}, "
        f"неразобранных ответов {judging.failures} | 0 | |",
        f"| токены судьи | {judging.stats.total_tokens} | показать | |",
    ]


def _clarity_canary_rows(judging: Judging) -> list[str]:
    if not judging.clarity_canaries:
        return []
    caught, accepted = clarity_canary_scores(
        judging.clarity_canaries, judging.clarity_marks
    )
    return [
        f"| канарейки грамотности: судья поймал плохие тексты | {caught} | все | |",
        f"| канарейки грамотности: судья принял хорошие | {accepted} | все | |",
    ]


def _judge_sections(
    results: Sequence[AnswerCase], judging: Judging | None
) -> list[str]:
    if judging is None:
        return []
    unsupported = []
    for r in results:
        for number, run in enumerate(r.runs):
            verdict = judging.verdicts.get((r.case.id, number))
            if verdict is None:
                continue
            kept = [s for s in run.checked if s.kept]
            for index, mark in enumerate(verdict.supported):
                if mark is False:
                    comment = (
                        verdict.comments[index] if index < len(verdict.comments) else ""
                    )
                    unsupported.append(
                        f"- {r.case.id}, прогон {number + 1}: «{kept[index].text}» — {comment}"
                    )
    missed = [
        f"- canary {c.id} (supported={c.supported}): судья сказал {mark}"
        for c, mark in zip(judging.canaries, judging.canary_marks, strict=True)
        if mark is not c.supported
    ]
    unclear = [
        f"- {r.case.id}, прогон {number + 1}: «{text}» — {'; '.join(problems)}"
        for r in results
        for number, run in enumerate(r.runs)
        if (verdict := judging.verdicts.get((r.case.id, number))) is not None
        for text, mark, problems in zip(
            [s.text for s in run.checked if s.kept],
            verdict.clear,
            verdict.problems or [[]] * len(verdict.clear),
            strict=False,
        )
        if mark is False
    ]
    lines = ["", f"## Судья: неподтверждённые утверждения ({len(unsupported)})", ""]
    lines += unsupported or ["Нет."]
    lines += [
        "",
        f"## Судья: неграмотные или непонятные утверждения ({len(unclear)})",
        "",
    ]
    lines += unclear or ["Нет."]
    lines += ["", f"## Судья: ошибки на канарейках ({len(missed)})", ""]
    lines += missed or ["Нет."]
    if judging.clarity_canaries:
        missed_clarity = [
            f"- {c.id} (clear={c.clear}): судья сказал {mark}"
            for c, mark in zip(
                judging.clarity_canaries, judging.clarity_marks, strict=True
            )
            if mark is not c.clear
        ]
        lines += [
            "",
            f"## Судья: ошибки на канарейках грамотности ({len(missed_clarity)})",
            "",
        ]
        lines += missed_clarity or ["Нет."]
    return lines
