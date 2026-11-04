"""Отчёт eval в markdown: параметры прогона, метрики с порогами, разбор кейсов.

Здесь только оформление: метрики считают `retrieval_metrics` и
`answer_metrics`, а запись файлов и код выхода — `run`. Отчёт этапа 1
(`render_report`, только поиск) остаётся для `--stage retrieval`.
"""

import math
import statistics
import subprocess
from collections import Counter
from collections.abc import Sequence
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
    citation_accuracy,
    cost_rub,
    expected_source_cited,
    false_refusal,
    filtered_out_share,
    language,
    must_include_found,
    must_not_include_violations,
    negative_control,
    outcomes,
    per_run,
    statement_coverage,
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
from evals.retrieval_metrics import (
    CaseResult,
    K,
    RetrievalRun,
    first_hit_rank,
    mrr_per_run,
    percentile,
    recall_per_run,
)

# Пороги из CLAUDE.md («Метрики и пороги»). Менять только с записью в отчёте.
THRESHOLDS = {"recall@5": 0.80}


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


def _instruction_cell(template: str | None) -> str:
    if not template:
        return "нет, вопрос уходит как есть"
    shown = template.format(task=RETRIEVAL_TASK, query="<вопрос>")
    # Перевод строки сломал бы таблицу markdown.
    return "`" + shown.replace("\n", "\\n") + "`"


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
) -> str:
    recall = recall_per_run(results)
    mrr = mrr_per_run(results)
    latencies = [lat for r in results for lat in r.latencies]
    passed = statistics.mean(recall) >= THRESHOLDS["recall@5"]
    cost = estimate_cost(manifest.embedding_model, stats.tokens)
    categories = Counter(r.case.category for r in results)
    chunks = ", ".join(f"{k} {v}" for k, v in sorted(manifest.chunks.items()))

    lines = [
        f"# Отчёт eval — {today.isoformat()}",
        "",
        "Этап 1: проверяется только поиск (LLM нет). Набор: `evals/dataset/howto_config.yaml`,",
        "его писал независимый субагент до появления кода поиска (`evals/dataset/README.md`).",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Коммит ассистента | {revision} |",
        f"| Paperless | {manifest.ref} ({manifest.commit or 'коммит неизвестен'}) |",
        f"| Модель эмбеддингов | {provider}: {manifest.embedding_model}, размерность {manifest.dim} |",
        f"| Инструкция к вопросу | {_instruction_cell(query_template)} |",
        f"| Индекс | {sum(manifest.chunks.values())} кусков: {chunks}; построен {manifest.built_at} |",
        f"| Кейсов | {len(results)} ({', '.join(f'{k} {v}' for k, v in sorted(categories.items()))}), "
        f"прогонов {len(results[0].ranks)} |",
        "| Промпт | не используется |",
        *_sample_row(sample),
        "",
        "## Метрики",
        "",
        "| Метрика | Значение | Порог | Статус |",
        "|---|---|---|---|",
        f"| recall@{K} | {_mean_std(recall)} | ≥ {THRESHOLDS['recall@5']:.2f} | "
        f"{'✅' if passed else '❌'} |",
        f"| MRR | {_mean_std(mrr)} | — | |",
        f"| задержка поиска p50 / p95 | {percentile(latencies, 0.5):.2f} / "
        f"{percentile(latencies, 0.95):.2f} с | показать | |",
        f"| токены запросов к API эмбеддингов | {stats.tokens}"
        f"{f' ({cost})' if cost else ''} | показать | |",
        f"| срабатывания маскера ПДн | {sum(stats.masked.values())} | — | |",
        "",
        "## По категориям и ролям",
        "",
        "| Группа | Кейсов | recall@5 |",
        "|---|---|---|",
    ]
    for label, key in (("категория", "category"), ("роль", "role")):
        for value in sorted({getattr(r.case, key) for r in results}):
            group = [r for r in results if getattr(r.case, key) == value]
            lines.append(
                f"| {label} {value} | {len(group)} | {statistics.mean(recall_per_run(group)):.2f} |"
            )

    lines += [
        "",
        "## По кейсам",
        "",
        "| id | роль | ранг по прогонам | топ-1 |",
        "|---|---|---|---|",
    ]
    for r in results:
        ranks = ", ".join(str(rank) if rank else "—" for rank in r.ranks)
        top1 = r.top[0][0] if r.top else "—"
        lines.append(f"| {r.case.id} | {r.case.role} | {ranks} | `{top1}` |")

    misses = [r for r in results if any(rank is None for rank in r.ranks)]
    lines += ["", f"## Промахи ({len(misses)})", ""]
    if not misses:
        lines.append("Нет.")
    for r in misses:
        lines += [
            f"### {r.case.id} — «{r.case.question}»",
            "",
            "Ожидалось: " + ", ".join(f"`{s}`" for s in r.case.expected_sources),
            "",
            "Найдено:",
            *[
                f"{n}. `{source}` ({score:.3f})"
                for n, (source, score) in enumerate(r.top, start=1)
            ],
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def report_path(directory: Path, today: date, label: str | None) -> Path:
    return directory / f"{today.isoformat()}{f'-{label}' if label else ''}.md"


# --- Этап 2: ответы со ссылками ---

ANSWER_THRESHOLDS = {
    "citation_accuracy": 0.90,
    "abstention": 0.90,
    # Не больше: доля отказов на вопросах из области.
    "false_refusal": 0.10,
    "groundedness": 0.85,
}


def retrieval_results(results: Sequence[AnswerCase]) -> list[CaseResult]:
    """Поиск из прогонов ассистента: тот же топ-5, что видела модель."""
    return [
        CaseResult(
            case=r.case,
            runs=[
                RetrievalRun(
                    top=run.top,
                    rank=first_hit_rank(
                        r.case.expected_sources, [s for s, _ in run.top]
                    ),
                    latency=run.retrieval_seconds,
                )
                for run in r.runs
            ],
        )
        for r in results
        if r.case.category in IN_SCOPE
    ]


def answer_gates(
    results: Sequence[AnswerCase], judging: Judging | None = None
) -> dict[str, bool]:
    """Метрики с порогом и прошли ли они его: по ним считается код выхода."""
    recall = statistics.mean(recall_per_run(retrieval_results(results)))
    gates = {
        "recall@5": recall >= THRESHOLDS["recall@5"],
        "citation_accuracy": citation_accuracy(results).value
        >= ANSWER_THRESHOLDS["citation_accuracy"],
        "abstention": abstention(results).value >= ANSWER_THRESHOLDS["abstention"],
        "false_refusal": false_refusal(results).value
        <= ANSWER_THRESHOLDS["false_refusal"],
    }
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


def _dataset_lines(split: str) -> list[str]:
    if split == "holdout":
        return [
            "Этап 2: ответы со ссылками на **отложенной выборке**",
            "`evals/dataset/holdout/holdout.yaml`. Её прогоняют один раз, когда модель,",
            "промпт и проверка уже выбраны на dev; кейсы писал независимый субагент.",
        ]
    return [
        "Этап 2: ответы со ссылками. Наборы: `evals/dataset/howto_config.yaml` и",
        "`out_of_scope.yaml`, их писали независимые субагенты (`evals/dataset/README.md`).",
    ]


def render_answer_report(
    results: Sequence[AnswerCase],
    *,
    meta: RunMeta,
    judging: Judging | None,
    today: date,
) -> str:
    manifest, llm_stats, llm_model = meta.manifest, meta.llm_stats, meta.llm_model
    retrieval = retrieval_results(results)
    recall = recall_per_run(retrieval)
    gates = answer_gates(results, judging)
    runs = [run for r in results for run in r.runs]
    api_models = sorted({run.answer.model for run in runs if run.answer.model})
    categories = Counter(r.case.category for r in results)
    chunks = ", ".join(f"{k} {v}" for k, v in sorted(manifest.chunks.items()))
    cost = cost_rub(api_models[0] if api_models else llm_model, llm_stats.total_tokens)
    accuracy = citation_accuracy(results)
    abstain = abstention(results)
    refusal = false_refusal(results)
    lang = language(results)

    lines = [
        f"# Отчёт eval — {today.isoformat()}",
        "",
        *_dataset_lines(meta.split),
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
        f"| Paperless | {manifest.ref} ({manifest.commit or 'коммит неизвестен'}) |",
        f"| Модель эмбеддингов | {meta.provider}: {manifest.embedding_model}, размерность {manifest.dim} |",
        f"| Инструкция к вопросу | {_instruction_cell(meta.query_template)} |",
        f"| Индекс | {sum(manifest.chunks.values())} кусков: {chunks}; построен {manifest.built_at} |",
        f"| LLM | {llm_model}; по ответам API: {', '.join(api_models) or '—'} |",
        f"| Температура | {meta.temperature} (GigaChat не принимает 0) |",
        f"| Промпт | `{meta.prompt_version}` |",
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
        f"| recall@{K} (howto, config) | {_mean_std(recall)} | ≥ {THRESHOLDS['recall@5']:.2f} | "
        f"{_status(gates['recall@5'])} |",
        f"| citation accuracy | {accuracy} {_spread(per_run(results, citation_accuracy))} | "
        f"≥ {ANSWER_THRESHOLDS['citation_accuracy']:.2f} | {_status(gates['citation_accuracy'])} |",
        f"| abstention (out_of_scope) | {abstain} {_spread(per_run(results, abstention))} | "
        f"≥ {ANSWER_THRESHOLDS['abstention']:.2f} | {_status(gates['abstention'])} |",
        f"| false refusal (howto, config) | {refusal} {_spread(per_run(results, false_refusal))} | "
        f"≤ {ANSWER_THRESHOLDS['false_refusal']:.2f} | {_status(gates['false_refusal'])} |",
        f"| покрытие утверждений цитатами | {statement_coverage(results)} | показать | |",
        f"| негативный контроль (цитата у соседнего источника) | {negative_control(results)} | ≈ 0 | |",
        f"| ответ ссылается на ожидаемый источник | {expected_source_cited(results)} | показать | |",
        f"| … если он был в топ-{K} поиска | {expected_source_cited(results, only_retrieved=True)} "
        "| показать | |",
        f"| out_of_scope: отказ дал фильтр, а не модель | {filtered_out_share(results)} | показать | |",
        f"| must_include в config (подстрокой) | {must_include_found(results)} | показать | |",
        f"| must_not_include нарушен | {must_not_include_violations(results)} | 0 | |",
        f"| доля кириллицы в ответе вне `кода` | {'—' if lang is None else f'{lang:.2f}'} "
        "| показать | |",
        f"| поиск p50 / p95 | {_p([run.retrieval_seconds for run in runs])} | показать | |",
        f"| генерация p50 / p95 | {_p([run.generation_seconds for run in runs if run.usage])} "
        "| показать | |",
        f"| токены LLM: промпт / ответ / из кэша | {llm_stats.prompt_tokens} / "
        f"{llm_stats.completion_tokens} / {llm_stats.precached_prompt_tokens} | показать | |",
        f"| стоимость по платному тарифу | "
        f"{'—' if cost is None else f'≈{cost:.2f} ₽ за прогон, ≈{cost / max(len(runs), 1):.3f} ₽ за ответ'}"
        " (на Freemium бесплатно) | показать | |",
        f"| срабатывания маскера ПДн (эмбеддинги / LLM) | {sum(meta.embedding_stats.masked.values())} / "
        f"{sum(llm_stats.masked.values())} | — | |",
        *_judge_rows(judging),
        "",
        "## Чем закончились прогоны",
        "",
        "| Группа | ответ | model_not_found | filtered_out | format_error | blacklist | length |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, group in (
        ("howto, config", IN_SCOPE),
        ("out_of_scope", {"out_of_scope"}),
    ):
        counts = outcomes(results, group)
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
    for label, group in (
        ("howto, config", IN_SCOPE),
        ("out_of_scope", {"out_of_scope"}),
    ):
        scores = top1_scores(results, group)
        if scores:
            lines.append(
                f"| {label} | {min(scores):.3f} | {statistics.median(scores):.3f} | {max(scores):.3f} |"
            )

    lines += [
        "",
        "## По кейсам",
        "",
        "| id | категория | итог по прогонам | цитаты: подтверждено / всего | ожидаемый источник в ответе |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        cited = [c for run in r.runs for s in run.checked for c in s.citations]
        expected = set(r.case.expected_sources)
        hits = (
            "—"
            if not expected
            else ", ".join(
                "да" if expected & run.cited_source_ids() else "нет" for run in r.runs
            )
        )
        lines.append(
            f"| {r.case.id} | {r.case.category} | {', '.join(_outcome_cell(run) for run in r.runs)} | "
            f"{sum(c.valid for c in cited)} / {len(cited)} | {hits} |"
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
                f"; нет в источнике: {', '.join(s.unknown_names)}"
                if s.unknown_names
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
        f"| must_include в howto (судья) | {must_include_judged(judging.verdicts)} | показать | |",
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
