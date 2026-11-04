import dataclasses
from datetime import date
from typing import Literal

import pytest
from eval_helpers import MANIFEST

from assistant.generation.answer import Answer, AnswerStatement, Reason
from assistant.generation.citations import CheckedCitation, CheckedStatement, SourceText
from assistant.generation.gigachat import LlmStats, Usage
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import EmbeddingStats
from evals.answer_metrics import (
    AnswerCase,
    AnswerRun,
    Rate,
    abstention,
    citation_accuracy,
    cost_rub,
    cyrillic_share,
    expected_source_cited,
    false_refusal,
    must_include_found,
    must_not_include_violations,
    negative_control,
    per_run,
    statement_coverage,
    term_fixes,
)
from evals.dataset import Case
from evals.report import RunMeta, answer_gates, render_answer_report

Category = Literal["howto", "config", "out_of_scope"]


def meta(llm_stats: LlmStats) -> RunMeta:
    return RunMeta(
        manifest=MANIFEST,
        revision="abc1234",
        provider="cloud.ru",
        query_template=None,
        embedding_stats=EmbeddingStats(),
        llm_stats=llm_stats,
        llm_model="GigaChat-2",
        temperature=1e-4,
        prompt_version="answer_v1",
    )


def chunk(anchor: str, text: str) -> Chunk:
    return Chunk(
        kind="doc",
        path="docs/usage.md",
        start_line=1,
        end_line=1 + text.count("\n"),
        title=anchor,
        breadcrumb=f"Usage > {anchor}",
        source_id=f"docs/usage.md#{anchor}",
        url="https://x/blob/v3.4.2/docs/usage.md?plain=1#L1-L2",
        site_url=None,
        ref="v3.4.2",
        text=text,
    )


SOURCES = [
    SourceText(
        1,
        chunk("mail", "Paperless can fetch attachments from IMAP mail accounts."),
        "Paperless can fetch attachments from IMAP mail accounts.",
    ),
    SourceText(
        2,
        chunk("trash", "Deleted documents go to the trash and can be restored."),
        "Deleted documents go to the trash and can be restored.",
    ),
]


def statement(
    text: str,
    *valid: bool,
    source: int = 1,
    quote: str = "fetch attachments from IMAP mail",
) -> CheckedStatement:
    return CheckedStatement(
        text=text,
        citations=[
            CheckedCitation(
                source,
                quote,
                None if ok else "фрагмента нет в источнике",
                1 if ok else None,
                1 if ok else None,
            )
            for ok in valid
        ],
        unknown_names=[],
    )


def run(
    reason: Reason | None = None,
    checked: list[CheckedStatement] | None = None,
    text: str = "Почта забирается из IMAP.",
    top: str = "docs/usage.md#mail",
) -> AnswerRun:
    checked = (
        checked
        if checked is not None
        else ([statement(text, True)] if reason is None else [])
    )
    return AnswerRun(
        answer=Answer(
            status="answer" if reason is None else "not_found",
            reason=reason,
            statements=[AnswerStatement(text=text, sources=[1])]
            if reason is None
            else [],
            sources=[],
            model="GigaChat-2:2.0",
            prompt_version="answer_v1",
            paperless_ref="v3.4.2",
        ),
        checked=checked,
        sources=SOURCES,
        top=[(top, 0.5)],
        retrieval_seconds=0.2,
        generation_seconds=1.5,
        usage=Usage(1000, 100, 0, 1100),
        finish_reason="stop",
        content="{}",
    )


def case(
    case_id: str,
    category: Category = "howto",
    sources: list[str] | None = None,
    must_include: list[str] | None = None,
    must_not_include: list[str] | None = None,
) -> Case:
    return Case(
        id=case_id,
        category=category,
        role="user",
        question="?",
        expected_behavior="abstain" if category == "out_of_scope" else "answer",
        expected_sources=[]
        if category == "out_of_scope"
        else (sources or ["docs/usage.md#mail"]),
        must_include=must_include or [],
        must_not_include=must_not_include or [],
    )


def test_rate_with_wilson_interval() -> None:
    rate = Rate(11, 12)

    low, high = rate.wilson()
    assert rate.value == pytest.approx(0.917, abs=1e-3)
    assert (round(low, 2), round(high, 2)) == (0.65, 0.99)
    assert str(rate) == "0.92 (11/12, ДИ 0.65–0.99)"
    assert str(Rate(0, 0)) == "—"


def test_citation_accuracy_counts_all_citations_before_filter() -> None:
    results = [
        AnswerCase(
            case("howto-001"),
            [run(checked=[statement("a", True, False), statement("b", False)])],
        )
    ]

    assert citation_accuracy(results) == Rate(1, 3)
    assert statement_coverage(results) == Rate(1, 2)


def test_negative_control_moves_quote_to_another_source() -> None:
    results = [AnswerCase(case("howto-001"), [run()])]

    # Цитата из источника 1, приписанная источнику 2, не подтверждается.
    assert negative_control(results) == Rate(0, 1)


def test_abstention_counts_only_honest_refusals() -> None:
    results = [
        AnswerCase(
            case("out_of_scope-001", "out_of_scope"),
            [
                run("model_not_found"),
                run("filtered_out"),
                run("format_error"),
                run(None),
            ],
        )
    ]

    assert abstention(results) == Rate(2, 4)


def test_false_refusal_counts_any_not_found_in_scope() -> None:
    results = [
        AnswerCase(case("howto-001"), [run(None), run("model_not_found")]),
        AnswerCase(case("config-001", "config"), [run(None), run("length")]),
        AnswerCase(
            case("out_of_scope-001", "out_of_scope"),
            [run("model_not_found"), run("model_not_found")],
        ),
    ]

    assert false_refusal(results) == Rate(2, 4)
    assert per_run(results, false_refusal) == [0.0, 1.0]


def test_expected_source_cited_overall_and_when_retrieved() -> None:
    results = [
        AnswerCase(
            case("howto-001", sources=["docs/usage.md#mail"]),
            [run(None), run("model_not_found"), run(None, top="docs/usage.md#other")],
        )
    ]

    assert expected_source_cited(results) == Rate(2, 3)
    # В третьем прогоне источника не было в топе: генерация тут ни при чём.
    assert expected_source_cited(results, only_retrieved=True) == Rate(1, 2)


def test_must_include_is_exact_and_must_not_include_ignores_case() -> None:
    results = [
        AnswerCase(
            case(
                "config-001", "config", must_include=["PAPERLESS_OCR_LANGUAGE", "rus"]
            ),
            [run(text="Задайте `PAPERLESS_OCR_LANGUAGE`.")],
        ),
        AnswerCase(
            case(
                "out_of_scope-001", "out_of_scope", must_not_include=["PAPERLESS_S3_"]
            ),
            [run(text="Задайте paperless_s3_bucket."), run("model_not_found")],
        ),
    ]

    assert must_include_found(results) == Rate(1, 2)
    assert must_not_include_violations(results) == Rate(1, 2)


def test_cyrillic_share_ignores_code() -> None:
    assert cyrillic_share("Задайте `PAPERLESS_OCR_LANGUAGE=rus`.") == 1.0
    assert cyrillic_share("Set the language") == 0.0
    assert cyrillic_share("`код`") is None


def test_cost_uses_model_without_version() -> None:
    assert cost_rub("GigaChat-2-Max:2.0.30.01", 1_000_000) == 650.0
    assert cost_rub("GigaChat-3-Ultra:1", 1000) is None


def test_answer_report_and_gates() -> None:
    results = [
        AnswerCase(case("howto-001"), [run(None), run(None)]),
        AnswerCase(
            case("out_of_scope-001", "out_of_scope"),
            [run("model_not_found"), run(None)],
        ),
    ]

    report = render_answer_report(
        results,
        meta=meta(LlmStats(total_tokens=4400)),
        judging=None,
        today=date(2026, 11, 4),
    )

    assert "| LLM | GigaChat-2; по ответам API: GigaChat-2:2.0 |" in report
    assert "| abstention (out_of_scope) | 0.50 (1/2," in report
    assert "## Ответы вне области (1)" in report
    assert "| out_of_scope-001 | out_of_scope | model_not_found, ответ |" in report
    assert answer_gates(results) == {
        "recall@5": True,
        "citation_accuracy": True,
        "abstention": False,
        "false_refusal": True,
    }


def test_report_shows_broken_answers_with_model_output() -> None:
    broken = run("format_error")
    results = [AnswerCase(case("howto-001"), [broken])]

    report = render_answer_report(
        results, meta=meta(LlmStats()), judging=None, today=date(2026, 11, 4)
    )

    assert "## Поломки: format_error, blacklist, length (1)" in report
    assert "- howto-001: format_error, finish_reason=stop, 2 символов: «{}»" in report


def test_holdout_report_says_so() -> None:
    results = [AnswerCase(case("holdout-001"), [run(None)])]
    holdout = dataclasses.replace(meta(LlmStats()), split="holdout")

    report = render_answer_report(
        results, meta=holdout, judging=None, today=date(2026, 11, 4)
    )

    assert "**отложенной выборке**" in report


def test_term_fixes_are_counted() -> None:
    fixed = dataclasses.replace(
        run(None), term_fixes=[("объёме", "томе"), ("объёма", "тома")]
    )
    results = [AnswerCase(case("howto-001"), [fixed, run(None)])]

    assert term_fixes(results) == {"объёме → томе": 1, "объёма → тома": 1}
