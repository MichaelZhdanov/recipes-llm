import dataclasses
from datetime import date

import pytest
from chunk_helpers import text_chunk
from eval_helpers import MANIFEST
from eval_helpers import case as make_case

from assistant.generation.answer import Answer, AnswerStatement, Reason
from assistant.generation.citations import CheckedCitation, CheckedStatement, SourceText
from assistant.generation.gigachat import LlmStats, Usage
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import EmbeddingStats
from assistant.retrieval.filters import IngredientFilter
from assistant.tools.search_recipes import SearchArgs
from evals.answer_metrics import (
    AnswerCase,
    AnswerRun,
    Rate,
    abstention,
    chat_handled,
    citation_accuracy,
    cost_rub,
    cost_usd,
    cyrillic_share,
    false_refusal,
    must_include_found,
    must_not_include_violations,
    negative_control,
    per_run,
    statement_coverage,
    tool_correctness,
)
from evals.dataset import Case
from evals.report import RunMeta, answer_gates, render_answer_report


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


# Номер рецепта вместо якоря документации: источник «recipe/<номер>».
RECIPE_IDS = {"mail": 1, "trash": 2, "timezone": 3}


def chunk(anchor: str, text: str) -> Chunk:
    return text_chunk(
        text, recipe_id=RECIPE_IDS.setdefault(anchor, len(RECIPE_IDS) + 1)
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
        unknown_numbers=[],
    )


SEARCH = SearchArgs("курица с рисом", IngredientFilter())


def run(
    reason: Reason | None = None,
    checked: list[CheckedStatement] | None = None,
    text: str = "Почта забирается из IMAP.",
    top: str = "recipe/1",
    searches: tuple[SearchArgs, ...] = (SEARCH,),
    chat: bool = False,
) -> AnswerRun:
    checked = (
        checked
        if checked is not None
        else ([statement(text, True)] if reason is None else [])
    )
    return AnswerRun(
        answer=Answer(
            status="chat" if chat else "answer" if reason is None else "not_found",
            reason=reason,
            statements=[AnswerStatement(text=text, sources=[1])]
            if reason is None and not chat
            else [],
            sources=[],
            model="GigaChat-2:2.0",
            prompt_version="answer_v1",
            dataset_revision="v3.4.2",
        ),
        checked=checked,
        sources=SOURCES,
        top=[(top, 0.5)],
        retrieval_seconds=0.2,
        generation_seconds=1.5,
        usage=Usage(1000, 100, 0, 1100),
        finish_reason="stop",
        content="{}",
        searches=searches,
    )


def case(
    case_id: str,
    category: str = "by_name",
    must_include: list[str] | None = None,
    must_not_include: list[str] | None = None,
) -> Case:
    return make_case(
        case_id,
        category,
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
            case("by_name-001"),
            [run(checked=[statement("a", True, False), statement("b", False)])],
        )
    ]

    assert citation_accuracy(results) == Rate(1, 3)
    assert statement_coverage(results) == Rate(1, 2)


def test_negative_control_moves_quote_to_another_source() -> None:
    results = [AnswerCase(case("by_name-001"), [run()])]

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
        AnswerCase(case("by_name-001"), [run(None), run("model_not_found")]),
        AnswerCase(case("include-001", "include"), [run(None), run("length")]),
        AnswerCase(
            case("out_of_scope-001", "out_of_scope"),
            [run("model_not_found"), run("model_not_found")],
        ),
    ]

    assert false_refusal(results) == Rate(2, 4)
    assert per_run(results, false_refusal) == [0.0, 1.0]


def test_must_include_is_exact_and_must_not_include_ignores_case() -> None:
    results = [
        AnswerCase(
            case("include-001", "include", must_include=["Курица", "Рис"]),
            [run(text="Подойдёт «Курица с рисом в духовке».")],
        ),
        AnswerCase(
            case("out_of_scope-001", "out_of_scope", must_not_include=["кенгуру"]),
            [run(text="Возьмите КЕНГУРУ."), run("model_not_found")],
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
        AnswerCase(case("by_name-001"), [run(None), run(None)]),
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
        "tool_correctness": True,
        "citation_accuracy": True,
        "abstention": False,
        "false_refusal": True,
    }


def test_report_shows_broken_answers_with_model_output() -> None:
    broken = run("format_error")
    results = [AnswerCase(case("by_name-001"), [broken])]

    report = render_answer_report(
        results, meta=meta(LlmStats()), judging=None, today=date(2026, 11, 4)
    )

    assert "## Поломки: format_error, blacklist, length (1)" in report
    assert "- by_name-001: format_error, finish_reason=stop, 2 символов: «{}»" in report


def test_holdout_report_says_so() -> None:
    results = [AnswerCase(case("holdout-001"), [run(None)])]
    holdout = dataclasses.replace(meta(LlmStats()), split="holdout")

    report = render_answer_report(
        results, meta=holdout, judging=None, today=date(2026, 11, 4)
    )

    assert "**отложенной выборке**" in report


def test_tool_correctness_needs_search_for_recipes_and_none_for_chat() -> None:
    """Режим always ищет и на «привет» — это промах по построению."""
    results = [
        AnswerCase(case("by_name-001"), [run(None), run(None, searches=())]),
        AnswerCase(
            case("chit_chat-001", "chit_chat"),
            [run(None, chat=True, searches=()), run(None, chat=True)],
        ),
        # follow_up не считается: про уже названный рецепт верны оба решения.
        AnswerCase(case("follow_up-001", "follow_up"), [run(None, searches=())]),
    ]

    assert tool_correctness(results) == Rate(2, 4)
    assert chat_handled(results) == Rate(2, 2)


def test_chat_on_out_of_scope_is_an_honest_refusal_but_not_in_scope() -> None:
    """«Я подбираю только рецепты» на «напиши код» — честный отказ;
    на вопрос про рецепт — отказ, которого быть не должно."""
    results = [
        AnswerCase(case("out_of_scope-001", "out_of_scope"), [run(None, chat=True)]),
        AnswerCase(case("include-001", "include"), [run(None, chat=True)]),
    ]

    assert abstention(results) == Rate(1, 1)
    assert false_refusal(results) == Rate(1, 1)


def test_claude_cost_counts_input_cache_and_output_separately() -> None:
    stats = LlmStats(
        prompt_tokens=10_000, completion_tokens=2_000, precached_prompt_tokens=100_000
    )

    # 10k × $0.10 + 2k × $0.50 + 100k × $0.01 за 1M = $0.003.
    assert cost_usd("claude-haiku-5-5", stats) == pytest.approx(0.003)
    assert cost_usd("GigaChat-2-Pro", stats) is None


def test_follow_up_about_a_recipe_not_in_the_index_is_not_counted() -> None:
    """«Сколько муки во втором?», а второго рецепта нет в выборке: честное
    «не нашёл» — не ложный отказ, и must_include его не касается."""
    follow_up = case("follow_up-003", "follow_up", must_include=["50 г"])
    results = [
        AnswerCase(follow_up, [run("model_not_found")], available=False),
        AnswerCase(case("include-001", "include"), [run(None)]),
    ]

    assert false_refusal(results) == Rate(0, 1)
    assert must_include_found(results) == Rate(0, 0)


def test_spread_by_run_keeps_cases_that_cannot_be_checked_out() -> None:
    """Разброс по прогонам считается по тем же кейсам, что и общая метрика."""
    follow_up = case("follow_up-003", "follow_up", must_include=["50 г"])
    results = [
        AnswerCase(
            follow_up, [run("model_not_found"), run("model_not_found")], available=False
        ),
        AnswerCase(case("include-001", "include"), [run(None), run(None)]),
    ]

    assert per_run(results, false_refusal) == [0.0, 0.0]
