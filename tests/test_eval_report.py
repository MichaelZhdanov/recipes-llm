import subprocess
from datetime import date
from pathlib import Path

import pytest
from eval_helpers import MANIFEST, case, found, result

from assistant.generation.gigachat import LlmStats
from assistant.retrieval.embeddings import EmbeddingStats
from assistant.retrieval.filters import IngredientFilter
from assistant.tools.search_recipes import SearchArgs
from evals.report import (
    Parsing,
    git_revision,
    render_report,
    report_path,
    retrieval_gates,
)
from evals.retrieval_metrics import CaseResult, RetrievalRun

PLOV = found("Плов с курицей", ("курица", "рис"))
PANCAKES = found("Блины", ("молоко", "мука пшеничная"))


def render(
    results: list[CaseResult],
    *,
    query_template: str | None = "Instruct: {task}\nQuery: {query}",
    sample: str | None = None,
    parsing: Parsing | None = None,
) -> str:
    return render_report(
        results,
        manifest=MANIFEST,
        revision="abc1234",
        provider="cloud.ru",
        query_template=query_template,
        stats=EmbeddingStats(tokens=42),
        today=date(2026, 11, 4),
        sample=sample,
        parsing=parsing,
    )


def test_report_shows_metrics_categories_and_misses() -> None:
    results = [
        result(case("by_name-001", expected_name=["плов"]), [[PLOV, PANCAKES]] * 3),
        result(
            case("by_name-002", question="борщ", expected_name=["борщ"]), [[PANCAKES]]
        ),
        result(
            case("include-001", "include", expected_include=[["рис"]]),
            [[PLOV, PANCAKES]],
        ),
        result(case("semantic-001", "semantic", question="к чаю"), [[PANCAKES]], None),
        result(
            case("by_name-003", question="кенгуру", expected_name=["кенгуру"]),
            [[PANCAKES]],
            available=False,
        ),
    ]

    report = render(results)

    assert report.startswith("# Отчёт eval — 2026-11-04")
    assert "| Данные | rogozinushka/povarenok-recipes, ревизия v3.4.2 |" in report
    assert "| Модель эмбеддингов | cloud.ru: fake-model, размерность 64 |" in report
    # Перевод строки в инструкции не должен ломать таблицу markdown.
    assert "\\nQuery: <вопрос>` |" in report
    assert "(by_name) | 0.75 | ≥ 0.85 ❌ |" in report
    assert "(include) | 0.50 | — |" in report
    assert (
        "| M этапа 1: среднее name_hit@5 и include_precision@5 | 0.62 | — |" in report
    )
    # include-001: половина топа без риса — условия выполнены на 0.50.
    assert "(include, exclude, follow_up)** | **0.50** | ≥ 0.90 ❌ |" in report
    assert "| by_name | 3 | 0.75 |" in report
    assert "- by_name-003 — «кенгуру»" in report
    assert "| Разбор вопроса | нет (--no-filters)" in report
    # Промахи: by_name без нужного названия и include, где не весь топ подходит.
    assert "## Промахи (2)" in report
    assert "### by_name-002 — «борщ»" in report
    assert "### include-001 — «?»" in report
    assert "1. Блины (0.500) — молоко, мука пшеничная" in report
    assert "## Вопросы по смыслу: что нашлось" in report


@pytest.mark.parametrize(
    ("label", "name"),
    [(None, "2026-11-04.md"), ("voyage-4-lite", "2026-11-04-voyage-4-lite.md")],
)
def test_report_path(label: str | None, name: str) -> None:
    assert report_path(Path("r"), date(2026, 11, 4), label) == Path("r") / name


def test_revision_ignores_untracked_files_but_not_edits(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")

    (tmp_path / "draft.py").write_text("черновик\n")
    assert "изменения" not in git_revision(tmp_path)

    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_revision(tmp_path).endswith("(есть незакоммиченные изменения)")


def test_sampled_report_says_it_is_a_quick_run() -> None:
    report = render(
        [result(case("by_name-001"), [[PLOV]])],
        query_template=None,
        sample="8 из 32, сид 20261104",
    )

    assert "| Выборка кейсов | 8 из 32, сид 20261104 — быстрый прогон" in report
    assert "| Инструкция к вопросу | нет, вопрос уходит как есть |" in report


def test_gates_follow_thresholds() -> None:
    """Пороги: conditions ≥ 0.90, exclude_violations ≤ 0.05, name_hit ≥ 0.85."""
    good = [
        result(case("by_name-001", expected_name=["плов"]), [[PLOV]]),
        result(case("exclude-001", "exclude", expected_exclude=["молоко"]), [[PLOV]]),
    ]
    bad = [
        result(case("by_name-001", expected_name=["борщ"]), [[PLOV]]),
        result(case("exclude-001", "exclude", expected_exclude=["рис"]), [[PLOV]]),
    ]

    assert retrieval_gates(good) == {
        "conditions": True,
        "exclude_violations": True,
        "name_hit": True,
    }
    assert retrieval_gates(bad) == {
        "conditions": False,
        "exclude_violations": False,
        "name_hit": False,
    }
    assert retrieval_gates([]) == {}


def test_report_shows_how_questions_were_parsed() -> None:
    args = SearchArgs(
        query="плов без молока",
        conditions=IngredientFilter.of(include=[["рис"]], exclude=["молоко"]),
        dropped=("рис басмати",),
    )
    kase = case(
        "exclude-001",
        "exclude",
        question="плов, молоко нельзя",
        expected_exclude=["молоко"],
    )
    results = [
        CaseResult(kase, [RetrievalRun(top=[PLOV], latency=1.5, args=args)], True),
        CaseResult(
            case("by_name-001", question="борщ", expected_name=["борщ"]),
            [
                RetrievalRun(
                    top=[], latency=0.2, error="GigaChat не вызвал search_recipes"
                )
            ],
            True,
        ),
    ]
    stats = LlmStats(requests=2, total_tokens=9000, precached_prompt_tokens=8000)
    parsing = Parsing(
        model="GigaChat-2-Pro",
        served=("GigaChat-2-Pro:2.0.30.01",),
        prompt_version="search_v1",
        vocabulary=820,
        stats=stats,
        retries=4,
    )

    report = render(results, parsing=parsing)

    assert "(из ответа API: GigaChat-2-Pro:2.0.30.01), промпт `search_v1`" in report
    assert "словарь индекса — 820 имён" in report
    assert "| имена из разбора не из словаря (отброшены) | 1 | — |" in report
    assert "| сбои разбора (выдача пустая) | 1 | — |" in report
    assert "| повторы разбора после зацикливания (length) | 4 | — |" in report
    assert "| токены GigaChat на разбор: всего / из кэша | 9000 / 8000 | — |" in report
    assert (
        "- exclude-001 «плов, молоко нельзя» → запрос «плов без молока» · с: рис"
        " · без: молоко · нет в словаре: рис басмати"
    ) in report
    assert (
        "- by_name-001 «борщ» → сбой разбора: GigaChat не вызвал search_recipes"
        in report
    )
