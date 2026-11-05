import subprocess
from datetime import date
from pathlib import Path

import pytest
from eval_helpers import MANIFEST, case, found, result

from assistant.retrieval.embeddings import EmbeddingStats
from evals.report import git_revision, render_report, report_path
from evals.retrieval_metrics import CaseResult

PLOV = found("Плов с курицей", ("курица", "рис"))
PANCAKES = found("Блины", ("молоко", "мука пшеничная"))


def render(
    results: list[CaseResult],
    *,
    query_template: str | None = "Instruct: {task}\nQuery: {query}",
    sample: str | None = None,
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
    assert "(by_name) | 0.75 |" in report
    assert "(include) | 0.50 |" in report
    assert (
        "**основная: среднее name_hit@5 и include_precision@5** | **0.62** |" in report
    )
    assert "| by_name | 3 | 0.75 |" in report
    assert "- by_name-003 — «кенгуру»" in report
    assert "## Промахи (1)" in report
    assert "### by_name-002 — «борщ»" in report
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
