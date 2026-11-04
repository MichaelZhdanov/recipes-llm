import subprocess
from datetime import date
from pathlib import Path

import pytest
from eval_helpers import MANIFEST, result

from assistant.retrieval.embeddings import EmbeddingStats
from evals.report import git_revision, render_report, report_path


def test_report_shows_metrics_threshold_and_misses() -> None:
    results = [
        result("howto-001", [1, 1, 1], role="engineer"),
        result("howto-002", [None, None, None]),
    ]

    report = render_report(
        results,
        manifest=MANIFEST,
        revision="abc1234",
        provider="cloud.ru",
        query_template="Instruct: {task}\nQuery: {query}",
        stats=EmbeddingStats(tokens=42),
        today=date(2026, 11, 4),
    )

    assert report.startswith("# Отчёт eval — 2026-11-04")
    assert "| recall@5 | 0.50 ± 0.00 | ≥ 0.80 | ❌ |" in report
    assert "| Paperless | v3.4.2 (d939d87) |" in report
    assert "| Модель эмбеддингов | cloud.ru: fake-model, размерность 64 |" in report
    # Перевод строки в инструкции не должен ломать таблицу markdown.
    assert "\\nQuery: <вопрос>` |" in report
    assert "| роль engineer | 1 | 1.00 |" in report
    assert "## Промахи (1)" in report
    assert "### howto-002" in report


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
