import pytest

from assistant.generation.answer import build_statements
from assistant.generation.citations import (
    CheckedCitation,
    CheckedStatement,
    SourceText,
)
from assistant.generation.history import changelog_version, history_note, mark
from assistant.indexing.chunks import Chunk


def chunk(path: str, breadcrumb: str) -> Chunk:
    return Chunk(
        kind="doc",
        path=path,
        start_line=1,
        end_line=2,
        title="t",
        breadcrumb=breadcrumb,
        source_id=f"{path}#{breadcrumb}",
        url="https://x",
        site_url=None,
        ref="v3.4.2",
        text="t",
    )


@pytest.mark.parametrize(
    ("breadcrumb", "version"),
    [
        ("Changelog > paperless-ngx 1.10.2 > Features", "paperless-ngx 1.10.2"),
        ("Changelog > Paperless-ng > paperless-ng 0.9.14", "paperless-ng 0.9.14"),
        # У исходного проекта версия — отдельный заголовок под «Paperless».
        ("Changelog > Paperless > 0.0.3", "Paperless 0.0.3"),
        ("Changelog > Paperless > Migrating from 1.x", None),
    ],
)
def test_version_from_breadcrumb(breadcrumb: str, version: str | None) -> None:
    assert changelog_version(breadcrumb) == version


HELM = chunk("docs/changelog.md", "Changelog > paperless-ngx 1.10.2 > Features")
REMOVED = chunk("docs/changelog.md", "Changelog > paperless-ngx 1.12.0 > Maintenance")
SETUP = chunk("docs/setup.md", "Setup > Docker")


def test_statement_only_from_changelog_gets_its_versions() -> None:
    assert history_note([HELM]) == " (история изменений: paperless-ngx 1.10.2)"
    assert history_note([HELM, REMOVED, HELM]) == (
        " (история изменений: paperless-ngx 1.10.2, paperless-ngx 1.12.0)"
    )


def test_statement_backed_by_current_docs_is_not_marked() -> None:
    assert history_note([HELM, SETUP]) == ""
    assert history_note([SETUP]) == ""
    assert history_note([]) == ""


def test_note_goes_before_the_final_period() -> None:
    assert mark("Команда поддерживает Helm-чарт.", [HELM]) == (
        "Команда поддерживает Helm-чарт (история изменений: paperless-ngx 1.10.2)."
    )
    assert mark("Есть Helm-чарт", [HELM]).endswith(
        "чарт (история изменений: paperless-ngx 1.10.2)"
    )
    assert mark("Том задаётся в compose.", [SETUP]) == "Том задаётся в compose."


def test_answer_marks_only_statements_from_changelog() -> None:
    sources = [SourceText(1, HELM, "t"), SourceText(2, SETUP, "t")]

    def statement(text: str, source: int) -> CheckedStatement:
        return CheckedStatement(
            text, [CheckedCitation(source, "q", start_line=1, end_line=1)], []
        )

    statements, _, _ = build_statements(
        [statement("Команда поддерживает Helm-чарт.", 1), statement("Есть Docker.", 2)],
        sources,
    )

    assert [s.text for s in statements] == [
        "Команда поддерживает Helm-чарт (история изменений: paperless-ngx 1.10.2).",
        "Есть Docker.",
    ]
