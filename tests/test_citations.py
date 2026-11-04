from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from assistant.generation.citations import (
    ANSWER_SCHEMA,
    Citation,
    ModelAnswer,
    SourceText,
    Statement,
    check_answer,
    check_citation,
    check_statement,
    heading_lines,
    names_in,
)
from assistant.indexing.chunks import Chunk

DOC = """\
#### Choosing the OCR language for documents {#PAPERLESS_OCR_LANGUAGE}

: Customize the language that paperless will attempt to use when
parsing documents.

    It should be a 3-letter code, see the **list of languages**
    [Tesseract supports](https://tesseract-ocr.github.io/tessdoc/Data-Files).

    Defaults to “eng” when nothing is set.

```bash
# Set the OCR language before start
export PAPERLESS_OCR_LANGUAGE=deu PAPERLESS_OCR_MODE=skip
```
"""

CODE = """\
def check_default_language_available(app_configs, **kwargs):
    # The default language is always installed
    lang = settings.OCR_LANGUAGE
    return []
"""


def chunk(text: str, kind: str = "doc", start: int = 891) -> Chunk:
    path = "docs/configuration.md" if kind == "doc" else "src/paperless/checks.py"
    return Chunk.model_validate(
        {
            "kind": kind,
            "path": path,
            "start_line": start,
            "end_line": start + text.count("\n"),
            "title": "t",
            "breadcrumb": "Configuration > OCR settings",
            "source_id": f"{path}#x",
            "url": f"https://github.com/x/blob/v3.4.2/{path}#L{start}-L{start + 9}",
            "site_url": None,
            "ref": "v3.4.2",
            "text": text,
        }
    )


SOURCES = [
    SourceText(number=1, chunk=chunk(DOC), text=DOC),
    SourceText(number=2, chunk=chunk(CODE, kind="code", start=351), text=CODE),
]


def cite(source: int, quote: str) -> Citation:
    return Citation(source=source, quote=quote)


def test_quote_found_and_narrowed_to_its_lines() -> None:
    checked = check_citation(
        cite(
            1, "the language that paperless will attempt to use when parsing documents"
        ),
        SOURCES,
    )

    assert checked.valid
    # Цитата идёт через перевод строки: строки 2–3 куска → 893–894 файла.
    assert (checked.start_line, checked.end_line) == (893, 894)


def test_whitespace_and_case_do_not_matter() -> None:
    checked = check_citation(
        cite(1, "CUSTOMIZE the   language that\tpaperless"), SOURCES
    )

    assert checked.valid


@pytest.mark.parametrize(
    "quote",
    [
        # Модель передала разметку по-своему: без ** и без ссылки Markdown.
        "see the list of languages Tesseract supports",
        # Типографские кавычки заменены прямыми.
        'Defaults to "eng" when nothing is set.',
        # Обратные кавычки вокруг имени.
        "export `PAPERLESS_OCR_LANGUAGE=deu` PAPERLESS_OCR_MODE=skip",
    ],
)
def test_formatting_is_normalized(quote: str) -> None:
    assert check_citation(cite(1, quote), SOURCES).valid


def test_underscore_is_not_normalized() -> None:
    checked = check_citation(cite(1, "export PAPERLESS OCR LANGUAGE=deu"), SOURCES)

    assert checked.error == "фрагмента нет в источнике"


@pytest.mark.parametrize(
    ("citation", "error"),
    [
        (cite(7, "Customize the language that paperless"), "источника 7 нет в промпте"),
        (cite(1, "Customize the"), "слишком короткая цитата"),
        (cite(1, "a b c d"), "слишком короткая цитата"),
        (cite(1, "language " * 60), "слишком длинная цитата"),
        # Пересказ и перевод — не цитата.
        (
            cite(1, "Задаёт язык, на котором paperless распознаёт документы"),
            "фрагмента нет в источнике",
        ),
        # Пропуск середины через многоточие — тоже не дословно.
        (
            cite(1, "Customize the language ... parsing documents"),
            "фрагмента нет в источнике",
        ),
    ],
)
def test_invalid_citations(citation: Citation, error: str) -> None:
    assert check_citation(citation, SOURCES).error == error


def test_quote_of_heading_only_is_rejected() -> None:
    checked = check_citation(
        cite(1, "Choosing the OCR language for documents"), SOURCES
    )

    assert checked.error == "цитата совпадает только с заголовком"


def test_hash_line_inside_code_block_is_not_a_heading() -> None:
    assert heading_lines(DOC) == {0}
    assert check_citation(cite(1, "# Set the OCR language before start"), SOURCES).valid


def test_code_comments_are_not_headings() -> None:
    checked = check_citation(
        cite(2, "# The default language is always installed"), SOURCES
    )

    assert checked.valid
    assert checked.start_line == checked.end_line == 352


def test_names_that_must_be_in_the_source() -> None:
    assert names_in(
        "Задайте PAPERLESS_OCR_LANGUAGE, затем выполните `manage.py document_exporter` "
        "или `document_index reindex`; слово `eng` не имя"
    ) == ["PAPERLESS_OCR_LANGUAGE", "document_exporter", "document_index"]


def test_statement_with_valid_quote_is_kept() -> None:
    checked = check_statement(
        Statement(
            text="Язык OCR задаёт PAPERLESS_OCR_LANGUAGE, по умолчанию `eng`.",
            citations=[
                cite(1, "Customize the language that paperless"),
                cite(1, "выдуманная цитата, которой нет"),
            ],
        ),
        SOURCES,
    )

    assert checked.kept
    assert [c.valid for c in checked.citations] == [True, False]


def test_made_up_parameter_drops_statement() -> None:
    checked = check_statement(
        Statement(
            text="Язык OCR задаёт PAPERLESS_OCR_LANG.",
            citations=[cite(1, "Customize the language that paperless")],
        ),
        SOURCES,
    )

    assert not checked.kept
    assert checked.unknown_names == ["PAPERLESS_OCR_LANG"]


def test_name_must_match_whole_not_as_prefix() -> None:
    source = [
        SourceText(
            number=1, chunk=chunk(DOC), text=DOC.replace("LANGUAGE", "LANGUAGES")
        )
    ]

    checked = check_statement(
        Statement(
            text="Нужен PAPERLESS_OCR_LANGUAGE.",
            citations=[cite(1, "Customize the language that paperless")],
        ),
        source,
    )

    assert checked.unknown_names == ["PAPERLESS_OCR_LANGUAGE"]


def test_name_must_be_in_a_cited_source_not_just_any() -> None:
    # Имя есть в источнике 1, но процитирован только источник 2.
    checked = check_statement(
        Statement(
            text="Язык задаёт PAPERLESS_OCR_LANGUAGE.",
            citations=[cite(2, "The default language is always installed")],
        ),
        SOURCES,
    )

    assert not checked.kept


def test_statement_without_valid_citation_is_dropped() -> None:
    answer = ModelAnswer(
        status="answer",
        statements=[
            Statement(text="Без цитат.", citations=[]),
            Statement(
                text="С чужой цитатой.",
                citations=[cite(1, "this text is not there at all")],
            ),
        ],
    )

    assert [s.kept for s in check_answer(answer, SOURCES)] == [False, False]


def test_model_answer_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate(
            {"status": "answer", "statements": [], "confidence": 0.9}
        )
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate({"status": "maybe", "statements": []})


def test_json_schema_for_gigachat_matches_the_model() -> None:
    """Схему для response_format пишем руками: она обязана совпадать с ModelAnswer."""

    def same(schema: Any, model: type[BaseModel]) -> None:
        fields = set(model.model_fields)
        assert set(schema["properties"]) == fields
        assert set(schema["required"]) == fields

    same(ANSWER_SCHEMA, ModelAnswer)
    statement = ANSWER_SCHEMA["properties"]["statements"]["items"]  # type: ignore[index]
    same(statement, Statement)
    same(statement["properties"]["citations"]["items"], Citation)


FORMATTED = """\
## Duplicates

As of version 3.0 Paperless-ngx allows duplicate documents to be consumed by
default, _except_ when this setting is enabled. "Share links" are public links
to files, see the [document
exporter](#exporter) for backups. The setting value is falsehood-proof.
"""


@pytest.mark.parametrize(
    "quote",
    [
        # Курсив _except_ модель передала без подчёркиваний.
        "allows duplicate documents to be consumed by default, except when",
        # Кавычки вокруг «Share links» модель потеряла.
        "Share links are public links to files",
        # Ссылка разбита переносом строки внутри текста ссылки.
        "see the document exporter for backups",
        # Точка после обрезанного предложения.
        "allows duplicate documents to be consumed.",
    ],
)
def test_formatting_differences_do_not_matter(quote: str) -> None:
    sources = [SourceText(number=1, chunk=chunk(FORMATTED), text=FORMATTED)]

    assert check_citation(cite(1, quote), sources).valid


def test_link_across_lines_keeps_line_numbers() -> None:
    sources = [SourceText(number=1, chunk=chunk(FORMATTED, start=10), text=FORMATTED)]

    checked = check_citation(cite(1, "see the document exporter for backups"), sources)

    assert (checked.start_line, checked.end_line) == (14, 15)


@pytest.mark.parametrize(
    "quote",
    [
        # Слова должны совпасть целиком: «false» — не начало «falsehood».
        "The setting value is false.",
        # И с начала: «llows» — не середина «allows».
        "llows duplicate documents to be consumed",
        # Другое слово — не цитата, сколько ни выравнивай оформление.
        "allows duplicate documents to be deleted",
    ],
)
def test_words_must_still_match(quote: str) -> None:
    sources = [SourceText(number=1, chunk=chunk(FORMATTED), text=FORMATTED)]

    assert check_citation(cite(1, quote), sources).error == "фрагмента нет в источнике"
