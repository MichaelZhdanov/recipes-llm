"""Проверка цитат кодом: цитата — целая строка рецепта, числа — из процитированных строк."""

from typing import Any

import pytest
from chunk_helpers import CHICKEN, COCKTAIL, make_chunk
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
    line_key,
    numbers_in,
)

# Текст рецепта в том виде, в каком он уходит модели:
# 1 «Густой молочно-клубничный коктейль», 2 «Молоко — 250 мл»,
# 3 «Клубника — 200 г», 4 «Сахар — 15 г».
SOURCES = [
    SourceText(1, COCKTAIL, COCKTAIL.text),
    SourceText(2, CHICKEN, CHICKEN.text),
]


def cite(source: int, quote: str) -> Citation:
    return Citation(source=source, quote=quote)


def test_whole_line_is_a_citation_with_its_line_number() -> None:
    checked = check_citation(cite(1, "Молоко — 250 мл"), SOURCES)

    assert checked.valid
    assert (checked.start_line, checked.end_line) == (2, 2)


def test_recipe_name_is_the_first_line() -> None:
    checked = check_citation(cite(1, "Густой молочно-клубничный коктейль"), SOURCES)

    assert checked.start_line == 1


@pytest.mark.parametrize(
    "quote",
    [
        "молоко - 250 мл",  # регистр и короткий дефис вместо тире
        "Молоко – 250 мл.",  # среднее тире и точка в конце
        "  Молоко —   250 мл ",  # пробелы
        "**Молоко** — 250 мл",  # разметка Markdown
        "«Молоко — 250 мл»",  # кавычки
    ],
)
def test_formatting_of_the_line_does_not_matter(quote: str) -> None:
    assert check_citation(cite(1, quote), SOURCES).valid


def test_yo_and_e_are_the_same() -> None:
    chunk = make_chunk("Пирог", {"Тесто слоёное": "500 г"}, recipe_id=7)
    sources = [SourceText(1, chunk, chunk.text)]

    assert check_citation(cite(1, "Тесто слоеное — 500 г"), sources).valid


@pytest.mark.parametrize(
    ("citation", "error"),
    [
        (cite(9, "Молоко — 250 мл"), "рецепта 9 нет в промпте"),
        (cite(1, "Молоко"), "это часть строки, а цитата — строка целиком"),
        (cite(1, "Молоко — 500 мл"), "такой строки нет в рецепте"),
        (cite(2, "Молоко — 250 мл"), "такой строки нет в рецепте"),
        (
            cite(1, "Молоко — 250 мл\nКлубника — 200 г"),
            "цитата — одна строка рецепта, а не несколько",
        ),
        (cite(1, " — "), "пустая цитата"),
    ],
)
def test_invalid_citations(citation: Citation, error: str) -> None:
    assert check_citation(citation, SOURCES).error == error


def test_numbers_are_found_with_comma_or_dot_and_fractions() -> None:
    assert numbers_in("1,5 стакана муки, 1/2 ч. л. соли и 2 яйца") == [
        "1.5",
        "1/2",
        "2",
    ]
    assert line_key("  Сахар — 15 г. ") == "сахар - 15 г"


def test_statement_with_numbers_from_the_cited_line_is_kept() -> None:
    statement = Statement(
        text="Молока нужно 250 мл.", citations=[cite(1, "Молоко — 250 мл")]
    )

    checked = check_statement(statement, SOURCES)

    assert checked.kept
    assert checked.unknown_numbers == []


def test_made_up_quantity_drops_the_statement() -> None:
    """Цитата верная, но 500 мл в ней нет: количество выдумано."""
    statement = Statement(
        text="Молока нужно 500 мл.", citations=[cite(1, "Молоко — 250 мл")]
    )

    checked = check_statement(statement, SOURCES)

    assert not checked.kept
    assert checked.unknown_numbers == ["500"]


def test_number_must_be_in_a_cited_line_not_just_in_the_recipe() -> None:
    """200 есть в рецепте (клубника), но процитирована строка молока."""
    statement = Statement(
        text="Нужно 200 г клубники.", citations=[cite(1, "Молоко — 250 мл")]
    )

    assert check_statement(statement, SOURCES).unknown_numbers == ["200"]


def test_statement_without_valid_citation_is_dropped() -> None:
    answer = ModelAnswer(
        status="answer",
        statements=[
            Statement(text="Есть клубника.", citations=[cite(1, "Клубника — 200 г")]),
            Statement(text="Есть ананас.", citations=[cite(1, "Ананас — 1 шт")]),
        ],
    )

    assert [s.kept for s in check_answer(answer, SOURCES)] == [True, False]


def test_model_answer_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate(
            {"status": "answer", "statements": [], "confidence": 0.9}
        )
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate({"status": "maybe", "statements": []})


def test_json_schema_for_the_model_matches_the_pydantic_model() -> None:
    """Схему для response_format пишем руками: она обязана совпадать с ModelAnswer."""

    def same(schema: Any, model: type[BaseModel]) -> None:
        fields = set(model.model_fields)
        assert set(schema["properties"]) == fields
        assert set(schema["required"]) == fields

    same(ANSWER_SCHEMA, ModelAnswer)
    statement = ANSWER_SCHEMA["properties"]["statements"]["items"]  # type: ignore[index]
    same(statement, Statement)
    same(statement["properties"]["citations"]["items"], Citation)
