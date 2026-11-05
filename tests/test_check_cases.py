from chunk_helpers import ROWS

from evals.check_cases import check, check_canaries, vocabulary


def test_vocabulary_counts_lowercase_names() -> None:
    counts = vocabulary(ROWS)

    assert counts["сахар"] == 2
    assert counts["курица"] == 1


def test_check_counts_matching_recipes_and_reports_problems() -> None:
    items = [
        {
            "id": "include-001",
            "category": "include",
            "question": "курица и рис",
            "expected_behavior": "answer",
            "expected_include": [["курица"], ["рис"]],
        },
        {
            "id": "exclude-001",
            "category": "exclude",
            "question": "без сахара",
            "expected_behavior": "answer",
            "expected_exclude": ["сахар"],
        },
        {
            "id": "include-002",
            "category": "include",
            "question": "курятина",
            "expected_behavior": "answer",
            "expected_include": [["курятина"]],
        },
        {"id": "broken-001", "category": "include", "question": "?"},
        {
            "id": "chit_chat-001",
            "category": "chit_chat",
            "question": "привет",
            "expected_behavior": "no_search",
        },
    ]

    lines, problems = check(ROWS, items)

    assert "include-001\tinclude\tподходит рецептов: 1" in lines
    assert "exclude-001\texclude\tподходит рецептов: 1" in lines
    assert "НЕТ В СЛОВАРЕ include-002: курятина" in lines
    assert any(line.startswith("СХЕМА broken-001") for line in lines)
    # У болтовни условий нет — и строки с числом нет.
    assert not any(line.startswith("chit_chat-001") for line in lines)
    assert problems == 2


def test_canaries_must_quote_lines_of_one_real_recipe() -> None:
    """Судья увидит строки рецепта — канарейка должна давать ему такие же."""
    files: dict[str, list[object]] = {
        "canaries": [
            {
                "id": "canary-001",
                "statement": "Нужно 250 мл молока и 200 г клубники.",
                "quote": "Молоко — 250 мл\nклубника - 200 г",
                "supported": True,
            },
            {
                # Строки из двух разных рецептов.
                "id": "canary-002",
                "statement": "?",
                "quote": "Молоко — 250 мл\nКурица — 500 г",
                "supported": False,
            },
            {
                "id": "canary-002",
                "statement": "?",
                "quote": "Сахар — 15 г",
                "supported": False,
            },
        ],
        "clarity_canaries": [{"id": "clarity-001", "statement": "?", "quote": "?"}],
        "relevance_canaries": [
            {
                "id": "relevance-001",
                "question": "к чаю",
                "name": "Овсяное печенье",
                "ingredients": ["хлопья овсяные", "сахар"],
                "relevant": True,
            },
            {
                "id": "relevance-002",
                "question": "к чаю",
                "name": "Овсяное печенье",
                "ingredients": ["шоколад"],
                "relevant": False,
            },
        ],
    }

    lines, problems = check_canaries(ROWS, files)

    assert "НЕ СТРОКИ РЕЦЕПТА canary-002: quote не из строк одного рецепта" in lines
    assert "ПОВТОР canaries/canary-002" in lines
    assert any(line.startswith("СХЕМА clarity_canaries/clarity-001") for line in lines)
    assert (
        "НЕТ РЕЦЕПТА relevance-002: «Овсяное печенье» с такими ингредиентами" in lines
    )
    assert "canaries: 2 шт., да 1, нет 2" in lines
    assert problems == 4
