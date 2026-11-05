from chunk_helpers import ROWS

from evals.check_cases import check, vocabulary


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
