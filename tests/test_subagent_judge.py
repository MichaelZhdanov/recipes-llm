"""Судья-субагент: вердикты против ключа — нарушения, релевантность, канарейки."""

import json
from pathlib import Path

from evals.subagent_judge import score


def test_score_counts_cases_and_checks_the_judge_on_canaries(tmp_path: Path) -> None:
    key = {
        "P001": {"kind": "case", "id": "semantic-011", "category": "semantic"},
        "P002": {"kind": "grounding", "id": "canary-001", "expected": True},
        "P003": {"kind": "relevance", "id": "relevance-032", "expected": False},
        "P004": {"kind": "case", "id": "include-001", "category": "include"},
    }
    verdicts = [
        {
            "item": "P001",
            "statements": [
                {"n": 1, "supported": True, "comment": "ок"},
                {"n": 2, "supported": False, "comment": "выдуман шаг"},
            ],
            "recipes": [{"n": 1, "relevant": True}, {"n": 2, "relevant": False}],
        },
        {"item": "P002", "statements": [{"n": 1, "supported": True}], "recipes": []},
        {"item": "P003", "statements": [], "recipes": [{"n": 1, "relevant": True}]},
    ]
    (tmp_path / "key.json").write_text(json.dumps(key), encoding="utf-8")
    (tmp_path / "verdicts1.json").write_text(json.dumps(verdicts), encoding="utf-8")

    text = score(tmp_path, "проба")

    assert "| утверждений без нарушений | 0.50 (1/2," in text
    assert "| рецепты ответа по вопросу | 0.50 (1/2," in text
    assert "| судья на канарейках опоры | 1.00 (1/1," in text
    assert "| судья на канарейках релевантности | 0.00 (0/1," in text
    assert "| пунктов без вердикта | 1 |" in text
    assert "- `semantic-011`: выдуман шаг" in text
