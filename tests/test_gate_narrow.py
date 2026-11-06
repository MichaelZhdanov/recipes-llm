"""Узкий гейт: эталон из задания судьи, счёт «поймал / зря», правило."""

import json
from datetime import date
from pathlib import Path

from evals.gate_narrow import NarrowResult, failed_rules, load_task, render, score

RECIPE = "Рецепт целиком:\nСалат с тунцом\nТунец — 1 банка\nЯйцо — 2 шт"
OTHER = "Рецепт целиком:\nГлинтвейн\nВино красное — 750 мл"


def _task(tmp_path: Path) -> Path:
    key = {
        "P001": {"kind": "case", "id": "semantic-001", "category": "semantic"},
        "P002": {"kind": "grounding", "id": "g-1", "expected": True},
    }
    part = [
        {
            "item": "P001",
            "question": "что подать к вину",
            "dialog": ["привет"],
            "statements": [
                {
                    "n": 1,
                    "text": "Салат с тунцом и яйцом.",
                    "intro": False,
                    "recipes": [RECIPE],
                },
                {
                    "n": 2,
                    "text": "Запечённый тунец.",
                    "intro": False,
                    "recipes": [RECIPE],
                },
                {
                    "n": 3,
                    "text": "Глинтвейн на вине.",
                    "intro": False,
                    "recipes": [OTHER],
                },
                {
                    "n": 4,
                    "text": "Вот что нашлось.",
                    "intro": True,
                    "recipes": [RECIPE, OTHER],
                },
            ],
            "recipes": [
                {"n": 1, "name": "Салат с тунцом", "ingredients": []},
                {"n": 2, "name": "Глинтвейн", "ingredients": []},
            ],
        },
        {"item": "P002", "question": "", "dialog": [], "statements": [], "recipes": []},
    ]
    verdicts = [
        {
            "item": "P001",
            "statements": [
                {"n": 1, "comment": "есть", "supported": True},
                {"n": 2, "comment": "способа нет", "supported": False},
                {"n": 3, "comment": "есть", "supported": True},
                {"n": 4, "comment": "без фактов", "supported": True},
            ],
            "recipes": [
                {"n": 1, "comment": "закуска", "relevant": True},
                {"n": 2, "comment": "готовят на вине", "relevant": False},
            ],
        }
    ]
    for name, data in (("key", key), ("part1", part), ("verdicts1", verdicts)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False))
    return tmp_path


def test_task_gives_reference_per_statement(tmp_path: Path) -> None:
    [answer] = load_task(_task(tmp_path))

    assert answer.case == "semantic-001"
    assert answer.dialog == ["привет"]
    assert answer.supported == [True, False, True, True]
    # Релевантность — по рецепту утверждения; у вступления её нет.
    assert answer.relevant == [True, True, False, None]
    assert answer.items[3].intro


def test_score_counts_as_the_answer_path(tmp_path: Path) -> None:
    answers = load_task(_task(tmp_path))
    result = NarrowResult("cand", calls=1)
    # Нарушение поймано; верное №1 — ответ не разобрался (None) — тоже на доработку;
    # глинтвейн удалён как «не по вопросу»; вступление «да».
    result.marks = {
        "P001": [
            (None, None, ""),
            (False, True, "x"),
            (True, False, ""),
            (True, True, ""),
        ]
    }

    s = score(result, answers)

    assert (s.violations, s.caught) == (1, 1)
    assert (s.correct, s.false_rejects) == (3, 1)
    assert (s.irrelevant, s.irrelevant_caught) == (1, 1)
    # Верное и по вопросу, но пользователь его бы не увидел — только №1.
    assert s.lost == 1


def test_rule_needs_enough_caught_and_few_false(tmp_path: Path) -> None:
    answers = load_task(_task(tmp_path))
    result = NarrowResult("cand", calls=1)
    result.marks = {"P001": [(True, True, "")] * 4}

    reasons = failed_rules(result, answers)

    # Одно нарушение из одного пропущено — меньше MIN_CAUGHT.
    assert any(r.startswith("1:") for r in reasons)
    broken = NarrowResult("broken", error="LlmError: 403")
    assert failed_rules(broken, answers) == ["сбой: LlmError: 403"]
    text = render([result, broken], answers, date(2026, 11, 7))
    assert "broken (оборвался: LlmError: 403)" in text
    assert "поймали: никто" in text
