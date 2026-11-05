from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.dataset import (
    CANARIES_FILE,
    CLARITY_CANARIES_FILE,
    HOLDOUT_DIR,
    Case,
    load_canaries,
    load_cases,
    load_clarity_canaries,
)

GOOD = """\
- id: include-001
  category: include
  question: "что приготовить из курицы и риса"
  style: plain
  turns: []
  expected_behavior: answer
  expected_include: [[курица, филе куриное], [рис]]
  expected_exclude: []
  expected_name: []
  escalation_reason: null
  must_include: []
  must_not_include: []
  notes: ""
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_loads_cases_from_all_yaml_files(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", GOOD)
    write(tmp_path, "b.yaml", GOOD.replace("include-001", "include-002"))

    cases = load_cases(tmp_path)

    assert [case.id for case in cases] == ["include-001", "include-002"]
    assert cases[0].question == "что приготовить из курицы и риса"


def test_repeated_ids_are_an_error(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", GOOD)
    write(tmp_path, "b.yaml", GOOD)

    with pytest.raises(ValueError, match="include-001"):
        load_cases(tmp_path)


def test_subdirectories_are_not_loaded(tmp_path: Path) -> None:
    """Отложенная выборка лежит в holdout/ и не должна попадать в обычный прогон."""
    write(tmp_path, "a.yaml", GOOD)
    (tmp_path / "holdout").mkdir()
    write(tmp_path / "holdout", "h.yaml", GOOD.replace("include-001", "holdout-001"))

    assert [case.id for case in load_cases(tmp_path)] == ["include-001"]
    assert [case.id for case in load_cases(tmp_path / "holdout")] == ["holdout-001"]


def test_typo_in_field_name_is_an_error(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", GOOD.replace("expected_include:", "expected_includes:"))

    with pytest.raises(ValidationError):
        load_cases(tmp_path)


def test_groups_of_alternatives_are_read(tmp_path: Path) -> None:
    [loaded] = load_cases(write(tmp_path, "a.yaml", GOOD))

    assert loaded.expected_include == [["курица", "филе куриное"], ["рис"]]


@pytest.mark.parametrize(
    ("category", "field"),
    [
        ("include", "expected_include"),
        ("exclude", "expected_exclude"),
        ("by_name", "expected_name"),
        ("follow_up", "turns"),
    ],
)
def test_category_needs_its_condition(category: str, field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Case(
            id=f"{category}-001",
            category=category,
            question="?",
            expected_behavior="answer",
        )


def test_empty_group_is_an_error() -> None:
    with pytest.raises(ValidationError, match="пустая группа"):
        Case(
            id="include-001",
            category="include",
            question="?",
            expected_behavior="answer",
            expected_include=[["курица"], []],
        )


@pytest.mark.parametrize(
    ("category", "behavior"), [("out_of_scope", "answer"), ("chit_chat", "answer")]
)
def test_fixed_behavior_categories(category: str, behavior: str) -> None:
    with pytest.raises(ValidationError, match="expected_behavior"):
        Case(
            id=f"{category}-001",
            category=category,
            question="?",
            expected_behavior=behavior,
        )


def test_escalation_reason_goes_with_escalate_only() -> None:
    Case(
        id="escalation-001",
        category="escalation",
        question="позовите человека",
        expected_behavior="escalate",
        escalation_reason="request",
    )
    Case(
        id="escalation-002",
        category="escalation",
        question="спасибо, всё понятно",
        expected_behavior="answer",
    )
    with pytest.raises(ValidationError, match="escalation_reason"):
        Case(
            id="escalation-003",
            category="escalation",
            question="?",
            expected_behavior="escalate",
        )
    with pytest.raises(ValidationError, match="escalation_reason"):
        Case(
            id="escalation-004",
            category="escalation",
            question="?",
            expected_behavior="answer",
            escalation_reason="health",
        )


def test_follow_up_keeps_dialog_turns() -> None:
    case = Case.model_validate(
        {
            "id": "follow_up-001",
            "category": "follow_up",
            "question": "а без лука?",
            "turns": [
                {"role": "user", "text": "что приготовить из курицы?"},
                {"role": "assistant", "text": "Курица с луком в духовке"},
            ],
            "expected_behavior": "answer",
            "expected_include": [["курица"]],
            "expected_exclude": ["лук репчатый"],
        }
    )

    assert [turn.role for turn in case.turns] == ["user", "assistant"]


def test_unknown_expected_behavior_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Case(
            id="semantic-001",
            category="semantic",
            question="?",
            expected_behavior="maybe",
        )


CANARY = """\
- id: canary-001
  statement: "Язык OCR задаётся параметром PAPERLESS_OCR_LANGUAGE"
  quote: "Customize the language that paperless will attempt to use when parsing documents."
  supported: true
  notes: ""
"""


def test_loads_canaries(tmp_path: Path) -> None:
    file = write(tmp_path, "canaries.yaml", CANARY) / "canaries.yaml"

    canaries = load_canaries(file)

    assert [c.id for c in canaries] == ["canary-001"]
    assert canaries[0].supported is True


def test_canary_without_verdict_is_an_error(tmp_path: Path) -> None:
    text = CANARY.replace("  supported: true\n", "")
    file = write(tmp_path, "c.yaml", text) / "c.yaml"

    with pytest.raises(ValidationError, match="supported"):
        load_canaries(file)


def test_repeated_canary_ids_are_an_error(tmp_path: Path) -> None:
    file = write(tmp_path, "c.yaml", CANARY + CANARY) / "c.yaml"

    with pytest.raises(ValueError, match="canary-001"):
        load_canaries(file)


def test_dataset_in_repository_follows_the_schema() -> None:
    """Наборы в evals/dataset/ проходят схему: ошибка разметки видна в make test, а не на прогоне.

    Набор про рецепты пишется на этапе 1, канарейки судьи — на этапе 3: пока
    файлов нет, проверять нечего, но загрузка не должна падать.
    """
    load_cases()
    load_cases(HOLDOUT_DIR)
    if CANARIES_FILE.exists():
        assert load_canaries()
    if CLARITY_CANARIES_FILE.exists():
        assert load_clarity_canaries()


CLARITY = """\
- id: clarity-001
  statement: "Документы хранятся в Docker-объёме `paperless_media`."
  quote: "By default, your documents are stored inside the docker volume paperless_media."
  clear: false
  notes: "volume — это том"
"""


def test_loads_clarity_canaries(tmp_path: Path) -> None:
    file = write(tmp_path, "c.yaml", CLARITY) / "c.yaml"

    [canary] = load_clarity_canaries(file)

    assert canary.clear is False


def test_clarity_canary_needs_its_own_id_prefix(tmp_path: Path) -> None:
    file = (
        write(tmp_path, "c.yaml", CLARITY.replace("clarity-001", "canary-001"))
        / "c.yaml"
    )

    with pytest.raises(ValidationError):
        load_clarity_canaries(file)
