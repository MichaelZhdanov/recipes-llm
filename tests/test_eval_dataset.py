from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.dataset import Case, load_canaries, load_cases

GOOD = """\
- id: howto-001
  category: howto
  role: user
  question: "как загрузить документ"
  expected_behavior: answer
  expected_sources:
    - docs/usage.md#web-ui-upload
  must_include: []
  must_not_include: []
  notes: ""
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_loads_cases_from_all_yaml_files(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", GOOD)
    write(tmp_path, "b.yaml", GOOD.replace("howto-001", "howto-002"))

    cases = load_cases(tmp_path)

    assert [case.id for case in cases] == ["howto-001", "howto-002"]
    assert cases[0].expected_sources == ["docs/usage.md#web-ui-upload"]


def test_repeated_ids_are_an_error(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", GOOD)
    write(tmp_path, "b.yaml", GOOD)

    with pytest.raises(ValueError, match="howto-001"):
        load_cases(tmp_path)


def test_subdirectories_are_not_loaded(tmp_path: Path) -> None:
    """Отложенная выборка лежит в holdout/ и не должна попадать в обычный прогон."""
    write(tmp_path, "a.yaml", GOOD)
    (tmp_path / "holdout").mkdir()
    write(tmp_path / "holdout", "h.yaml", GOOD.replace("howto-001", "holdout-001"))

    assert [case.id for case in load_cases(tmp_path)] == ["howto-001"]
    assert [case.id for case in load_cases(tmp_path / "holdout")] == ["holdout-001"]


def test_typo_in_field_name_is_an_error(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", GOOD.replace("expected_sources:", "expected_source:"))

    with pytest.raises(ValidationError):
        load_cases(tmp_path)


def test_retrieval_case_without_sources_is_an_error() -> None:
    with pytest.raises(ValidationError, match="expected_sources"):
        Case(
            id="config-001",
            category="config",
            role="user",
            question="?",
            expected_behavior="answer",
        )


def test_out_of_scope_case_may_have_no_sources() -> None:
    case = Case(
        id="out_of_scope-001",
        category="out_of_scope",
        role="user",
        question="?",
        expected_behavior="abstain",
    )

    assert case.expected_sources == []


@pytest.mark.parametrize(
    ("behavior", "sources"),
    [("answer", []), ("abstain", ["docs/usage.md#x"])],
)
def test_out_of_scope_case_must_abstain_without_sources(
    behavior: str, sources: list[str]
) -> None:
    with pytest.raises(ValidationError, match="abstain"):
        Case(
            id="out_of_scope-001",
            category="out_of_scope",
            role="user",
            question="?",
            expected_behavior=behavior,
            expected_sources=sources,
        )


@pytest.mark.parametrize(
    "behavior", ["answer", "abstain", "escalate", "tool:get_tasks"]
)
def test_expected_behavior_values(behavior: str) -> None:
    Case(
        id="tool-001",
        category="tool",
        role="engineer",
        question="?",
        expected_behavior=behavior,
    )


def test_unknown_expected_behavior_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Case(
            id="tool-001",
            category="tool",
            role="user",
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
