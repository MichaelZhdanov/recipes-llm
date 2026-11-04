import random
from pathlib import Path

from evals.dataset import Case
from evals.generate_errors import (
    MAX_PER_FILE,
    Template,
    cases,
    collect_templates,
    fill,
    pick,
    render,
)

SOURCE = """\
def consume(self, mime_type):
    self._fail(ShortMessage.UNSUPPORTED, f"Unsupported mime type {mime_type}")
    logger.warning("Consuming duplicate %s: document already exists", self.name)
    logger.debug("Debug details about the consumption of %s", self.name)
    raise ValueError("The same message appears in two places")


def other():
    raise ValueError("The same message appears in two places")
"""


def test_collects_visible_unique_messages(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "consumer.py").write_text(SOURCE)

    templates = collect_templates(tmp_path, "v3.4.2", "https://x")

    # debug не берётся, а повторяющийся текст неоднозначен по месту.
    assert [(t.line, t.kind) for t in templates] == [(2, "_fail"), (3, "log.warning")]
    assert templates[0].tail_hole and not templates[0].head_hole


def template(
    parts: tuple[str, ...], kind: str = "_fail", tail: bool = False
) -> Template:
    return Template("src/documents/consumer.py", 478, kind, parts, False, tail)


def test_holes_get_plausible_values() -> None:
    rng = random.Random(1)

    assert fill(template(("Unsupported mime type ",), tail=True), rng) == (
        "Unsupported mime type application/octet-stream"
    )
    assert fill(template(("Not consuming x: Given ASN ", " already exists!")), rng) == (
        "Not consuming x: Given ASN 42 already exists!"
    )


def test_forms_wrap_the_message_like_users_do() -> None:
    t = template(("Unsupported mime type ",), tail=True)
    rng = random.Random(1)

    assert render(t, "строка лога", rng).startswith(
        "[2026-11-04 10:15:02,123] [ERROR] [paperless.consumer] [3f2a9c1d] Unsupported mime type"
    )
    assert render(t, "ошибка задачи", rng).startswith("scan_0042.pdf: Unsupported")
    assert render(t, "вопрос", rng).startswith(
        "В логах paperless вот такое: «Unsupported"
    )
    assert len(render(t, "обрезано", rng)) < len(render(t, "сообщение", rng))


def test_pick_is_balanced_capped_per_file_and_disjoint() -> None:
    templates = [
        Template(
            f"src/f{n % 4}.py",
            n,
            kind,
            (f"message number {n} of kind {kind}",),
            False,
            False,
        )
        for n, kind in enumerate(["_fail", "log.error", "log.warning", "raise"] * 6)
    ]
    rng = random.Random(0)

    first = pick(templates, 8, rng, set())
    second = pick(templates, 4, rng, set(first))

    assert len(first) == 8 and len(second) == 4
    assert not set(first) & set(second)
    assert (
        max(sum(t.path == p for t in first) for p in {t.path for t in first})
        <= MAX_PER_FILE
    )
    assert {t.kind for t in first} == {"_fail", "log.error", "log.warning", "raise"}


def test_generated_cases_follow_the_schema() -> None:
    items = cases(
        [template(("Unsupported mime type ",), tail=True)], "error", random.Random(0)
    )

    case = Case.model_validate(items[0])

    assert case.expected_line == "src/documents/consumer.py:478"
    assert case.category == "error"
