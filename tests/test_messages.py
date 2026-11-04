import pytest

from assistant.indexing.code import chunk_python
from assistant.indexing.messages import extract_messages

SOURCE = """\
import logging

logger = logging.getLogger(__name__)


class Consumer:
    def run(self, mime_type, asn):
        if not mime_type:
            self._fail(
                ConsumerStatusShortMessage.UNSUPPORTED_TYPE,
                f"Unsupported mime type {mime_type}",
            )
        if asn:
            log_msg = f"Not consuming {self.filename}: Given ASN {asn} already exists!"
            if self.in_trash:
                log_msg += " Note: existing document is in the trash."
            self._fail(ConsumerStatusShortMessage.ASN_ALREADY_EXISTS, log_msg)

    def parse(self, path):
        logger.warning("Could not parse %s, trying fallback for %d pages", path, 3)
        raise ParseError(
            f"Error while parsing document {path}: "
            f"the parser returned no text",
        )

    def short(self, e):
        logger.error(f"Error: {e}")
        raise ValueError(_("Insufficient permissions to change owner."))


def helper(name):
    raise RuntimeError("Unable to load {} configuration file".format(name))
"""


def templates() -> dict[int, list[str]]:
    return {m.line: m.parts for m in extract_messages(SOURCE).messages}


def test_fstring_becomes_parts_with_holes() -> None:
    assert templates()[11] == ["Unsupported mime type "]


def test_message_built_in_a_variable_is_found_with_its_suffix() -> None:
    found = templates()

    assert found[14] == ["Not consuming ", ": Given ASN ", " already exists!"]
    assert found[16] == [" Note: existing document is in the trash."]


def test_percent_placeholders_of_logger_are_holes() -> None:
    assert templates()[20] == ["Could not parse ", ", trying fallback for ", " pages"]


def test_implicitly_joined_fstrings_are_one_message() -> None:
    assert templates()[22] == [
        "Error while parsing document ",
        ": the parser returned no text",
    ]


def test_gettext_and_format_are_unwrapped() -> None:
    found = templates()

    assert found[28] == ["Insufficient permissions to change owner."]
    assert found[32] == ["Unable to load ", " configuration file"]


def test_messages_with_too_little_text_are_skipped() -> None:
    """«Error: {e}» совпал бы с чем угодно."""
    assert 27 not in templates()


def test_kinds_and_coverage() -> None:
    extraction = extract_messages(SOURCE)

    kinds = {m.line: m.kind for m in extraction.messages}
    assert kinds[11] == "_fail" and kinds[20] == "log.warning" and kinds[22] == "raise"
    # 7 вызовов с текстом: два _fail, два логгера, три raise; у «Error: {e}» шаблона нет.
    assert (extraction.calls, extraction.covered) == (7, 6)


def test_variable_from_another_function_is_not_used() -> None:
    source = """\
def a():
    msg = "This message belongs to function a only"

def b(msg):
    raise ValueError(msg)
"""

    extraction = extract_messages(source)

    assert extraction.messages == []
    assert (extraction.calls, extraction.covered) == (1, 0)


@pytest.mark.parametrize(
    ("line", "source_id"),
    [(11, "m.py::Consumer.run"), (20, "m.py::Consumer.parse"), (32, "m.py::helper")],
)
def test_chunks_carry_the_messages_on_their_lines(line: int, source_id: str) -> None:
    chunks = chunk_python(SOURCE, path="m.py", ref="v3.4.2", repo_url="https://x")

    owner = next(c for c in chunks if any(m.line == line for m in c.messages))

    assert owner.source_id == source_id
    assert owner.start_line <= line <= owner.end_line


def test_holes_at_the_edges_are_remembered() -> None:
    found = {m.line: m for m in extract_messages(SOURCE).messages}

    assert found[11].tail_hole and not found[11].head_hole
    assert not found[22].head_hole and not found[22].tail_hole
