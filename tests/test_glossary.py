import os
from pathlib import Path

import pytest
from glossary_candidates import PO_FILE, frequency, is_term, parse_po

from assistant.generation.glossary import (
    fix_terms,
    load_glossary,
    parse_glossary,
    render_terms,
    terms_in,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "paperless-src" / "v3.4.2"

PO = """\
msgid ""
msgstr ""
"Language: ru_RU\\n"

#: documents/models.py:40
msgid "correspondent"
msgstr "корреспондент"

#: documents/models.py:41 documents/views.py:7
msgid "storage "
"path"
msgstr "путь к "
"хранилищу"

#: documents/views.py:90
msgid "Download"
msgstr "Загрузка"

#: documents/models.py:99
msgid "no translation yet"
msgstr ""
"""


def test_po_parser_joins_lines_and_keeps_references() -> None:
    entries = list(parse_po(PO))

    assert [(e.msgid, e.msgstr) for e in entries] == [
        ("correspondent", "корреспондент"),
        ("storage path", "путь к хранилищу"),
        ("Download", "Загрузка"),
    ]
    assert entries[1].references == ("documents/models.py:41", "documents/views.py:7")


def test_candidates_are_model_terms_only() -> None:
    entries = list(parse_po(PO))

    assert [e.msgid for e in entries if is_term(e)] == ["correspondent", "storage path"]
    # Частота считается по слову целиком: «Tags» и «tagged» — другие слова.
    assert frequency("tag", "Tags, tag; tagged") == 1


def test_glossary_file_has_unique_terms_with_translations() -> None:
    glossary = load_glossary()

    assert len({t.en for t in glossary}) == len(glossary)
    assert all(t.en and t.ru for t in glossary)
    volume = next(t for t in glossary if t.en == "volume")
    assert volume.ru == "том" and volume.po_msgid is None


def test_paperless_terms_are_taken_from_official_translation_as_is() -> None:
    """Решение автора: термины Paperless — из django.po тега как есть."""
    po = SOURCES / PO_FILE
    if not po.exists():
        message = "нет исходников Paperless: выполните make sources"
        if os.environ.get("REQUIRE_PAPERLESS_SOURCES") == "1":
            pytest.fail(message)
        pytest.skip(message)
    official = {e.msgid: e.msgstr for e in parse_po(po.read_text(encoding="utf-8"))}

    wrong = [
        f"{t.en}: «{t.ru}», а в переводе «{official.get(t.po_msgid)}»"
        for t in load_glossary()
        if t.po_msgid is not None
        and t.ru.lower() not in official.get(t.po_msgid, "").lower()
    ]

    assert not wrong, "\n".join(wrong)


def test_terms_are_found_as_whole_words_with_plural() -> None:
    glossary = parse_glossary(
        "en\tru\tpo_msgid\tnote\n"
        "volume\tтом\t-\tтом Docker\n"
        "storage path\tпуть к хранилищу\tstorage path\t\n"
        "tag\tтег\ttag\t\n"
    )

    found = terms_in(
        ["Back up the docker volumes.", "Set the storage\npath here."], glossary
    )

    assert [t.en for t in found] == ["volume", "storage path"]
    # «tagged» — не «tag»: термин ищется целым словом.
    assert terms_in(["documents are tagged"], glossary) == []
    assert (
        render_terms(found)
        == "- volume — том (том Docker)\n- storage path — путь к хранилищу"
    )


VOLUME = parse_glossary("volume\tтом\t-\tтом Docker\tобъём|объем\n")


@pytest.mark.parametrize(
    ("text", "fixed"),
    [
        (
            "Документы в Docker-объёме `paperless_media`.",
            "Документы в Docker-томе `paperless_media`.",
        ),
        ("Объём управляется Docker.", "Том управляется Docker."),
        ("копии объемов баз данных", "копии томов баз данных"),
    ],
)
def test_wrong_translation_is_fixed_with_its_ending(text: str, fixed: str) -> None:
    assert fix_terms(text, VOLUME)[0] == fixed


@pytest.mark.parametrize(
    "text",
    [
        "Нужен объёмный диск.",  # прилагательное — не термин
        "Задайте `объём` как есть.",  # в обратных кавычках не трогаем
    ],
)
def test_other_words_are_left_alone(text: str) -> None:
    assert fix_terms(text, VOLUME) == (text, [])


def test_fixes_are_reported() -> None:
    assert fix_terms("Docker-объёма", VOLUME)[1] == [("объёма", "тома")]
