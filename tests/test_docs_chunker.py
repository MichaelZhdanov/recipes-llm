from pathlib import Path

import pytest

from assistant.indexing.chunks import Chunk, blob_url, docs_site_url
from assistant.indexing.docs import chunk_docs, chunk_markdown, clean_heading, slugify

REPO = "https://github.com/example/fork"


def chunks_of(
    text: str, path: str = "docs/usage.md", max_chars: int = 2000
) -> list[Chunk]:
    return chunk_markdown(
        text, path=path, ref="v3.4.2", repo_url=REPO, max_chars=max_chars
    )


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("The consumption directory", "the-consumption-directory"),
        ("Paperless, a history", "paperless-a-history"),
        ("paperless-ngx 3.4.1", "paperless-ngx-341"),
        ("MariaDB/MySQL: Error: Illegal mix", "mariadbmysql-error-illegal-mix"),
    ],
)
def test_slugify_matches_python_markdown(heading: str, expected: str) -> None:
    assert slugify(heading) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "[`PAPERLESS_OCR_LANGUAGE=<lang>`](#PAPERLESS_OCR_LANGUAGE) {#PAPERLESS_OCR_LANGUAGE}",
            "PAPERLESS_OCR_LANGUAGE=<lang>",
        ),
        (
            "~~[`PAPERLESS_NLTK_DIR`](#PAPERLESS_NLTK_DIR)~~ {#PAPERLESS_NLTK_DIR}",
            "PAPERLESS_NLTK_DIR",
        ),
        (
            "_What's the general plan for Paperless-ngx?_",
            "What's the general plan for Paperless-ngx?",
        ),
        (
            "Database warns about `documents_tag_name_uniq`",
            "Database warns about documents_tag_name_uniq",
        ),
    ],
)
def test_clean_heading(raw: str, expected: str) -> None:
    assert clean_heading(raw) == expected


def test_sections_get_anchor_breadcrumb_and_exact_lines() -> None:
    text = "# Usage\n\nIntro.\n\n## Adding documents\n\n### The consumption directory\n\nPut files here.\n"

    chunks = chunks_of(text)

    assert [c.source_id for c in chunks] == [
        "docs/usage.md#usage",
        "docs/usage.md#the-consumption-directory",
    ]
    consume = chunks[1]
    assert consume.breadcrumb == "Usage > Adding documents > The consumption directory"
    assert (consume.start_line, consume.end_line) == (7, 9)
    assert consume.text == "### The consumption directory\n\nPut files here."


def test_heading_without_text_is_not_a_chunk_but_stays_in_breadcrumb() -> None:
    chunks = chunks_of("# Usage\n## Adding documents\n### Upload\nText.\n")

    assert [c.title for c in chunks] == ["Upload"]
    assert chunks[0].breadcrumb == "Usage > Adding documents > Upload"


def test_explicit_anchor_wins_over_slug() -> None:
    text = "## [`PAPERLESS_REDIS=<url>`](#PAPERLESS_REDIS) {#PAPERLESS_REDIS}\n\nBroker URL.\n"

    chunk = chunks_of(text, path="docs/configuration.md")[0]

    assert chunk.source_id == "docs/configuration.md#PAPERLESS_REDIS"
    assert chunk.title == "PAPERLESS_REDIS=<url>"


def test_html_heading_with_id_is_a_section() -> None:
    text = (
        '# Setup\n\n<h3 id="migration_ng">Migrating from Paperless-ng</h3>\n\nSteps.\n'
    )

    chunk = chunks_of(text, path="docs/setup.md")[-1]

    assert chunk.source_id == "docs/setup.md#migration_ng"
    assert chunk.title == "Migrating from Paperless-ng"


def test_repeated_headings_get_suffixes_like_the_site() -> None:
    text = "# Setup\n## Installation\nA.\n## Docker\n### Installation\nB.\n### Installation\nC.\n"

    anchors = [c.source_id.split("#")[1] for c in chunks_of(text, path="docs/setup.md")]

    assert anchors == ["installation", "installation_1", "installation_2"]


def test_hash_lines_inside_code_blocks_are_not_headings() -> None:
    text = (
        "# Advanced\n\n## Scripts\n\n```bash\n# install parsers\napt install foo\n```\n"
    )

    chunks = chunks_of(text, path="docs/advanced_usage.md")

    assert [c.title for c in chunks] == ["Scripts"]
    assert "# install parsers" in chunks[0].text


def test_comment_in_code_block_does_not_count_as_page_h1() -> None:
    text = (
        "---\ntitle: Home\n---\n\n```sh\n# not a heading\n```\n\n## Features\n\nMany.\n"
    )

    chunks = chunks_of(text, path="docs/index.md")

    assert chunks[-1].breadcrumb == "Home > Features"


def test_page_intro_uses_front_matter_title_and_has_no_anchor() -> None:
    text = "---\ntitle: Home\n---\n\n\nPaperless is a DMS.\n\n## Features\n\nMany.\n"

    intro = chunks_of(text, path="docs/index.md")[0]

    assert intro.source_id == "docs/index.md"
    assert intro.title == "Home"
    assert intro.start_line == 6
    assert intro.site_url == "https://docs.paperless-ngx.com/"


def test_long_section_is_split_into_parts_with_same_source_id() -> None:
    paragraphs = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(10))
    text = f"## Big\n\n{paragraphs}\n"

    parts = chunks_of(text, max_chars=300)

    assert len(parts) > 1
    assert {p.source_id for p in parts} == {"docs/usage.md#big"}
    lines = text.splitlines()
    for part in parts:
        assert "\n".join(lines[part.start_line - 1 : part.end_line]) == part.text


def test_list_without_blank_lines_is_still_split() -> None:
    items = "\n".join(f"- item {i} " + "y" * 60 for i in range(40))

    parts = chunks_of(f"## Changes\n{items}\n", path="docs/changelog.md", max_chars=500)

    assert len(parts) > 1
    assert max(len(p.text) for p in parts) <= 2 * 500 + 100


def test_code_block_is_never_split() -> None:
    code = "\n".join(["```", *[f"line {i}" for i in range(50)], "", "more", "```"])

    parts = chunks_of(f"## Example\n\n{code}\n", max_chars=100)

    assert sum(p.text.count("```") for p in parts) == 2
    assert all(p.text.count("```") in (0, 2) for p in parts)


def test_urls_point_to_fork_lines_and_site() -> None:
    assert blob_url(REPO, "v3.4.2", "docs/usage.md", 3, 9) == (
        "https://github.com/example/fork/blob/v3.4.2/docs/usage.md?plain=1#L3-L9"
    )
    assert blob_url(REPO, "v3.4.2", "src/a.py", 1, 2) == (
        "https://github.com/example/fork/blob/v3.4.2/src/a.py#L1-L2"
    )
    assert docs_site_url("docs/usage.md", "the-web-ui") == (
        "https://docs.paperless-ngx.com/usage/#the-web-ui"
    )
    assert (
        docs_site_url("docs/index.md", "features")
        == "https://docs.paperless-ngx.com/#features"
    )


def test_chunk_docs_reads_all_markdown_including_changelog(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/usage.md").write_text("# Usage\n\nText.\n")
    (tmp_path / "docs/changelog.md").write_text("# Changelog\n\n## 3.4.1\n\n- fix\n")

    chunks = chunk_docs(tmp_path, ref="v3.4.2", repo_url=REPO)

    assert {c.path for c in chunks} == {"docs/usage.md", "docs/changelog.md"}
    assert all(c.kind == "doc" and c.ref == "v3.4.2" for c in chunks)
