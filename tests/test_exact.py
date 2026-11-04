import pytest
from eval_helpers import MANIFEST

from assistant.indexing.chunks import Chunk
from assistant.indexing.code import chunk_python
from assistant.indexing.docs import chunk_markdown
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.exact import MessageIndex
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore

CODE = """\
class ConsumerPlugin:
    def run(self, mime_type, asn):
        if not mime_type:
            self._fail(
                ConsumerStatusShortMessage.UNSUPPORTED_TYPE,
                f"Unsupported mime type {mime_type}",
            )
        log_msg = f"Not consuming {self.filename}: Given ASN {asn} already exists!"
        self._fail(ConsumerStatusShortMessage.ASN_ALREADY_EXISTS, log_msg)


def parse(path):
    raise ParseError(
        f"Error while parsing document {path}: the parser returned no text at all"
    )


def short(name):
    raise ValueError(f"File {name} not found")
"""

DOCS = """\
# Usage

## Consumption

Paperless consumes documents from the consumption directory and parses them.

## OCR language

Set PAPERLESS_OCR_LANGUAGE to the language of your documents.
"""


def code_chunks() -> list[Chunk]:
    return chunk_python(
        CODE, path="src/documents/consumer.py", ref="v3.4.2", repo_url="https://x"
    )


@pytest.fixture
def index() -> MessageIndex:
    return MessageIndex(code_chunks())


@pytest.mark.parametrize(
    "question",
    [
        "[2026-11-04 10:15:02,123] [ERROR] [paperless.consumer] [3f2a9c1d] "
        "Unsupported mime type application/octet-stream",
        "scan_0042.pdf: UNSUPPORTED MIME TYPE application/pdf",
        "у меня пишет\nUnsupported   mime\ntype image/heic что делать",
    ],
)
def test_message_is_found_in_log_lines_and_any_case(
    index: MessageIndex, question: str
) -> None:
    [hit] = index.search(question)

    assert hit.full
    assert hit.message.line == 6
    assert hit.chunk.source_id == "src/documents/consumer.py::ConsumerPlugin.run"


def test_message_built_in_a_variable_is_found(index: MessageIndex) -> None:
    [hit] = index.search(
        "scan.pdf: Not consuming scan.pdf: Given ASN 42 already exists!"
    )

    assert hit.message.line == 8


def test_truncated_message_falls_back_to_its_longest_part(index: MessageIndex) -> None:
    [hit] = index.search("Error while parsing document /tmp/a.pdf: the parser ret")

    assert not hit.full
    assert hit.chunk.source_id == "src/documents/consumer.py::parse"


@pytest.mark.parametrize(
    "question",
    [
        "как сделать чтобы paperless сам забирал почту",
        "почему не распознаётся русский текст",
        # «File {name} not found»: куски короткие, такой шаблон в поиск не идёт.
        "File backup.tar not found",
    ],
)
def test_ordinary_questions_do_not_match(index: MessageIndex, question: str) -> None:
    assert index.search(question) == []


def test_parts_too_far_apart_do_not_match(index: MessageIndex) -> None:
    question = "Not consuming scan.pdf: " + "x" * 300 + " Given ASN 42 already exists!"

    assert all(hit.message.line != 8 or not hit.full for hit in index.search(question))


def test_one_hit_per_chunk_and_longest_first(index: MessageIndex) -> None:
    question = "Unsupported mime type text/x and Not consuming a.pdf: Given ASN 1 already exists!"

    hits = index.search(question, limit=5)

    # Оба шаблона в одном куске: остаётся один, с бóльшим совпавшим текстом.
    assert [h.message.line for h in hits] == [8]


@pytest.fixture
def retriever(store: VectorStore, fake_embedder: Embedder) -> Retriever:
    chunks = [
        *chunk_markdown(DOCS, path="docs/usage.md", ref="v3.4.2", repo_url="https://x"),
        *code_chunks(),
    ]
    store.recreate(MANIFEST)
    store.upsert(
        chunks, fake_embedder.embed_documents([c.embedding_text() for c in chunks])
    )
    return Retriever(store, fake_embedder, ref="v3.4.2")


def test_exact_hit_goes_first(retriever: Retriever) -> None:
    hits = retriever.search("[ERROR] Unsupported mime type application/zip", k=3)

    assert hits[0].exact
    assert hits[0].chunk.source_id == "src/documents/consumer.py::ConsumerPlugin.run"
    assert len({h.chunk.source_id for h in hits}) == len(hits)


def test_without_exact_hits_search_is_the_same_as_vector_only(
    retriever: Retriever, store: VectorStore, fake_embedder: Embedder
) -> None:
    """Вопросы «как сделать» точный поиск не меняет: выдача та же, что у векторного."""
    vector_only = Retriever(store, fake_embedder, ref="v3.4.2", exact=False)
    question = "consumption directory documents"

    assert retriever.search(question, k=3) == vector_only.search(question, k=3)
    assert not any(h.exact for h in retriever.search(question, k=3))


def test_store_returns_only_chunks_with_messages(retriever: Retriever) -> None:
    assert len(retriever.messages) == 3
