from eval_helpers import CHUNK

from assistant.retrieval.store import Hit
from assistant.router import route


def test_exact_hit_means_error_question() -> None:
    assert route([Hit(CHUNK, 0.5), Hit(CHUNK, 1.0, exact=True, line=12)]) == "error"


def test_without_exact_hit_question_is_general() -> None:
    assert route([Hit(CHUNK, 0.5)]) == "general"
    assert route([]) == "general"
