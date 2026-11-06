"""CI-гейт: сохранённые ответы модели проходят проверку цитат так же, как в прогоне.

Ключей API в CI нет. Фикстура (`python -m evals.saved_answers RAW_JSON`) —
сырые ответы модели и id рецептов, которые она видела; тексты рецептов
собираются из датасета тем же кодом, что и в ассистенте. Если поменялись
`citations.py`, вид текста рецепта или маскирование ПДн, и цитаты перестали
проверяться так же, — тест падает на PR.

Без датасета тест пропускается локально; в CI датасет скачивается (`make data`),
и REQUIRE_DATASET=1 превращает пропуск в ошибку.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from assistant.config import Settings
from assistant.generation.answer import check_completion, escape_source_text
from assistant.generation.citations import SourceText
from assistant.generation.gigachat import Completion, Usage
from assistant.indexing.chunks import recipe_chunk
from assistant.indexing.dataset import DATASET_FILE, Dataset, read_dataset
from assistant.security.pii import mask_pii

FIXTURE = Path(__file__).parent / "fixtures" / "saved_answers.json"


@pytest.fixture(scope="module")
def saved() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def dataset(saved: dict[str, Any]) -> Dataset:
    settings = Settings()
    assert saved["dataset_revision"] == settings.dataset_revision, (
        "фикстура снята на другой ревизии датасета: пересоберите её"
    )
    path = settings.data_path / settings.dataset_revision / DATASET_FILE
    if not path.exists():
        if os.environ.get("REQUIRE_DATASET"):
            pytest.fail(f"нет датасета {path}: make data")
        pytest.skip("нет датасета: make data")
    return read_dataset(path)


def sources(recipe_ids: list[int], dataset: Dataset, ref: str) -> list[SourceText]:
    """Тексты рецептов ровно так, как они ушли в модель (prepare_sources)."""
    by_id = {r.id: r for r in dataset.recipes}
    result = []
    for number, recipe_id in enumerate(recipe_ids, start=1):
        chunk = recipe_chunk(by_id[recipe_id], ref)
        result.append(
            SourceText(number, chunk, escape_source_text(mask_pii(chunk.text).text))
        )
    return result


def test_fixture_covers_both_kept_and_dropped_statements(saved: dict[str, Any]) -> None:
    kept = [k for a in saved["answers"] for k in a["expected"]["kept"]]
    assert True in kept and False in kept
    assert {a["expected"]["status"] for a in saved["answers"]} >= {"answer", "chat"}


def test_saved_answers_pass_the_citation_check_as_in_the_run(
    saved: dict[str, Any], dataset: Dataset
) -> None:
    differ = []
    for answer in saved["answers"]:
        completion = Completion(
            answer["content"], answer["finish_reason"], "saved", Usage(0, 0, 0, 0)
        )
        checked = check_completion(
            completion,
            sources(answer["recipe_ids"], dataset, saved["dataset_revision"]),
        )
        kept = [s.kept for s in checked.checked]
        status = (
            "chat"
            if checked.chat
            else "answer"
            if checked.reason is None
            else "not_found"
        )
        if kept != answer["expected"]["kept"] or status != answer["expected"]["status"]:
            differ.append(
                f"{answer['id']}, прогон {answer['run']}: {kept} вместо {answer['expected']['kept']}"
            )
    assert not differ, "проверка цитат изменилась:\n" + "\n".join(differ)
