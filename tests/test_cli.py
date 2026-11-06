import pytest
from eval_helpers import MANIFEST

from assistant.cli import INDEX_OUTDATED, format_plan, index_check, main, rebuild_reason
from assistant.config import Settings
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.store import VectorStore
from assistant.tools.search_recipes import SearchArgs


@pytest.mark.parametrize("command", ["nope", "sources"])
def test_unknown_command_is_rejected(command: str) -> None:
    with pytest.raises(SystemExit):
        main([command])


def test_index_without_embeddings_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ключ проверяется первым: до скачивания датасета и до Qdrant."""
    assert main(["index"]) == 1
    assert "CLOUD_RU_API_KEY" in capsys.readouterr().err


def test_index_is_rebuilt_only_when_needed(store: VectorStore) -> None:
    """index --if-needed: для деплоя; полная сборка стоит денег."""
    assert "не построен" in (
        rebuild_reason(store, ref="v3.4.2", model="fake-model") or ""
    )

    store.recreate(MANIFEST)

    assert rebuild_reason(store, ref="v3.4.2", model="fake-model") is None
    assert "v3.4.3" in (rebuild_reason(store, ref="v3.4.3", model="fake-model") or "")
    assert "other" in (rebuild_reason(store, ref="v3.4.2", model="other") or "")

    store.recreate(MANIFEST.model_copy(update={"sample": 1000, "seed": 1}))

    assert "выборке из 1000" in (
        rebuild_reason(store, ref="v3.4.2", model="fake-model") or ""
    )


def test_sample_index_is_kept_when_the_sample_is_wanted(store: VectorStore) -> None:
    """INDEX_SAMPLE на сервере: выборка того же размера и сида не пересобирается
    на каждом деплое, другая — пересобирается."""
    store.recreate(MANIFEST.model_copy(update={"sample": 10000, "seed": 7}))

    def reason(sample: int | None, seed: int = 7) -> str:
        return (
            rebuild_reason(
                store, ref="v3.4.2", model="fake-model", sample=sample, seed=seed
            )
            or ""
        )

    assert reason(10000) == ""
    assert "нужна выборка из 5000" in reason(5000)
    assert "сид 8" in reason(10000, seed=8)
    assert "нужен весь датасет" in reason(None)

    store.recreate(MANIFEST)

    assert "всём датасете, а нужна выборка из 10000" in reason(10000)


def test_search_without_embeddings_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["search", "курица без лука"])

    assert code == 1
    error = capsys.readouterr().err
    assert "CLOUD_RU_API_KEY" in error
    assert "уровня «Проект»" in error


def test_ask_without_llm_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUD_RU_API_KEY", "test-key")

    code = main(["ask", "как включить OCR"])

    assert code == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("provider", "key"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("gigachat", "GIGACHAT_API_KEY")],
)
def test_search_without_llm_key_explains_what_to_do(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key: str,
) -> None:
    """Разбор вопроса идёт через LLM: без ключа — понятная ошибка до Qdrant."""
    monkeypatch.setenv("CLOUD_RU_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", provider)

    assert main(["search", "курица без лука"]) == 1
    assert key in capsys.readouterr().err


def test_format_plan_shows_query_conditions_and_dropped_names() -> None:
    args = SearchArgs(
        query="курица без лука",
        conditions=IngredientFilter.of(
            include=[["курица", "филе куриное"], ["рис"]], exclude=["лук репчатый"]
        ),
        dropped=("куриное мясо",),
    )

    assert format_plan(args) == (
        "запрос: курица без лука\n"
        "с: курица или филе куриное; рис\n"
        "без: лук репчатый\n"
        "нет в словаре, отброшено: куриное мясо"
    )
    assert format_plan(SearchArgs("борщ", IngredientFilter())) == "запрос: борщ"


def test_index_check_builds_nothing_and_says_whether_a_build_is_needed(
    store: VectorStore,
) -> None:
    """Деплой строит индекс только с разрешением: сначала он спрашивает --check."""
    settings = Settings(dataset_revision="v3.4.2", embedding_model="fake-model")

    assert index_check(store, settings, sample=None, seed=1) == INDEX_OUTDATED
    store.recreate(MANIFEST)
    assert index_check(store, settings, sample=None, seed=1) == 0
    assert index_check(store, settings, sample=1000, seed=1) == INDEX_OUTDATED
    # Ключа эмбеддингов нет — проверке он и не нужен.
    assert not settings.cloud_ru_api_key.get_secret_value()
