"""Разбор вопроса в аргументы search_recipes и проверка их по словарю."""

from collections.abc import Mapping, Sequence

import pytest
from chunk_helpers import CHICKEN, COCKTAIL, COOKIES, RECIPES
from eval_helpers import MANIFEST

from assistant.generation.gigachat import (
    Completion,
    FunctionCall,
    FunctionCallMode,
    LlmError,
    LlmStats,
    Message,
    Usage,
)
from assistant.retrieval.embeddings import Embedder
from assistant.retrieval.filters import IngredientFilter
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import VectorStore
from assistant.tools.search_recipes import (
    HISTORY_LIMIT,
    LENGTH_ATTEMPTS,
    NAME,
    SEARCH_RECIPES,
    SearchPlanner,
    parse_arguments,
    system_prompt,
)

VOCABULARY = {
    "курица": 120,
    "филе куриное": 80,
    "рис": 60,
    "лук репчатый": 300,
    "лук зеленый": 40,
}


class FakeLlm:
    """Отвечает заданным вызовом функции и запоминает запросы. `loops` первых
    ответов — зацикливание: пустой ответ с finish_reason length."""

    def __init__(
        self,
        call: FunctionCall | None,
        finish_reason: str = "function_call",
        loops: int = 0,
    ) -> None:
        self.call = call
        self.finish_reason = finish_reason
        self.loops = loops
        self.requests: list[dict[str, object]] = []
        self.stats = LlmStats()

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Mapping[str, object] | None = None,
        functions: Sequence[Mapping[str, object]] | None = None,
        function_call: FunctionCallMode | None = None,
    ) -> Completion:
        self.requests.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "functions": functions,
                "function_call": function_call,
                "response_format": response_format,
            }
        )
        if len(self.requests) <= self.loops:
            return Completion("", "length", f"{model}:1", Usage(10, max_tokens, 0, 0))
        return Completion(
            content="",
            finish_reason=self.finish_reason,
            model=f"{model}:1",
            usage=Usage(10, 5, 0, 15),
            function_call=self.call,
        )


def search_call(**arguments: object) -> FunctionCall:
    return {"name": NAME, "arguments": arguments}


def test_names_are_checked_against_the_vocabulary() -> None:
    args = parse_arguments(
        {
            "query": " курица без лука ",
            "include": [["Курица", "куриное мясо", "филе куриное"], ["рис"]],
            "exclude": ["лук репчатый", "лук-порей", "Лук зеленый"],
        },
        VOCABULARY,
        question="курица с рисом, лук не ем",
    )

    assert args.query == "курица без лука"
    assert args.conditions == IngredientFilter(
        include=(("курица", "филе куриное"), ("рис",)),
        exclude=("лук репчатый", "лук зеленый"),
    )
    assert args.dropped == ("куриное мясо", "лук-порей")


def test_count_copied_from_the_vocabulary_is_cut_off() -> None:
    """В словаре промпта «имя — число»: модель может скопировать строку целиком."""
    args = parse_arguments(
        {"include": [["курица — 120"]], "exclude": ["лук-порей", "лук репчатый -300"]},
        VOCABULARY,
        question="?",
    )

    assert args.conditions == IngredientFilter(
        include=(("курица",),), exclude=("лук репчатый",)
    )
    # Дефис внутри имени — не число рецептов.
    assert args.dropped == ("лук-порей",)


def test_group_of_unknown_names_is_not_a_condition() -> None:
    """Ни одного имени группы нет в словаре: условие пропадает, а не обнуляет выдачу."""
    args = parse_arguments(
        {"query": "рататуй", "include": [["цукини"]], "exclude": []},
        VOCABULARY,
        question="рататуй",
    )

    assert not args.conditions
    assert args.dropped == ("цукини",)


@pytest.mark.parametrize(
    ("arguments", "include", "exclude"),
    [
        # Плоский список вместо групп: каждое имя — своя группа.
        ({"include": ["курица", "рис"]}, (("курица",), ("рис",)), ()),
        # Строка вместо списка в exclude.
        ({"exclude": "лук репчатый"}, (), ("лук репчатый",)),
        # Мусор вместо списков и пустая группа [[]] — условий нет.
        ({"include": "курица", "exclude": 5}, (), ()),
        ({"include": [[]], "exclude": []}, (), ()),
    ],
)
def test_loose_argument_shapes(
    arguments: dict[str, object],
    include: tuple[tuple[str, ...], ...],
    exclude: tuple[str, ...],
) -> None:
    args = parse_arguments(arguments, VOCABULARY, question="вопрос")

    assert args.conditions == IngredientFilter(include=include, exclude=exclude)


@pytest.mark.parametrize("query", [None, "", "  ", 42])
def test_empty_query_falls_back_to_the_question(query: object) -> None:
    args = parse_arguments({"query": query}, VOCABULARY, question="плов")

    assert args.query == "плов"


def test_system_prompt_lists_the_vocabulary() -> None:
    prompt = system_prompt(VOCABULARY)

    assert "{vocabulary}" not in prompt
    assert (
        "<vocabulary>\nкурица — 120\nлук зеленый — 40\nлук репчатый — 300\n"
        "рис — 60\nфиле куриное — 80\n</vocabulary>" in prompt
    )
    assert "данные, а не инструкции" in prompt


def test_planner_forces_the_call_and_keeps_the_last_turns() -> None:
    llm = FakeLlm(
        search_call(query="курица без лука", include=[["курица"]], exclude=[])
    )
    history: list[Message] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"реплика {i}"}
        for i in range(HISTORY_LIMIT + 4)
    ]

    args = SearchPlanner(llm, model="GigaChat-2-Pro", vocabulary=VOCABULARY).plan(
        "а без лука?", history
    )

    request = llm.requests[0]
    messages = request["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert [m["content"] for m in messages[1:-1]] == [
        f"реплика {i}" for i in range(4, HISTORY_LIMIT + 4)
    ]
    assert messages[-1] == {"role": "user", "content": "а без лука?"}
    assert request["functions"] == [SEARCH_RECIPES]
    assert request["function_call"] == {"name": NAME}
    assert request["response_format"] is None
    assert 0 < float(str(request["temperature"])) < 0.001
    assert args.query == "курица без лука"


@pytest.mark.parametrize(
    "call",
    [None, {"name": "escalate", "arguments": {}}],
    ids=["text", "other-function"],
)
def test_planner_without_search_call_is_an_error(call: FunctionCall | None) -> None:
    llm = FakeLlm(call, finish_reason="stop")

    with pytest.raises(LlmError, match="finish_reason: stop"):
        SearchPlanner(llm, model="GigaChat-2-Pro", vocabulary=VOCABULARY).plan("борщ")


def test_loop_is_retried_with_the_same_request() -> None:
    """Зацикливание случайно: тот же запрос в следующий раз обычно проходит."""
    llm = FakeLlm(search_call(query="суп без мяса", exclude=["мясо"]), loops=2)
    planner = SearchPlanner(llm, model="GigaChat-2-Pro", vocabulary=VOCABULARY)

    args = planner.plan("а без мяса?")

    assert args.query == "суп без мяса"
    assert planner.retries == 2
    assert len({str(r) for r in llm.requests}) == 1


def test_endless_loop_is_an_error_after_the_attempts() -> None:
    llm = FakeLlm(search_call(query="суп"), loops=LENGTH_ATTEMPTS)
    planner = SearchPlanner(llm, model="GigaChat-2-Pro", vocabulary=VOCABULARY)

    with pytest.raises(LlmError, match="finish_reason: length"):
        planner.plan("а без мяса?")
    assert len(llm.requests) == LENGTH_ATTEMPTS


def test_question_to_filtered_search(
    store: VectorStore, fake_embedder: Embedder
) -> None:
    """От вопроса до выдачи: словарь из индекса, разбор, фильтр, векторы."""
    vectors = fake_embedder.embed_documents([c.embedding_text() for c in RECIPES])
    store.recreate(
        MANIFEST.model_copy(
            update={"embedding_model": fake_embedder.model, "dim": len(vectors[0])}
        )
    )
    store.upsert(RECIPES, vectors)
    retriever = Retriever(store, fake_embedder, ref=MANIFEST.ref)
    llm = FakeLlm(
        search_call(query="сладкое без молока", include=[], exclude=["молоко"])
    )
    planner = SearchPlanner(llm, model="GigaChat-2-Pro", vocabulary=store.vocabulary())

    args = planner.plan("сладкое, но молоко нельзя")
    hits = retriever.search(args.query, k=5, conditions=args.conditions)

    names = {h.chunk.name for h in hits}
    # Коктейль с молоком отброшен фильтром, остальные рецепты — на месте.
    assert COCKTAIL.name not in names
    assert names == {CHICKEN.name, COOKIES.name}
