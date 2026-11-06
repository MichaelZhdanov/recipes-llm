"""Ответ на вопрос со ссылками на рецепты. Два режима (решение 7).

Режим «всегда ищем» (always, контроль):

1. Разбор вопроса с историей диалога в запрос и условия (`SearchPlanner`,
   этап 2), поиск с фильтром — топ-K рецептов.
2. Рецепты, которые ассистент уже называл в диалоге, восстанавливаются по
   ссылкам povarenok.ru в его прошлых ответах: «сколько муки во втором?»
   опирается на них.
3. Каждый рецепт проходит маскирование ПДн и экранирование и попадает в
   промпт блоком `<recipe id="N">`. Ровно этот текст потом сверяется с цитатами.
4. Модель отвечает JSON по схеме (`citations.py`, промпт `prompts/recipes_v1.md`):
   утверждения с цитатами — строками рецепта, «не нашёл» или болтовня.
5. Код проверяет каждую цитату и числа. Утверждение без подтверждения
   удаляется, в источниках ответа остаются только процитированные рецепты.
6. Если не осталось ни одного подтверждённого утверждения, ответ — «не
   нашёл», и к нему прикладываются найденные рецепты: вдруг они всё же
   помогут. Если «не нашёл» сказала сама модель, подсказок нет.

Режим «модель решает» (auto, тест): модель получает search_recipes
(`tool_choice: auto`) и словарь индекса (промпт `recipes_auto_v1`) и сама
решает, искать ли и сколько раз (до MAX_SEARCHES). Рецепты прошлых ответов
она видит сразу, найденные приходят результатом функции с теми же
номерами, на которые потом ссылаются цитаты. Последний ответ — JSON по той
же схеме, проверка та же. На болтовню модель отвечает без поиска.

Порога по score поиска нет: решение «есть ответ или нет» принимает модель,
а код проверяет, что ответ подтверждён. Подобрать порог на 32 кейсах значило
бы подогнать его под набор.
"""

import dataclasses
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx2
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient

from assistant.config import AnswerMode, Settings
from assistant.generation.citations import (
    ANSWER_SCHEMA,
    CheckedStatement,
    ModelAnswer,
    SourceText,
    check_answer,
    numbers_in,
)
from assistant.generation.claude import Claude
from assistant.generation.gigachat import (
    DETERMINISTIC_TEMPERATURE,
    ChatModel,
    Completion,
    Message,
    call_message,
    function_result,
)
from assistant.generation.llm import llm_model, make_chat_model
from assistant.generation.prompt import load_prompt
from assistant.generation.review import EFFORT as REVIEW_EFFORT
from assistant.generation.review import Review, Reviewer, review_answer
from assistant.generation.revise import Reviser
from assistant.indexing.chunks import Chunk
from assistant.indexing.dataset import recipe_ids_in
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import Hit, VectorStore
from assistant.security.pii import mask_pii
from assistant.tools.search_recipes import (
    HISTORY_LIMIT,
    NAME,
    SEARCH_RECIPES,
    SearchArgs,
    SearchPlanner,
    parse_arguments,
    system_prompt,
)

# v2 (2026-11-06, решения автора): живой тон на «вы», эпитеты и немного эмодзи,
# вступление к ответу без фактов, явный список того, чего в данных нет.
# v3: заправка в любой форме, части блюда и роль ингредиента в процессе — нет
# в данных (уточнённое правило автора 2026-11-06).
PROMPT_VERSION = "recipes_v3"
AUTO_PROMPT_VERSION = "recipes_auto_v3"
# «Модель решает»: сколько поисков за один вопрос, потом — ответ без функций.
MAX_SEARCHES = 3
K = 5
# Сколько найденных рецептов приложить к ответу «не нашёл».
SUGGESTIONS = 3
# 6 утверждений с цитатами укладываются примерно в 1000 токенов, остальное — запас.
MAX_TOKENS = 2000

# Почему ответа нет. Первые два — честный отказ, остальные — поломки:
# - model_not_found — модель сама сказала, что в источниках ответа нет;
# - filtered_out — модель ответила, но ни одна цитата не подтвердилась;
# - format_error — ответ не по схеме;
# - blacklist — модель отказалась отвечать (ограничения GigaChat, отказ Claude);
# - length — ответ упёрся в MAX_TOKENS;
# - no_sources — поиск ничего не вернул (пустой индекс).
Reason = Literal[
    "model_not_found",
    "filtered_out",
    "format_error",
    "blacklist",
    "length",
    "no_sources",
    # Проверка Sonnet не подтвердила ни одного утверждения и после доработки.
    "review_out",
]


# finish_reason GigaChat, у которых есть своя причина; остальные (function_call,
# error) значат, что ответ не по формату.
_FINISH_REASONS: dict[str, Reason] = {"blacklist": "blacklist", "length": "length"}


class SourceLink(BaseModel):
    """Источник в ответе: рецепт и ссылка на его страницу на povarenok.ru."""

    number: int
    recipe_id: int
    name: str
    url: str


class AnswerStatement(BaseModel):
    text: str
    # Номера источников из Answer.sources.
    sources: list[int]


class Answer(BaseModel):
    """Ответ для пользователя: то, что видят CLI и `POST /ask`."""

    status: Literal["answer", "not_found", "chat"]
    reason: Reason | None
    # Ответ на болтовню или подсказка модели при «не нашёл»; без фактов о рецептах.
    message: str = ""
    statements: list[AnswerStatement]
    # У ответа — процитированные источники, у «не нашёл» — что может пригодиться.
    sources: list[SourceLink]
    # Модель с версией из ответа API; пусто, если модель не вызывалась.
    model: str
    prompt_version: str
    dataset_revision: str


@dataclass(frozen=True)
class Trace:
    """Всё, что нужно eval: что видела модель, что ответила и что прошло проверку."""

    hits: list[Hit]
    sources: list[SourceText]
    completion: Completion | None
    checked: list[CheckedStatement]
    retrieval_seconds: float
    generation_seconds: float
    # Во что разобран вопрос; None — без разбора (пересчёт старых прогонов).
    args: SearchArgs | None = None
    # Все поиски модели в режиме auto (в always — один, тот же, что args).
    searches: tuple[SearchArgs, ...] = ()
    # Проверка ответа перед отправкой (review.py); None — проверки не было.
    review: Review | None = None


@dataclass(frozen=True)
class AnswerResult:
    answer: Answer
    trace: Trace


_RECIPE_TAG = re.compile(r"<(/?recipe)", re.IGNORECASE)


def escape_source_text(text: str) -> str:
    """Рецепт не может закрыть свой блок: `</recipe>` в тексте становится `&lt;/recipe>`.

    Иначе текст с сайта мог бы «выйти» из блока данных и дописать промпт.
    Переводы строк не меняются, номера строк сохраняются.
    """
    return _RECIPE_TAG.sub(r"&lt;\1", text)


def prepare_sources(
    hits: Sequence[Hit], earlier: Sequence[Chunk] = ()
) -> list[SourceText]:
    """Тексты ровно в том виде, в каком они уйдут в модель: найденные сейчас,
    потом названные раньше в диалоге (без повторов).

    Маскирование здесь, а не только в клиенте LLM: цитаты сверяются
    с тем, что видела модель. Клиент маскирует ещё раз, это ничего не меняет.
    """
    chunks = {hit.chunk.recipe_id: hit.chunk for hit in hits}
    for chunk in earlier:
        chunks.setdefault(chunk.recipe_id, chunk)
    return [
        SourceText(
            number=number,
            chunk=chunk,
            text=escape_source_text(mask_pii(chunk.text).text),
        )
        for number, chunk in enumerate(chunks.values(), start=1)
    ]


def user_message(question: str, sources: Sequence[SourceText]) -> str:
    """Рецепты и вопрос."""
    # Название рецепта — первая строка текста, в атрибуты оно не идёт: это
    # текст с сайта, и только text проходит маскирование и экранирование.
    blocks = [f'<recipe id="{s.number}">\n{s.text}\n</recipe>' for s in sources]
    return "\n\n".join(
        [
            "Рецепты:",
            "\n\n".join(blocks) if blocks else "(ничего не нашлось)",
            f"Вопрос пользователя: {question}",
        ]
    )


def add_sources(
    sources: Sequence[SourceText], chunks: Sequence[Chunk]
) -> tuple[list[SourceText], list[SourceText]]:
    """Дописывает рецепты к уже показанным модели, не меняя их номера.

    Возвращает все источники и источники этих chunks (новые и уже бывшие):
    в режиме auto номера не должны сдвигаться между шагами, иначе цитаты
    модели указывали бы не на те рецепты.
    """
    result = list(sources)
    by_id = {s.chunk.recipe_id: s for s in result}
    found = []
    for chunk in chunks:
        if chunk.recipe_id not in by_id:
            source = SourceText(
                number=len(result) + 1,
                chunk=chunk,
                text=escape_source_text(mask_pii(chunk.text).text),
            )
            result.append(source)
            by_id[chunk.recipe_id] = source
        found.append(by_id[chunk.recipe_id])
    # Без повторов, по номеру: поиск мог вернуть рецепт дважды.
    return result, list({s.number: s for s in found}.values())


def auto_user_message(question: str, earlier: Sequence[SourceText]) -> str:
    """Вопрос и рецепты из прошлых ответов (режим auto: остальное найдёт модель)."""
    if not earlier:
        return f"Вопрос пользователя: {question}"
    blocks = [f'<recipe id="{s.number}">\n{s.text}\n</recipe>' for s in earlier]
    return "\n\n".join(
        [
            "Рецепты из прошлых ответов:",
            "\n\n".join(blocks),
            f"Вопрос пользователя: {question}",
        ]
    )


def earlier_recipe_ids(history: Sequence[Message]) -> list[int]:
    """id рецептов из ссылок в прошлых ответах ассистента."""
    return recipe_ids_in(
        "\n".join(m["content"] for m in history if m["role"] == "assistant")
    )


def _link(number: int, chunk: Chunk) -> SourceLink:
    return SourceLink(
        number=number, recipe_id=chunk.recipe_id, name=chunk.name, url=chunk.url
    )


def build_statements(
    checked: Sequence[CheckedStatement],
    sources: Sequence[SourceText],
) -> tuple[list[AnswerStatement], list[SourceLink]]:
    """Оставляет подтверждённые утверждения и собирает список источников.

    Один процитированный рецепт — один источник в ответе. Номера идут
    по первому упоминанию.
    """
    kept = [s for s in checked if s.kept]
    cited = [c.source for s in kept for c in s.citations if c.valid]
    order = {source: number for number, source in enumerate(dict.fromkeys(cited), 1)}
    chunks = {s.number: s.chunk for s in sources}
    links = [_link(number, chunks[source]) for source, number in order.items()]
    statements = []
    for statement in kept:
        cited = list(dict.fromkeys(c.source for c in statement.citations if c.valid))
        statements.append(
            AnswerStatement(
                text=statement.text, sources=[order[number] for number in cited]
            )
        )
    return statements, links


def suggestions(hits: Sequence[Hit], limit: int = SUGGESTIONS) -> list[SourceLink]:
    return [_link(n, h.chunk) for n, h in enumerate(hits[:limit], start=1)]


@dataclass(frozen=True)
class Checked:
    """Ответ модели после проверки кодом."""

    checked: list[CheckedStatement]
    reason: Reason | None
    chat: bool = False
    message: str = ""


def check_completion(completion: Completion, sources: Sequence[SourceText]) -> Checked:
    """Разбирает ответ модели и проверяет ссылки. Без сети: так же работает пересчёт eval."""
    if completion.finish_reason != "stop":
        return Checked(
            [], _FINISH_REASONS.get(completion.finish_reason, "format_error")
        )
    try:
        parsed = ModelAnswer.model_validate_json(completion.content)
    except ValidationError:
        return Checked([], "format_error")
    if parsed.status == "chat":
        return Checked([], None, chat=True, message=parsed.message)
    if parsed.status == "not_found":
        return Checked([], "model_not_found", message=parsed.message)
    checked = check_answer(parsed, sources)
    if not any(s.kept for s in checked):
        return Checked(checked, "filtered_out")
    # Вступление — живая фраза без фактов о рецептах. Число в нём проверить
    # нечем (цитат у вступления нет), поэтому такое вступление код убирает.
    intro = parsed.message.strip()
    return Checked(checked, None, message="" if numbers_in(intro) else intro)


def assemble(
    hits: list[Hit],
    sources: list[SourceText],
    completion: Completion | None,
    *,
    prompt_version: str,
    ref: str,
    retrieval_seconds: float = 0.0,
    generation_seconds: float = 0.0,
    args: SearchArgs | None = None,
    searches: Sequence[SearchArgs] = (),
) -> AnswerResult:
    """Собирает ответ из найденного и ответа модели (None — модель не вызывалась)."""
    result = (
        Checked([], "no_sources")
        if completion is None
        else check_completion(completion, sources)
    )
    checked, reason = result.checked, result.reason
    # У болтовни нет ни утверждений, ни ссылок. Если модель сама сказала «не
    # нашёл», найденное ей не подошло — подсказки из него только мешают
    # («курс доллара» → салат «777»). При остальных отказах подсказки есть.
    statements: list[AnswerStatement] = []
    links: list[SourceLink] = []
    if reason is not None:
        links = [] if reason == "model_not_found" else suggestions(hits)
    elif not result.chat:
        statements, links = build_statements(checked, sources)
    status: Literal["answer", "not_found", "chat"] = (
        "chat" if result.chat else "answer" if reason is None else "not_found"
    )
    answer = Answer(
        status=status,
        reason=reason,
        message=result.message,
        statements=statements,
        sources=links,
        model=completion.model if completion else "",
        prompt_version=prompt_version,
        dataset_revision=ref,
    )
    trace = Trace(
        hits=hits,
        sources=sources,
        completion=completion,
        checked=checked,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
        args=args,
        searches=tuple(searches) or ((args,) if args is not None else ()),
    )
    return AnswerResult(answer=answer, trace=trace)


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        llm: ChatModel,
        *,
        model: str,
        ref: str,
        mode: AnswerMode = "always",
        prompt_version: str | None = None,
        k: int = K,
        clock: Callable[[], float] = time.perf_counter,
        reviewer: Reviewer | None = None,
        reviser: Reviser | None = None,
    ) -> None:
        self.model = model
        self.mode = mode
        self.prompt_version = prompt_version or (
            AUTO_PROMPT_VERSION if mode == "auto" else PROMPT_VERSION
        )
        self._retriever = retriever
        self._llm = llm
        self._ref = ref
        self._k = k
        self._clock = clock
        # Словарь ингредиентов индекса: им разбирают вопрос в обоих режимах.
        self._vocabulary = retriever.vocabulary()
        self._planner = SearchPlanner(llm, model=model, vocabulary=self._vocabulary)
        # Проверка перед отправкой: оба или ни одного.
        self._reviewer = reviewer
        self._reviser = reviser

    def answer(self, question: str, history: Sequence[Message] = ()) -> AnswerResult:
        """Ответ с учётом до HISTORY_LIMIT прошлых реплик диалога."""
        history = list(history)[-HISTORY_LIMIT:]
        if self.mode == "auto":
            return self._reviewed(question, self._answer_auto(question, history))
        return self._reviewed(question, self._answer_always(question, history))

    def _reviewed(self, question: str, result: AnswerResult) -> AnswerResult:
        """Sonnet проверяет ответ; «нет» — один раз Haiku на доработку (review.py)."""
        if self._reviewer is None or self._reviser is None:
            return result
        if result.answer.status != "answer":
            return result
        trace = result.trace
        review = review_answer(
            question,
            trace.sources,
            [s for s in trace.checked if s.kept],
            result.answer.message,
            self._reviewer,
            self._reviser,
            self._clock,
        )
        trace = dataclasses.replace(trace, review=review)
        if not review.final:
            answer = result.answer.model_copy(
                update={
                    "status": "not_found",
                    "reason": "review_out",
                    "message": "",
                    "statements": [],
                    "sources": suggestions(trace.hits),
                }
            )
            return AnswerResult(answer=answer, trace=trace)
        statements, links = build_statements(review.final, trace.sources)
        answer = result.answer.model_copy(
            update={
                "statements": statements,
                "sources": links,
                "message": result.answer.message if review.intro_kept else "",
            }
        )
        return AnswerResult(answer=answer, trace=trace)

    def _answer_always(self, question: str, history: list[Message]) -> AnswerResult:
        started = self._clock()
        args = self._planner.plan(question, history)
        hits = self._retriever.search(args.query, k=self._k, conditions=args.conditions)
        earlier = self._retriever.recipes(earlier_recipe_ids(history))
        retrieved = self._clock()
        sources = prepare_sources(hits, earlier)
        completion = None
        if sources:
            completion = self._llm.chat(
                [
                    {"role": "system", "content": load_prompt(self.prompt_version)},
                    *history,
                    {"role": "user", "content": user_message(question, sources)},
                ],
                model=self.model,
                temperature=DETERMINISTIC_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "schema": ANSWER_SCHEMA,
                    "strict": True,
                },
            )
        return assemble(
            hits,
            sources,
            completion,
            prompt_version=self.prompt_version,
            ref=self._ref,
            retrieval_seconds=retrieved - started,
            generation_seconds=self._clock() - retrieved if completion else 0.0,
            args=args,
        )

    def _answer_auto(self, question: str, history: list[Message]) -> AnswerResult:
        """«Модель решает»: модель ⇄ search_recipes, до MAX_SEARCHES поисков."""
        started = self._clock()
        sources, earlier = add_sources(
            [], self._retriever.recipes(earlier_recipe_ids(history))
        )
        searching = self._clock() - started
        messages: list[Message] = [
            {
                "role": "system",
                "content": system_prompt(self._vocabulary, self.prompt_version),
            },
            *history,
            {"role": "user", "content": auto_user_message(question, earlier)},
        ]
        hits: dict[int, Hit] = {}
        searches: list[SearchArgs] = []
        completion = None
        for step in range(MAX_SEARCHES + 1):
            completion = self._llm.chat(
                messages,
                model=self.model,
                temperature=DETERMINISTIC_TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "schema": ANSWER_SCHEMA,
                    "strict": True,
                },
                functions=[SEARCH_RECIPES],
                # После MAX_SEARCHES поисков — только ответ.
                function_call="none" if step == MAX_SEARCHES else "auto",
            )
            call = completion.function_call
            if call is None or call["name"] != NAME:
                break
            args = parse_arguments(
                call["arguments"], self._vocabulary, question=question
            )
            searches.append(args)
            before = self._clock()
            found = self._retriever.search(
                args.query, k=self._k, conditions=args.conditions
            )
            searching += self._clock() - before
            for hit in found:
                hits.setdefault(hit.chunk.recipe_id, hit)
            sources, shown = add_sources(sources, [hit.chunk for hit in found])
            result = {"recipes": [{"id": s.number, "text": s.text} for s in shown]}
            messages += [call_message(completion), function_result(NAME, result)]
        total = self._clock() - started
        return assemble(
            list(hits.values()),
            sources,
            completion,
            prompt_version=self.prompt_version,
            ref=self._ref,
            retrieval_seconds=searching,
            generation_seconds=total - searching,
            args=searches[0] if searches else None,
            searches=searches,
        )


_REASON_TEXT: dict[Reason, str] = {
    "model_not_found": "Не нашёл ответа в найденных источниках",
    "filtered_out": "Не нашёл ответа, который подтверждают источники",
    "no_sources": "Не нашёл ничего по вопросу",
    "blacklist": "Модель не стала отвечать на этот вопрос",
    "length": "Не получилось собрать ответ: он не уместился в лимит модели",
    "format_error": "Не получилось собрать ответ: модель ответила не по формату",
    "review_out": "Не нашёл ответа, который подтверждают рецепты",
}


def _source_lines(link: SourceLink) -> list[str]:
    return [f"[{link.number}] {link.name}", f"    {link.url}"]


def render_markdown(answer: Answer) -> str:
    """Ответ текстом для консоли и чата: утверждения с [n] и ссылки на рецепты."""
    if answer.status == "chat":
        return answer.message
    if answer.reason is not None:
        lines = [f"{_REASON_TEXT[answer.reason]}."]
        if answer.message:
            lines.append(answer.message)
        if answer.sources:
            lines += ["", "Вот что может пригодиться:"]
    else:
        lines = [answer.message] if answer.message else []
        lines += [
            f"{s.text} " + "".join(f"[{n}]" for n in s.sources)
            for s in answer.statements
        ]
        lines += ["", "Рецепты:"]
    for link in answer.sources:
        lines += _source_lines(link)
    # Пометка о сервисе — условие Freemium GigaChat (п. 5.4 соглашения);
    # у Claude её нет (решение автора).
    if answer.model.startswith("GigaChat"):
        lines += [
            "",
            f"Ответ сгенерирован GigaChat ({answer.model}) по найденным рецептам.",
        ]
    return "\n".join(lines)


def make_answerer(
    settings: Settings,
    *,
    embeddings_http: httpx2.Client,
    llm_http: httpx2.Client,
    qdrant: QdrantClient,
) -> Answerer:
    """Собирает ассистента из настроек. HTTP-клиент LLM — `llm.make_http_client`:
    у GigaChat свой TLS-корень."""
    # Сначала то, что падает без ключей: ошибка понятна ещё до сетевых вызовов.
    embedder = make_embedder(settings, embeddings_http)
    llm = make_chat_model(settings, llm_http)
    retriever = Retriever(VectorStore(qdrant), embedder, ref=settings.dataset_revision)
    reviewer = reviser = None
    if settings.answer_review:
        # Проверяет Sonnet с effort high тем же ключом Anthropic; дорабатывает
        # та же модель, что отвечала.
        checker = Claude(
            api_key=settings.anthropic_api_key,
            api_url=settings.anthropic_api_url,
            client=llm_http,
            effort=REVIEW_EFFORT,
        )
        reviewer = Reviewer(checker)
        reviser = Reviser(llm, model=llm_model(settings))
    return Answerer(
        retriever,
        llm,
        model=llm_model(settings),
        ref=settings.dataset_revision,
        mode=settings.answer_mode,
        reviewer=reviewer,
        reviser=reviser,
    )
