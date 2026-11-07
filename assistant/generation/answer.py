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
from typing import Literal, Protocol

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
    foreign_links,
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
from assistant.retrieval.stock import (
    NEAR_CANDIDATES,
    Stock,
    known_stock,
    near_conditions,
    near_hits,
    strict_conditions,
)
from assistant.retrieval.store import Hit, VectorStore
from assistant.security.pii import mask_pii
from assistant.tools.search_recipes import (
    HISTORY_LIMIT,
    NAME,
    SEARCH_RECIPES,
    Escalation,
    SearchArgs,
    SearchPlanner,
    parse_arguments,
    system_prompt,
)

# v2 (2026-11-06, решения автора): живой тон на «вы», эпитеты и немного эмодзи,
# вступление к ответу без фактов, явный список того, чего в данных нет.
# v3: заправка в любой форме, части блюда и роль ингредиента в процессе — нет
# в данных (уточнённое правило автора 2026-11-06).
# v4 (2026-11-07): строже — проверки перед отправкой нет. Раздел «Как выбрать
# рецепты» («к вину» — не блюдо на вине) и проверка утверждения по пунктам с
# самопроверкой: закономерности ошибок писателя на эталонном наборе
# (evals/reports/2026-11-07-writer-errors.md).
# v5 (этап 6): блок <stock> — что есть дома; чего не хватает, показывает код.
PROMPT_VERSION = "recipes_v5"
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
# Реплика вместо болтовни модели, которую нельзя показать (чужая ссылка) или
# которая пуста: пустое сообщение Telegram не отправит.
CHAT_FALLBACK = "Я подбираю рецепты из povarenok.ru — напишите, что хотите приготовить."


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
    """Ответ для пользователя: то, что видят CLI и `POST /ask`.

    status "escalate" — разговор надо передать человеку (повод и сводка — в
    escalation): поиска и ответа модели нет, передачу делает канал (бот)."""

    status: Literal["answer", "not_found", "chat", "escalate"]
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
    escalation: Escalation | None = None
    # Пользователь назвал аллергию: канал добавит предупреждение (решение 66).
    allergy: bool = False
    # Подбор по складу (этап 6): чего не хватает рецепту в расширенном поиске,
    # recipe_id → продукты; считает код по строкам рецепта.
    missing: dict[int, list[str]] = {}
    # Строгий подбор: есть рецепты, которым не хватает 1–4 продуктов, — канал
    # предложит расширенный поиск.
    near_available: bool = False


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
    # Первая попытка, если её рецепты оказались не по вопросу и ассистент искал
    # заново (Answerer._researched); None — нового поиска не было.
    first: Trace | None = None
    # Склад, по которому подбирали (этап 6); None — подбор по всем рецептам.
    stock: Stock | None = None
    # Строгий подбор: что предложит расширенный поиск — рецепт и чего ему не
    # хватает (для eval: без второго вызова модели).
    stock_near: tuple[tuple[Chunk, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class StockRequest:
    """Подбор «из того, что есть»: названия продуктов со склада (Grocy);
    near — расширенный поиск, где не хватает 1–4 продуктов."""

    names: tuple[str, ...]
    near: bool = False


@dataclass(frozen=True)
class StockHits:
    hits: list[Hit]
    stock: Stock
    missing: dict[int, list[str]]
    near_available: bool
    near: tuple[tuple[Chunk, tuple[str, ...]], ...] = ()


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


def user_message(
    question: str, sources: Sequence[SourceText], stock: Sequence[str] = ()
) -> str:
    """Рецепты и вопрос; stock — что есть у пользователя дома (этап 6)."""
    # Название рецепта — первая строка текста, в атрибуты оно не идёт: это
    # текст с сайта, и только text проходит маскирование и экранирование.
    blocks = [f'<recipe id="{s.number}">\n{s.text}\n</recipe>' for s in sources]
    parts = [
        "Рецепты:",
        "\n\n".join(blocks) if blocks else "(ничего не нашлось)",
    ]
    if stock:
        # Имена — из словаря индекса (retrieval/stock.py): из блока тегом не выйти.
        parts.append(
            "Что есть у пользователя дома:\n<stock>\n" + "\n".join(stock) + "\n</stock>"
        )
    parts.append(f"Вопрос пользователя: {question}")
    return "\n\n".join(parts)


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
    # Текст модели без цитат: ссылку в нём проверить нечем, кроме списка
    # разрешённых (citations.foreign_links). С чужой ссылкой текст не идёт.
    safe = parsed.message.strip()
    if foreign_links(safe, sources):
        safe = ""
    if parsed.status == "chat":
        return Checked([], None, chat=True, message=safe or CHAT_FALLBACK)
    if parsed.status == "not_found":
        return Checked([], "model_not_found", message=safe)
    checked = check_answer(parsed, sources)
    if not any(s.kept for s in checked):
        return Checked(checked, "filtered_out")
    # Вступление — живая фраза без фактов о рецептах. Число в нём проверить
    # нечем (цитат у вступления нет), поэтому такое вступление код убирает.
    return Checked(checked, None, message="" if numbers_in(safe) else safe)


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
        allergy=args is not None and args.allergy,
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


# Стадии ответа для канала (бот показывает «Ищу рецепты…», «Сверяю…»):
# search — разбор вопроса и поиск, answer — модель пишет ответ, review — проверка,
# research — рецепты оказались не по вопросу, ищем заново.
Stage = Literal["search", "answer", "review", "research"]
Progress = Callable[[Stage], None]


class Planner(Protocol):
    def plan(
        self, question: str, history: Sequence[Message] = (), note: str = ""
    ) -> SearchArgs: ...


def _notify(progress: Progress | None, stage: Stage) -> None:
    if progress is not None:
        progress(stage)


def _user_questions(history: Sequence[Message]) -> list[str]:
    """Прошлые вопросы пользователя: проверке — чтобы понять уточнение."""
    return [m["content"] for m in history if m["role"] == "user"]


def _rejected_recipes(trace: Trace) -> list[int]:
    """id рецептов, на которые опирались утверждения «не по вопросу»."""
    assert trace.review is not None
    kept = dict(enumerate((s for s in trace.checked if s.kept), start=1))
    by_number = {s.number: s.chunk.recipe_id for s in trace.sources}
    ids = (
        by_number[citation.source]
        for n in trace.review.irrelevant
        for citation in kept[n].citations
        if citation.valid and citation.source in by_number
    )
    return list(dict.fromkeys(ids))


def _with_review(answer: Answer, trace: Trace, review: Review) -> AnswerResult:
    """Ответ после проверки: только одобренные утверждения; ничего — «не нашёл»."""
    if not review.final:
        refused = answer.model_copy(
            update={
                "status": "not_found",
                "reason": "review_out",
                "message": "",
                "statements": [],
                "sources": suggestions(trace.hits),
            }
        )
        return AnswerResult(answer=refused, trace=trace)
    statements, links = build_statements(review.final, trace.sources)
    reviewed = answer.model_copy(
        update={
            "statements": statements,
            "sources": links,
            "message": answer.message if review.intro_kept else "",
        }
    )
    return AnswerResult(answer=reviewed, trace=trace)


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
        planner: Planner | None = None,
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
        # Разбор вопроса в аргументы поиска; другой — в eval: повтор ответа по
        # сохранённому разбору (evals/replay_answers.py).
        self._planner: Planner = planner or SearchPlanner(
            llm, model=model, vocabulary=self._vocabulary
        )
        # Проверка перед отправкой: оба или ни одного.
        self._reviewer = reviewer
        self._reviser = reviser

    def answer(
        self,
        question: str,
        history: Sequence[Message] = (),
        progress: Progress | None = None,
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        """Ответ с учётом до HISTORY_LIMIT прошлых реплик диалога.

        progress — для канала: вызывается в начале каждой стадии. stock —
        подбор «из того, что есть» (этап 6): только в режиме «всегда ищем»."""
        history = list(history)[-HISTORY_LIMIT:]
        if self.mode == "auto" and stock is None:
            result = self._answer_auto(question, history, progress)
        else:
            result = self._answer_always(question, history, progress, stock=stock)
        return self._reviewed(question, history, result, progress)

    def _reviewed(
        self,
        question: str,
        history: list[Message],
        result: AnswerResult,
        progress: Progress | None = None,
    ) -> AnswerResult:
        """Sonnet проверяет ответ; доработка — одна (review.py):

        - утверждения с «нет» по опоре на рецепт — Haiku правит текст;
        - если не осталось ни одного рецепта по вопросу — поиск заново
          (`_researched`); это и есть доработка, второй правки после него нет;
        - Haiku сам сказал «не нашёл», хотя рецепты были, — тоже поиск заново:
          среди найденных не оказалось ответа («к вину» — а нашлись блюда,
          тушённые в вине).
        """
        if self._reviewer is None or self._reviser is None:
            return result
        context = _user_questions(history)
        trace = result.trace
        if (
            result.answer.reason == "model_not_found"
            and trace.sources
            and self.mode == "always"
        ):
            names = ", ".join(f"«{s.chunk.name}»" for s in trace.sources)
            note = f"Среди найденных рецептов нет ответа на вопрос: {names}."
            rejected = [s.chunk.recipe_id for s in trace.sources]
            return self._researched(
                question, history, context, trace, progress, note, rejected
            )
        if result.answer.status != "answer":
            return result
        _notify(progress, "review")
        review = self._review(question, context, result, self._reviser)
        trace = dataclasses.replace(trace, review=review)
        if not review.final and review.irrelevant and self.mode == "always":
            note = "\n".join(dict.fromkeys(review.irrelevant.values()))
            return self._researched(
                question,
                history,
                context,
                trace,
                progress,
                note,
                _rejected_recipes(trace),
            )
        return _with_review(result.answer, trace, review)

    def _review(
        self,
        question: str,
        context: Sequence[str],
        result: AnswerResult,
        reviser: Reviser | None,
    ) -> Review:
        assert self._reviewer is not None
        trace = result.trace
        return review_answer(
            question,
            trace.sources,
            [s for s in trace.checked if s.kept],
            result.answer.message,
            self._reviewer,
            reviser,
            self._clock,
            context=context,
        )

    def _researched(
        self,
        question: str,
        history: list[Message],
        context: Sequence[str],
        first: Trace,
        progress: Progress | None,
        note: str,
        rejected: Sequence[int],
    ) -> AnswerResult:
        """Рецепты ответа не по вопросу («к вину» — а нашлись блюда, тушённые в
        вине): вопрос разбирается заново с замечанием (note), отклонённые
        рецепты в поиск не попадают, новый ответ проверяется — уже без правки
        текста, «нет» удаляется. Нового поиска после нового поиска нет."""
        _notify(progress, "research")
        result = self._answer_always(question, history, None, note=note, skip=rejected)
        if result.answer.status != "answer":
            return AnswerResult(
                answer=result.answer,
                trace=dataclasses.replace(result.trace, first=first),
            )
        review = self._review(question, context, result, None)
        trace = dataclasses.replace(result.trace, review=review, first=first)
        return _with_review(result.answer, trace, review)

    def _answer_always(
        self,
        question: str,
        history: list[Message],
        progress: Progress | None,
        *,
        note: str = "",
        skip: Sequence[int] = (),
        stock: StockRequest | None = None,
    ) -> AnswerResult:
        """note и skip — для нового поиска после проверки (`_researched`)."""
        _notify(progress, "search")
        started = self._clock()
        args = self._planner.plan(question, history, note)
        if args.escalation is not None:
            # Человеку: искать и отвечать не нужно (решение 65).
            return self._escalated(args, self._clock() - started)
        shown = earlier_recipe_ids(history)
        # «А ещё?» — уже показанные рецепты не ищутся и не подмешиваются.
        skip = [*skip, *shown] if args.other else list(skip)
        found = self._stock_hits(args, skip, stock) if stock else None
        hits = (
            found.hits
            if found
            else self._retriever.search_all(
                args.all_queries, k=self._k, conditions=args.conditions, skip=skip
            )
        )
        earlier = (
            []
            if args.other
            else [c for c in self._retriever.recipes(shown) if c.recipe_id not in skip]
        )
        retrieved = self._clock()
        sources = prepare_sources(hits, earlier)
        completion = None
        if sources:
            _notify(progress, "answer")
            completion = self._llm.chat(
                [
                    {"role": "system", "content": load_prompt(self.prompt_version)},
                    *history,
                    {
                        "role": "user",
                        "content": user_message(
                            question, sources, found.stock.names if found else ()
                        ),
                    },
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
        result = assemble(
            hits,
            sources,
            completion,
            prompt_version=self.prompt_version,
            ref=self._ref,
            retrieval_seconds=retrieved - started,
            generation_seconds=self._clock() - retrieved if completion else 0.0,
            args=args,
        )
        if found is None:
            return result
        answer = result.answer.model_copy(
            update={
                "missing": {
                    link.recipe_id: found.missing[link.recipe_id]
                    for link in result.answer.sources
                    if link.recipe_id in found.missing
                },
                "near_available": found.near_available,
            }
        )
        trace = dataclasses.replace(
            result.trace, stock=found.stock, stock_near=found.near
        )
        return AnswerResult(answer, trace)

    def _stock_hits(
        self, args: SearchArgs, skip: Sequence[int], request: StockRequest
    ) -> StockHits:
        """Подбор по складу: строгий (всё со склада) или расширенный (не хватает
        1–4). В строгом заодно проверяется, есть ли что предложить расширенным."""
        stock = known_stock(request.names, self._vocabulary)
        near = near_hits(
            self._retriever.search_all(
                args.all_queries,
                k=NEAR_CANDIDATES,
                conditions=near_conditions(args.conditions, stock),
                skip=skip,
            ),
            stock,
            limit=self._k,
        )
        if request.near:
            return StockHits(
                [hit for hit, _ in near],
                stock,
                {hit.chunk.recipe_id: list(lacking) for hit, lacking in near},
                near_available=False,
            )
        strict = self._retriever.search_all(
            args.all_queries,
            k=self._k,
            conditions=strict_conditions(args.conditions, stock),
            skip=skip,
        )
        return StockHits(
            strict,
            stock,
            {},
            near_available=bool(near),
            near=tuple((hit.chunk, lacking) for hit, lacking in near),
        )

    def _escalated(self, args: SearchArgs, seconds: float) -> AnswerResult:
        answer = Answer(
            status="escalate",
            reason=None,
            statements=[],
            sources=[],
            model="",
            prompt_version=self.prompt_version,
            dataset_revision=self._ref,
            escalation=args.escalation,
        )
        trace = Trace(
            hits=[],
            sources=[],
            completion=None,
            checked=[],
            retrieval_seconds=seconds,
            generation_seconds=0.0,
            args=args,
            searches=(args,),
        )
        return AnswerResult(answer=answer, trace=trace)

    def _answer_auto(
        self, question: str, history: list[Message], progress: Progress | None
    ) -> AnswerResult:
        """«Модель решает»: модель ⇄ search_recipes, до MAX_SEARCHES поисков."""
        _notify(progress, "answer")
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
            _notify(progress, "search")
            before = self._clock()
            found = self._retriever.search_all(
                args.all_queries, k=self._k, conditions=args.conditions
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


REASON_TEXT: dict[Reason, str] = {
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
    if answer.escalation is not None:
        summary = f": {answer.escalation.summary}" if answer.escalation.summary else ""
        return f"Передаю человеку (повод {answer.escalation.reason}){summary}"
    if answer.reason is not None:
        lines = [f"{REASON_TEXT[answer.reason]}."]
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
