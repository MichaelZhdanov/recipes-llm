"""Ответ на вопрос со ссылками: поиск → промпт → GigaChat → проверка ссылок.

1. Поиск даёт топ-K кусков (этап 1).
2. Каждый кусок проходит маскирование ПДн и экранирование и попадает в промпт
   блоком `<source id="N">`. Ровно этот текст потом сверяется с цитатами.
3. GigaChat отвечает JSON по схеме: утверждения с цитатами
   (`citations.py`, промпт `prompts/answer_v1.md`).
4. Код проверяет каждую цитату. Утверждение без подтверждённой цитаты
   удаляется, в источниках ответа остаются только процитированные рецепты.
5. Если не осталось ни одного утверждения, ответ — «не нашёл», и к нему
   прикладываются найденные рецепты: вдруг они всё же помогут.

Порога по score поиска нет: решение «есть ответ или нет» принимает модель,
а код проверяет, что ответ подтверждён. Подобрать порог на 32 кейсах значило
бы подогнать его под набор.
"""

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import httpx2
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient

from assistant.config import Settings
from assistant.generation.citations import (
    ANSWER_SCHEMA,
    CheckedStatement,
    ModelAnswer,
    SourceText,
    check_answer,
)
from assistant.generation.gigachat import (
    DETERMINISTIC_TEMPERATURE,
    ChatModel,
    Completion,
)
from assistant.generation.llm import llm_model, make_chat_model
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import Hit, VectorStore
from assistant.security.pii import mask_pii

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_VERSION = "answer_v1"
K = 5
# Сколько найденных рецептов приложить к ответу «не нашёл».
SUGGESTIONS = 3
# 6 утверждений с цитатами укладываются примерно в 1000 токенов, остальное — запас.
MAX_TOKENS = 2000

# Почему ответа нет. Первые два — честный отказ, остальные — поломки:
# - model_not_found — модель сама сказала, что в источниках ответа нет;
# - filtered_out — модель ответила, но ни одна цитата не подтвердилась;
# - format_error — ответ не по схеме;
# - blacklist — сработали тематические ограничения GigaChat;
# - length — ответ упёрся в MAX_TOKENS;
# - no_sources — поиск ничего не вернул (пустой индекс).
Reason = Literal[
    "model_not_found",
    "filtered_out",
    "format_error",
    "blacklist",
    "length",
    "no_sources",
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

    status: Literal["answer", "not_found"]
    reason: Reason | None
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


@dataclass(frozen=True)
class AnswerResult:
    answer: Answer
    trace: Trace


@cache
def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")


_SOURCE_TAG = re.compile(r"<(/?source)", re.IGNORECASE)


def escape_source_text(text: str) -> str:
    """Кусок не может закрыть свой блок: `</source>` в тексте становится `&lt;/source>`.

    Иначе документ мог бы «выйти» из блока данных и дописать промпт.
    Переводы строк не меняются, номера строк сохраняются.
    """
    return _SOURCE_TAG.sub(r"&lt;\1", text)


def prepare_sources(hits: Sequence[Hit]) -> list[SourceText]:
    """Тексты ровно в том виде, в каком они уйдут в модель.

    Маскирование здесь, а не только в клиенте GigaChat: цитаты сверяются
    с тем, что видела модель. Клиент маскирует ещё раз, это ничего не меняет.
    """
    return [
        SourceText(
            number=number,
            chunk=hit.chunk,
            text=escape_source_text(mask_pii(hit.chunk.text).text),
        )
        for number, hit in enumerate(hits, start=1)
    ]


def user_message(question: str, sources: Sequence[SourceText]) -> str:
    """Источники и вопрос."""
    # Название рецепта — первая строка текста, в атрибуты оно не идёт: это
    # текст с сайта, и только text проходит маскирование и экранирование.
    blocks = [
        f'<source id="{s.number}" recipe="{s.chunk.recipe_id}">\n{s.text}\n</source>'
        for s in sources
    ]
    return "\n\n".join(
        [
            "Источники:",
            "\n\n".join(blocks),
            f"Вопрос пользователя: {question}",
        ]
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


def check_completion(
    completion: Completion, sources: Sequence[SourceText]
) -> tuple[list[CheckedStatement], Reason | None]:
    """Разбирает ответ модели и проверяет ссылки. Без сети: так же работает пересчёт eval."""
    if completion.finish_reason != "stop":
        return [], _FINISH_REASONS.get(completion.finish_reason, "format_error")
    try:
        parsed = ModelAnswer.model_validate_json(completion.content)
    except ValidationError:
        return [], "format_error"
    if parsed.status == "not_found":
        return [], "model_not_found"
    checked = check_answer(parsed, sources)
    return checked, None if any(s.kept for s in checked) else "filtered_out"


def assemble(
    hits: list[Hit],
    sources: list[SourceText],
    completion: Completion | None,
    *,
    prompt_version: str,
    ref: str,
    retrieval_seconds: float = 0.0,
    generation_seconds: float = 0.0,
) -> AnswerResult:
    """Собирает ответ из найденного и ответа модели (None — модель не вызывалась)."""
    if completion is None:
        checked: list[CheckedStatement] = []
        reason: Reason | None = "no_sources"
    else:
        checked, reason = check_completion(completion, sources)
    if reason is None:
        statements, links = build_statements(checked, sources)
    else:
        statements, links = [], suggestions(hits)
    answer = Answer(
        status="answer" if reason is None else "not_found",
        reason=reason,
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
        prompt_version: str = PROMPT_VERSION,
        k: int = K,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self._retriever = retriever
        self._llm = llm
        self._ref = ref
        self._k = k
        self._clock = clock

    def answer(self, question: str) -> AnswerResult:
        started = self._clock()
        hits = self._retriever.search(question, k=self._k)
        retrieved = self._clock()
        sources = prepare_sources(hits)
        completion = None
        if sources:
            completion = self._llm.chat(
                [
                    {"role": "system", "content": load_prompt(self.prompt_version)},
                    {
                        "role": "user",
                        "content": user_message(question, sources),
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
        return assemble(
            hits,
            sources,
            completion,
            prompt_version=self.prompt_version,
            ref=self._ref,
            retrieval_seconds=retrieved - started,
            generation_seconds=self._clock() - retrieved if completion else 0.0,
        )


_REASON_TEXT: dict[Reason, str] = {
    "model_not_found": "Не нашёл ответа в найденных источниках",
    "filtered_out": "Не нашёл ответа, который подтверждают источники",
    "no_sources": "Не нашёл ничего по вопросу",
    "blacklist": "GigaChat не стал отвечать на этот вопрос (тематические ограничения)",
    "length": "Не получилось собрать ответ: он не уместился в лимит модели",
    "format_error": "Не получилось собрать ответ: модель ответила не по формату",
}


def _source_lines(link: SourceLink) -> list[str]:
    return [f"[{link.number}] {link.name}", f"    {link.url}"]


def render_markdown(answer: Answer) -> str:
    """Ответ текстом для консоли и чата: утверждения с [n], источники, пометка GigaChat."""
    if answer.reason is not None:
        lines = [f"{_REASON_TEXT[answer.reason]}."]
        if answer.sources:
            lines += ["", "Вот что может пригодиться:"]
    else:
        lines = [
            f"{s.text} " + "".join(f"[{n}]" for n in s.sources)
            for s in answer.statements
        ]
        lines += ["", "Источники:"]
    for link in answer.sources:
        lines += _source_lines(link)
    if answer.model:
        lines += [
            "",
            f"Ответ сгенерирован GigaChat ({answer.model}) по найденным источникам.",
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
    return Answerer(
        retriever, llm, model=llm_model(settings), ref=settings.dataset_revision
    )
