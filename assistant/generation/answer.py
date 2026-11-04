"""Ответ на вопрос со ссылками: поиск → промпт → GigaChat → проверка ссылок.

1. Поиск даёт топ-K кусков (этап 1).
2. Каждый кусок проходит маскирование ПДн и экранирование и попадает в промпт
   блоком `<source id="N">`. Ровно этот текст потом сверяется с цитатами.
3. GigaChat отвечает JSON по схеме: утверждения с цитатами
   (`citations.py`, промпт `prompts/answer_v1.md`).
4. Код проверяет каждую цитату. Утверждение без подтверждённой цитаты
   удаляется, ссылки сужаются до строк цитаты.
5. Если не осталось ни одного утверждения, ответ — «не нашёл», и к нему
   прикладываются найденные куски: вдруг они всё же помогут.

Порога по score поиска нет: решение «есть ответ или нет» принимает модель,
а код проверяет, что ответ подтверждён. Подобрать порог на 32 кейсах значило
бы подогнать его под набор.
"""

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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
    llm_model,
    make_llm,
)
from assistant.generation.glossary import (
    Term,
    fix_terms,
    load_glossary,
    render_terms,
    terms_in,
)
from assistant.generation.history import mark
from assistant.indexing.chunks import Chunk
from assistant.retrieval.embeddings import make_embedder
from assistant.retrieval.search import Retriever
from assistant.retrieval.store import Hit, VectorStore
from assistant.router import QuestionType, route
from assistant.security.pii import mask_pii

PROMPTS_DIR = Path(__file__).parent / "prompts"
# answer_v1 — основной. v2 и v3 добавляли в промпт блок «Термины» из глоссария,
# но модель с ним хуже цитировала (0.93 → 0.87 и 0.84 на одних и тех же кейсах).
# Поэтому известные неверные переводы исправляет код после проверки ссылок
# (fix_terms), а блок терминов идёт только в промпты, которые его ждут.
PROMPT_VERSION = "answer_v1"
PROMPTS_WITH_TERMS = frozenset({"answer_v2", "answer_v3"})
# Разбор ошибки (этап 3): включается, когда точный поиск нашёл текст ошибки в коде.
ERROR_PROMPT_VERSION = "error_v1"
K = 5
# Сколько найденных кусков приложить к ответу «не нашёл».
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
    """Источник в ответе: ссылка на форк на строки цитаты и страница документации."""

    number: int
    kind: str
    path: str
    start_line: int
    end_line: int
    title: str
    url: str
    site_url: str | None


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
    paperless_ref: str
    # error — в вопросе текст ошибки из кода (router.py), general — обычный вопрос.
    question_type: QuestionType = "general"


@dataclass(frozen=True)
class Trace:
    """Всё, что нужно eval: что видела модель, что ответила и что прошло проверку."""

    hits: list[Hit]
    sources: list[SourceText]
    completion: Completion | None
    checked: list[CheckedStatement]
    retrieval_seconds: float
    generation_seconds: float
    # Исправленные кодом неверные переводы терминов: («объёма», «тома»).
    term_fixes: list[tuple[str, str]] = field(default_factory=list)


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


def _attribute(value: str) -> str:
    return value.replace('"', "'").replace("\n", " ")


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


def user_message(
    question: str,
    sources: Sequence[SourceText],
    ref: str,
    glossary: Sequence[Term] = (),
    error_lines: Sequence[str] = (),
) -> str:
    """Источники, термины из них (если есть в глоссарии), подсказка о месте ошибки и вопрос."""
    blocks = [
        f'<source id="{s.number}" path="{s.chunk.path}" '
        f'lines="{s.chunk.start_line}-{s.chunk.end_line}" '
        f'title="{_attribute(s.chunk.breadcrumb)}">\n{s.text}\n</source>'
        for s in sources
    ]
    # Термины — до источников: модель читает их раньше, чем начинает переводить.
    parts = []
    terms = terms_in((s.text for s in sources), glossary)
    if terms:
        parts.append("Термины (переводи только так):\n" + render_terms(terms))
    parts += [f"Источники (Paperless-ngx {ref}):", "\n\n".join(blocks)]
    if error_lines:
        # Подсказка, где совпал текст ошибки. Цитату из этого куска модель всё равно
        # должна привести сама: проверка ссылок та же.
        parts.append(
            "Текст ошибки из вопроса совпал со строкой кода: "
            + ", ".join(error_lines)
            + "."
        )
    parts.append(f"Вопрос пользователя: {question}")
    return "\n\n".join(parts)


def _link(number: int, chunk: Chunk, start: int, end: int) -> SourceLink:
    # Ссылка куска ведёт на все его строки; сужаем до строк цитаты.
    base = chunk.url.rsplit("#L", 1)[0]
    return SourceLink(
        number=number,
        kind=chunk.kind,
        path=chunk.path,
        start_line=start,
        end_line=end,
        title=chunk.breadcrumb,
        url=f"{base}#L{start}-L{end}",
        site_url=chunk.site_url,
    )


def build_statements(
    checked: Sequence[CheckedStatement],
    sources: Sequence[SourceText],
    glossary: Sequence[Term] = (),
) -> tuple[list[AnswerStatement], list[SourceLink], list[tuple[str, str]]]:
    """Оставляет подтверждённые утверждения и собирает список источников.

    Один процитированный кусок — один источник в ответе. Его строки — от первой
    до последней подтверждённой цитаты из него. Номера идут по первому упоминанию.

    Известные неверные переводы терминов исправляются уже после проверки
    ссылок, и только для терминов из процитированных кусков: «объём»
    в ответе про Docker volume — ошибка, в ответе про размер диска — нет.
    Цитаты и имена в обратных кавычках не меняются.

    Утверждение, у которого все цитаты из changelog, помечается версией
    записи (`history.py`): это история, а не текущее состояние.
    """
    kept = [s for s in checked if s.kept]
    spans: dict[int, tuple[int, int]] = {}
    for statement in kept:
        for c in statement.citations:
            if c.start_line is None or c.end_line is None:
                continue  # цитата не подтвердилась
            first, last = spans.get(c.source, (c.start_line, c.end_line))
            spans[c.source] = (min(first, c.start_line), max(last, c.end_line))

    chunks = {s.number: s.chunk for s in sources}
    order = {source: number for number, source in enumerate(spans, start=1)}
    links = [
        _link(order[source], chunks[source], first, last)
        for source, (first, last) in spans.items()
    ]
    texts = {s.number: s.text for s in sources}
    statements = []
    fixes: list[tuple[str, str]] = []
    for statement in kept:
        cited = list(dict.fromkeys(c.source for c in statement.citations if c.valid))
        terms = terms_in((texts[number] for number in cited), glossary)
        text, fixed = fix_terms(statement.text, terms)
        fixes += fixed
        text = mark(text, [chunks[number] for number in cited])
        statements.append(
            AnswerStatement(text=text, sources=[order[number] for number in cited])
        )
    return statements, links, fixes


def suggestions(hits: Sequence[Hit], limit: int = SUGGESTIONS) -> list[SourceLink]:
    return [
        _link(n, h.chunk, h.chunk.start_line, h.chunk.end_line)
        for n, h in enumerate(hits[:limit], start=1)
    ]


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
    glossary: Sequence[Term] = (),
    question_type: QuestionType = "general",
) -> AnswerResult:
    """Собирает ответ из найденного и ответа модели (None — модель не вызывалась)."""
    if completion is None:
        checked: list[CheckedStatement] = []
        reason: Reason | None = "no_sources"
    else:
        checked, reason = check_completion(completion, sources)
    fixes: list[tuple[str, str]] = []
    if reason is None:
        statements, links, fixes = build_statements(checked, sources, glossary)
    else:
        statements, links = [], suggestions(hits)
    answer = Answer(
        status="answer" if reason is None else "not_found",
        reason=reason,
        statements=statements,
        sources=links,
        model=completion.model if completion else "",
        prompt_version=prompt_version,
        paperless_ref=ref,
        question_type=question_type,
    )
    trace = Trace(
        hits=hits,
        sources=sources,
        completion=completion,
        checked=checked,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
        term_fixes=fixes,
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
        glossary: Sequence[Term] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self._retriever = retriever
        self._llm = llm
        self._ref = ref
        self._k = k
        self._glossary = load_glossary() if glossary is None else glossary
        self._clock = clock

    def answer(self, question: str) -> AnswerResult:
        started = self._clock()
        hits = self._retriever.search(question, k=self._k)
        retrieved = self._clock()
        sources = prepare_sources(hits)
        question_type = route(hits)
        prompt_version = (
            ERROR_PROMPT_VERSION if question_type == "error" else self.prompt_version
        )
        error_lines = [
            f"`{hit.chunk.path}:{hit.line}` (источник {number})"
            for number, hit in enumerate(hits, start=1)
            if hit.exact
        ]
        completion = None
        if sources:
            completion = self._llm.chat(
                [
                    {"role": "system", "content": load_prompt(prompt_version)},
                    {
                        "role": "user",
                        "content": user_message(
                            question,
                            sources,
                            self._ref,
                            self._glossary
                            if prompt_version in PROMPTS_WITH_TERMS
                            else (),
                            error_lines,
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
        return assemble(
            hits,
            sources,
            completion,
            prompt_version=prompt_version,
            ref=self._ref,
            retrieval_seconds=retrieved - started,
            generation_seconds=self._clock() - retrieved if completion else 0.0,
            glossary=self._glossary,
            question_type=question_type,
        )


_REASON_TEXT: dict[Reason, str] = {
    "model_not_found": "Не нашёл ответа в документации и коде",
    "filtered_out": "Не нашёл ответа, который подтверждают документация и код",
    "no_sources": "Не нашёл ничего по вопросу в документации и коде",
    "blacklist": "GigaChat не стал отвечать на этот вопрос (тематические ограничения)",
    "length": "Не получилось собрать ответ: он не уместился в лимит модели",
    "format_error": "Не получилось собрать ответ: модель ответила не по формату",
}


def _source_lines(link: SourceLink) -> list[str]:
    span = (
        f"{link.start_line}"
        if link.start_line == link.end_line
        else f"{link.start_line}-{link.end_line}"
    )
    lines = [
        f"[{link.number}] {link.path}:{span} — {link.title}",
        f"    {link.url}",
    ]
    if link.site_url:
        lines.append(f"    сайт: {link.site_url}")
    return lines


def render_markdown(answer: Answer) -> str:
    """Ответ текстом для консоли и чата: утверждения с [n], источники, пометка GigaChat."""
    if answer.reason is not None:
        lines = [f"{_REASON_TEXT[answer.reason]} Paperless-ngx {answer.paperless_ref}."]
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
            f"Ответ сгенерирован GigaChat ({answer.model}) по документации и коду "
            f"Paperless-ngx {answer.paperless_ref}.",
        ]
    return "\n".join(lines)


def make_answerer(
    settings: Settings,
    *,
    embeddings_http: httpx2.Client,
    llm_http: httpx2.Client,
    qdrant: QdrantClient,
) -> Answerer:
    """Собирает ассистента из настроек. У GigaChat свой HTTP-клиент: свой TLS-корень."""
    # Сначала то, что падает без ключей: ошибка понятна ещё до сетевых вызовов.
    embedder = make_embedder(settings, embeddings_http)
    llm = make_llm(settings, llm_http)
    retriever = Retriever(VectorStore(qdrant), embedder, ref=settings.paperless_ref)
    return Answerer(
        retriever, llm, model=llm_model(settings), ref=settings.paperless_ref
    )
