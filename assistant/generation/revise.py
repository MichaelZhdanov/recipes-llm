"""Исправление утверждений ответа по замечаниям судей (эксперимент, 2026-11-06).

Судьи сказали «не подтверждено» — утверждение уходит на исправление.
Это отдельный вызов той же модели, что отвечала, с коротким промптом
`recipes_revise_v1`, а не продолжение диалога:
- в продолжении модель защищает уже данный ответ;
- в режиме auto в истории — вызовы поиска: модель могла бы искать заново
  и поменять рецепты, по которым её проверяют;
- исправлению нужны только вопрос, те же рецепты, утверждения и замечания.

Модель возвращает новый текст только для утверждений с замечаниями или
`drop` — «не исправить». Одобренные утверждения код не трогает. Новый
текст снова проходит проверку цитат кодом (`citations.py`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from assistant.generation.citations import (
    CheckedStatement,
    Citation,
    SourceText,
    Statement,
    check_statement,
)
from assistant.generation.gigachat import DETERMINISTIC_TEMPERATURE, ChatModel
from assistant.generation.prompt import load_prompt

PROMPT_VERSION = "recipes_revise_v1"
MAX_TOKENS = 2000

REVISE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "number": {"type": "integer"},
                    "drop": {"type": "boolean"},
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source": {"type": "integer"},
                                "quote": {"type": "string"},
                            },
                            "required": ["source", "quote"],
                        },
                    },
                },
                "required": ["number", "drop", "text", "citations"],
            },
        }
    },
    "required": ["statements"],
}


@dataclass(frozen=True)
class Remark:
    """Замечания судей к утверждению под номером `number` (с 1)."""

    number: int
    comments: list[str]


class _Revised(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    drop: bool
    text: str
    citations: list[Citation]


class _Revision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    statements: list[_Revised]


def _escape(text: str) -> str:
    """Утверждения и замечания — текст моделей: тегом из них не выйти."""
    return text.replace("<", "&lt;")


def revise_message(
    question: str,
    sources: Sequence[SourceText],
    statements: Mapping[int, CheckedStatement],
    remarks: Sequence[Remark],
) -> str:
    """statements — номер → утверждение: номера сохраняются, даже если часть
    утверждений уже удалена."""
    recipes = [f'<recipe id="{s.number}">\n{s.text}\n</recipe>' for s in sources]
    parts = ["Рецепты:", "\n\n".join(recipes), f"Вопрос пользователя: {question}"]
    parts.append("Утверждения ответа:")
    for number, statement in sorted(statements.items()):
        quotes = [
            f'<quote source="{c.source}">{_escape(c.quote)}</quote>'
            for c in statement.citations
            if c.valid
        ]
        parts.append(
            "\n".join(
                [
                    f'<statement number="{number}">',
                    f"<text>{_escape(statement.text)}</text>",
                    *quotes,
                    "</statement>",
                ]
            )
        )
    parts.append("Замечания проверки:")
    parts += [
        f'<remark number="{r.number}">\n'
        + "\n".join(_escape(c) for c in r.comments)
        + "\n</remark>"
        for r in remarks
    ]
    return "\n\n".join(parts)


class Reviser:
    def __init__(
        self, llm: ChatModel, *, model: str, prompt_version: str = PROMPT_VERSION
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = load_prompt(prompt_version)
        # Ответы, которые не разобрались: исправления нет, утверждение остаётся с замечанием.
        self.failures = 0

    def revise(
        self,
        question: str,
        sources: Sequence[SourceText],
        statements: Mapping[int, CheckedStatement],
        remarks: Sequence[Remark],
    ) -> dict[int, CheckedStatement | None]:
        """Номер утверждения → исправленное и проверенное кодом, или None —
        удалить. Утверждения без ответа модели в словарь не попадают."""
        if not remarks:
            return {}
        completion = self.llm.chat(
            [
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": revise_message(question, sources, statements, remarks),
                },
            ],
            model=self.model,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "schema": REVISE_SCHEMA,
                "strict": True,
            },
        )
        try:
            parsed = _Revision.model_validate_json(completion.content)
        except ValidationError:
            self.failures += 1
            return {}
        asked = {r.number for r in remarks}
        result: dict[int, CheckedStatement | None] = {}
        for item in parsed.statements:
            if item.number not in asked:
                continue  # одобренные утверждения модель менять не может
            if item.drop or not item.text.strip():
                result[item.number] = None
                continue
            checked = check_statement(
                Statement(text=item.text, citations=item.citations), sources
            )
            # Цитата не прошла проверку кодом — такое исправление не показываем.
            result[item.number] = checked if checked.kept else None
        return result
