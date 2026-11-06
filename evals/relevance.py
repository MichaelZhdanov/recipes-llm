"""Судья релевантности: подходят ли найденные рецепты к вопросу «по смыслу».

У вопросов вроде «что-нибудь к чаю» или «постное на ужин» нет условий,
которые проверил бы код: ни слова в названии, ни ингредиентов. Поэтому
выдачу поиска оценивает модель-судья (тот же, что у ответов, `JUDGE_MODEL`; рубрика
`evals/prompts/relevance_v1.md`): про каждый рецепт из топ-5 она отвечает,
подходит ли он к вопросу. Метрика `semantic_relevance@5` — доля подходящих.

Судье тоже нельзя верить на слово: канарейки (`judge/relevance_canaries.yaml`)
— пары «вопрос — рецепт» с заранее известным ответом. Если судья их
путает, его оценкам верить нельзя, и это видно в отчёте.

Поиск детерминирован, поэтому судится только первый прогон кейса.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from assistant.generation.gigachat import DETERMINISTIC_TEMPERATURE, ChatModel
from evals.answer_metrics import Rate
from evals.dataset import RelevanceCanary
from evals.judge import JUDGE_MODEL, MAX_TOKENS, PROMPTS_DIR
from evals.retrieval_metrics import CaseResult, Found, K

RELEVANCE_PROMPT = "relevance_v1"

RELEVANCE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "number": {"type": "integer"},
                    "comment": {"type": "string"},
                    "relevant": {"type": "boolean"},
                },
                "required": ["number", "comment", "relevant"],
            },
        }
    },
    "required": ["recipes"],
}


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    number: int
    comment: str = ""
    relevant: bool


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recipes: list[_Verdict]


def _data(text: str) -> str:
    """Рецепт не может закрыть свой блок данных и дописать промпт."""
    return text.replace("<", "&lt;")


def relevance_message(question: str, recipes: Sequence[Found]) -> str:
    blocks = [
        f'<recipe number="{n}">\n{_data(f.name)}\n'
        f"Ингредиенты: {_data(', '.join(f.ingredients))}\n</recipe>"
        for n, f in enumerate(recipes, start=1)
    ]
    return "\n\n".join([f"Вопрос: {question}", "Найденные рецепты:", *blocks])


class RelevanceJudge:
    def __init__(
        self,
        llm: ChatModel,
        *,
        model: str = JUDGE_MODEL,
        prompt_version: str = RELEVANCE_PROMPT,
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = (PROMPTS_DIR / f"{prompt_version}.md").read_text(
            encoding="utf-8"
        )
        # Ответы, которые не разобрались: в отчёт, чтобы видеть, что судья ломается.
        self.failures = 0

    def judge(self, question: str, recipes: Sequence[Found]) -> list[bool | None]:
        """Вердикт по каждому рецепту; None — судья ответил не по схеме."""
        if not recipes:
            return []
        content = self.llm.chat(
            [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": relevance_message(question, recipes)},
            ],
            model=self.model,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "schema": RELEVANCE_SCHEMA,
                "strict": True,
            },
        ).content
        try:
            parsed = _Answer.model_validate_json(content)
        except ValidationError:
            self.failures += 1
            return [None] * len(recipes)
        by_number = {v.number: v.relevant for v in parsed.recipes}
        return [by_number.get(n) for n in range(1, len(recipes) + 1)]


@dataclass(frozen=True)
class Relevance:
    """Итог судьи по выдаче: вердикты по кейсам и проверка на канарейках."""

    model: str
    prompt_version: str
    # id кейса → вердикт по каждому рецепту топ-5 первого прогона.
    marks: dict[str, list[bool | None]]
    canaries: list[RelevanceCanary]
    canary_marks: list[bool | None]
    failures: int
    tokens: int


def judge_relevance(
    results: Sequence[CaseResult],
    judge: RelevanceJudge,
    canaries: Sequence[RelevanceCanary] = (),
) -> Relevance:
    marks = {
        r.case.id: judge.judge(r.case.question, r.runs[0].top[:K])
        for r in results
        if r.case.category == "semantic" and r.runs
    }
    canary_marks = [
        judge.judge(c.question, [Found(0, c.name, tuple(c.ingredients), 0.0)])[0]
        for c in canaries
    ]
    return Relevance(
        model=judge.model,
        prompt_version=judge.prompt_version,
        marks=marks,
        canaries=list(canaries),
        canary_marks=canary_marks,
        failures=judge.failures,
        tokens=judge.llm.stats.total_tokens,
    )


def semantic_relevance(relevance: Relevance) -> Rate:
    """Доля подходящих рецептов среди тех, по которым судья вынес вердикт."""
    marks = [m for kase in relevance.marks.values() for m in kase if m is not None]
    return Rate(sum(marks), len(marks))


def relevance_canary_scores(relevance: Relevance) -> tuple[Rate, Rate]:
    """(отвергнуты неподходящие, приняты подходящие). Без ответа судьи — промах."""
    pairs = list(zip(relevance.canaries, relevance.canary_marks, strict=True))
    wrong = [m for c, m in pairs if not c.relevant]
    right = [m for c, m in pairs if c.relevant]
    return (
        Rate(sum(m is False for m in wrong), len(wrong)),
        Rate(sum(m is True for m in right), len(right)),
    )
