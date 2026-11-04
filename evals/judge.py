"""Модель-судья: groundedness и факты must_include у howto.

Код проверяет, что цитата есть в источнике (`citations.py`). Но он не может
проверить смысл: подтверждает ли английский фрагмент русское утверждение.
Это делает судья — другая, более сильная модель (GigaChat-3-Ultra) по
рубрике `evals/prompts/judge_v1.md`:
- на каждое утверждение итогового ответа: подтверждают ли его процитированные
  строки источника (groundedness, порог 0.85);
- на каждый факт must_include у howto: есть ли он в ответе. Факты там
  записаны по-русски своими словами, подстрокой их не проверить.

Судье тоже нельзя верить на слово. Поэтому есть канарейки: пары
«утверждение — фрагмент» с известным ответом, где 8 из 12 утверждений
тонко неверны. Если судья их пропускает, его groundedness ничего не стоит.
Дальше его оценки сверяются с ручной разметкой автора.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from assistant.generation.citations import SourceText
from assistant.generation.gigachat import DETERMINISTIC_TEMPERATURE, ChatModel, LlmStats
from evals.answer_metrics import AnswerCase, AnswerRun, Rate
from evals.dataset import Canary

JUDGE_MODEL = "GigaChat-3-Ultra"
JUDGE_PROMPT = "judge_v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_TOKENS = 2000

JUDGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "comment": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["number", "comment", "supported"],
            },
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "present": {"type": "boolean"},
                },
                "required": ["number", "present"],
            },
        },
    },
    "required": ["statements", "facts"],
}


class _StatementVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    comment: str
    supported: bool


class _FactVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    present: bool


class _JudgeAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    statements: list[_StatementVerdict]
    facts: list[_FactVerdict]


@dataclass(frozen=True)
class Claim:
    """Утверждение и строки источников, на которые оно ссылается."""

    text: str
    fragments: list[str]


@dataclass(frozen=True)
class Verdict:
    # По утверждениям и фактам; None — судья не дал ответа на этот пункт.
    supported: list[bool | None]
    present: list[bool | None]
    comments: list[str]


def quoted_lines(source: SourceText, start: int, end: int) -> str:
    """Строки источника, где найдена цитата: их судья и сравнивает с утверждением."""
    lines = source.text.split("\n")
    first = start - source.chunk.start_line
    return "\n".join(lines[first : first + end - start + 1])


def claims_of(run: AnswerRun) -> list[Claim]:
    """Утверждения, которые увидел пользователь, с процитированными строками."""
    by_number = {s.number: s for s in run.sources}
    claims = []
    for statement in run.checked:
        if not statement.kept:
            continue
        fragments = [
            f"{by_number[c.source].chunk.path}:{c.start_line}-{c.end_line}\n"
            + quoted_lines(by_number[c.source], c.start_line, c.end_line)
            for c in statement.citations
            if c.valid and c.start_line is not None and c.end_line is not None
        ]
        claims.append(Claim(statement.text, fragments))
    return claims


def judge_message(question: str, claims: Sequence[Claim], facts: Sequence[str]) -> str:
    parts = [
        f"Вопрос пользователя: {question}",
        "",
        "Утверждения ответа и их фрагменты:",
    ]
    for number, claim in enumerate(claims, start=1):
        parts += ["", f'<statement number="{number}">', f"<text>{claim.text}</text>"]
        parts += [
            f"<fragment>\n{fragment}\n</fragment>" for fragment in claim.fragments
        ]
        parts.append("</statement>")
    if facts:
        parts += ["", "Факты, которые должны быть в ответе:"]
        parts += [f"{number}. {fact}" for number, fact in enumerate(facts, start=1)]
    else:
        parts += ["", "Фактов для проверки нет: верни пустой список facts."]
    return "\n".join(parts)


class Judge:
    def __init__(
        self,
        llm: ChatModel,
        *,
        model: str = JUDGE_MODEL,
        prompt_version: str = JUDGE_PROMPT,
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = (PROMPTS_DIR / f"{prompt_version}.md").read_text(
            encoding="utf-8"
        )
        # Ответы, которые не разобрались: в отчёт, чтобы видеть, что судья ломается.
        self.failures = 0

    def judge(
        self, question: str, claims: Sequence[Claim], facts: Sequence[str]
    ) -> Verdict:
        if not claims and not facts:
            return Verdict([], [], [])
        completion = self.llm.chat(
            [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": judge_message(question, claims, facts)},
            ],
            model=self.model,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "schema": JUDGE_SCHEMA,
                "strict": True,
            },
        )
        try:
            parsed = _JudgeAnswer.model_validate_json(completion.content)
        except ValidationError:
            self.failures += 1
            return Verdict([None] * len(claims), [None] * len(facts), [])
        supported = {v.number: v.supported for v in parsed.statements}
        present = {v.number: v.present for v in parsed.facts}
        return Verdict(
            supported=[supported.get(n) for n in range(1, len(claims) + 1)],
            present=[present.get(n) for n in range(1, len(facts) + 1)],
            comments=[v.comment for v in parsed.statements],
        )


def facts_to_judge(result: AnswerCase) -> list[str]:
    """У howto факты сформулированы своими словами — их проверяет судья.

    У config это точные имена параметров, их проверяет код подстрокой.
    """
    return result.case.must_include if result.case.category == "howto" else []


Verdicts = dict[tuple[str, int], Verdict]


def judge_results(results: Sequence[AnswerCase], judge: Judge) -> Verdicts:
    verdicts: Verdicts = {}
    for r in results:
        facts = facts_to_judge(r)
        for number, run in enumerate(r.runs):
            claims = claims_of(run)
            if claims:
                verdict = judge.judge(r.case.question, claims, facts)
            else:
                # Ответа нет — значит, и фактов в нём нет; судья не нужен.
                verdict = Verdict([], [False] * len(facts), [])
            verdicts[(r.case.id, number)] = verdict
    return verdicts


@dataclass(frozen=True)
class Judging:
    """Всё о работе судьи для отчёта."""

    model: str
    prompt_version: str
    verdicts: Verdicts
    canaries: list[Canary]
    canary_marks: list[bool | None]
    stats: LlmStats
    failures: int


def run_judge(
    results: Sequence[AnswerCase], judge: Judge, canaries: Sequence[Canary]
) -> Judging:
    verdicts = judge_results(results, judge)
    marks = judge_canaries(canaries, judge)
    return Judging(
        model=judge.model,
        prompt_version=judge.prompt_version,
        verdicts=verdicts,
        canaries=list(canaries),
        canary_marks=marks,
        stats=judge.llm.stats,
        failures=judge.failures,
    )


def judge_canaries(canaries: Sequence[Canary], judge: Judge) -> list[bool | None]:
    return [
        judge.judge("—", [Claim(c.statement, [c.quote])], []).supported[0]
        for c in canaries
    ]


# --- Метрики судьи ---


def groundedness(verdicts: Verdicts) -> Rate:
    marks = [m for v in verdicts.values() for m in v.supported if m is not None]
    return Rate(sum(marks), len(marks))


def must_include_judged(verdicts: Verdicts) -> Rate:
    marks = [m for v in verdicts.values() for m in v.present if m is not None]
    return Rate(sum(marks), len(marks))


def unjudged(verdicts: Verdicts) -> int:
    return sum(m is None for v in verdicts.values() for m in [*v.supported, *v.present])


def canary_scores(
    canaries: Sequence[Canary], marks: Sequence[bool | None]
) -> tuple[Rate, Rate]:
    """(пойманы неверные, приняты верные). Без ответа судьи — промах."""
    wrong = [m for c, m in zip(canaries, marks, strict=True) if not c.supported]
    right = [m for c, m in zip(canaries, marks, strict=True) if c.supported]
    return (
        Rate(sum(m is False for m in wrong), len(wrong)),
        Rate(sum(m is True for m in right), len(right)),
    )
