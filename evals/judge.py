"""Модель-судья ответов: groundedness, грамотность и факты must_include у semantic.

Код проверяет, что цитата — строка рецепта и что числа утверждения есть
в процитированных строках (`citations.py`). Но смысл он проверить не может:
не выдуман ли шаг или время, честно ли «постное» и «без лука». Это делает
судья — Claude Sonnet 5.5, двумя отдельными вызовами на каждый ответ:
1. Смысл, рубрика `evals/prompts/judge_v10.md`:
   - подтверждают ли рецепты каждое утверждение (groundedness, порог 0.85).
     Судья видит процитированный рецепт целиком: по полному списку
     ингредиентов видно, честно ли «без лука»;
   - есть ли в ответе факты must_include у semantic. Там факты записаны своими
     словами, подстрокой их не проверить.
2. Грамотность, рубрика `evals/prompts/clarity_v4.md`, по одному тексту
   утверждений, без рецептов. Модель выписывает конкретные проблемы текста:
   ошибка, английское слово, обрывок разметки, обрыв. Утверждение грамотно,
   если проблем нет: вердикт выводит код.

Почему двумя вызовами. Рубрики v2 и v3 спрашивали о грамотности в том же
вызове, что и о смысле. Канарейки показали, что так судья её не оценивает:
он поймал 1 плохой текст из 8 и пропустил «Docker-объём». Отдельный вызов
с одной задачей ловит 8 из 8.

Судье тоже нельзя верить на слово. Поэтому есть канарейки — записи с
известным ответом (`evals/dataset/judge/`), а дальше его оценки сверяются
с ручной разметкой автора.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from assistant.generation.citations import CheckedStatement, SourceText
from assistant.generation.gigachat import DETERMINISTIC_TEMPERATURE, ChatModel, LlmStats
from evals.answer_metrics import AnswerCase, AnswerRun, Rate
from evals.dataset import Canary, ClarityCanary

# Судья — Claude Sonnet 5.5 (решение автора 2026-11-06). До этого судил
# GigaChat-3-Ultra — модель другого семейства, чем отвечающий Claude. Слепая
# ручная разметка расхождений показала: «не подтверждено» только у Sonnet
# верно в 13 из 16, только у Ultra — в 1 из 14 (evals/reports/2026-11-06-judges-manual.md).
# Судья одного семейства с моделью ответа — известное ограничение: модели
# склонны узнавать и предпочитать свои тексты. Поэтому в промптах нет имён
# моделей, а оценки сверяются с ручной разметкой. Ultra — по `--judge-provider gigachat`.
JUDGE_MODEL = "claude-sonnet-5-5"
# Решение автора 2026-11-06: Sonnet на high «достаточно придирчив».
JUDGE_EFFORT = "high"
JUDGE_MODELS = {
    "gigachat": "GigaChat-3-Ultra",
    "anthropic": JUDGE_MODEL,
    # Судьи другого семейства из каталога cloud.ru (эксперимент 2026-11-06, evals/loop.py).
    "cloudru": "deepseek-ai/DeepSeek-V4-Flash",
}
# На Paperless в тот же вызов, что и смысл, пробовали добавить грамотность —
# судья её не оценивал (канарейки 1 из 8), поэтому грамотность — отдельный вызов.
# v1 — под Paperless (документация, английские фрагменты), v2 — под рецепты,
# v3 — после сравнения двух судей на одних ответах (evals/reports/2026-11-06-stage3.md):
# неполный список ингредиентов — не ошибка, утверждения о данных («шагов нет»,
# «рецепт по ссылке») верны; судья грамотности не видит рецептов.
# v4 — после ручной разметки расхождений (evals/reports/2026-11-06-judges-manual.md):
# что не ошибка (следует из названия, ярлык «классический», морепродукты — не мясо)
# и что ошибка (глагол способа приготовления, подача, «вместо X», оценка, вид блюда,
# изменённое название). Примеры в рубрике выдуманные, не из размеченных ответов.
# Как пополнять: ложное «нет» или пропуск судьи в ручной разметке → общее правило
# с выдуманным примером → новая версия рубрики → канарейки и пересчёт.
# v5 — после свежей слепой разметки v3 против v4 (evals/reports/2026-11-06-judge-v4-manual.md):
# «простой» — ярлык, не ошибка; ошибка — свойство блюда не из данных («потоньше»),
# составная часть блюда, которой нет в данных («соус из сметаны и чеснока»), «только X и Y»
# при других ингредиентах.
# v6 (решения автора 2026-11-06): стиль свободный, факты — только из рецептов:
# эпитеты, впечатления («получится сытно»), синонимы, вид блюда в двух словах и
# роль готового ингредиента — не ошибка; «вместо X» и состав части блюда — ошибка;
# вступление к ответу (kind="intro") проверяется по всем рецептам ответа.
# v7 (2026-11-06): основное правило — проверяемые факты о рецепте, всё остальное
# стиль. В v6 старое «true, только если всё сказанное следует из рецептов» спорило
# с разделом о стиле, и судья выбирал строгое: «сочный», «полегче», «песочное
# печенье» — «нет». Ещё: «без X» — сам продукт X, а не вся категория; утверждения
# о данных и о диалоге верны; заправка в любой форме — стадия приготовления (автор).
# v8 — после слепой разметки v6 против v7 (evals/reports/2026-11-06-judge-v7-manual.md):
# v7 пропускал части блюда («основа из…», «в тесте…», «X — основа теста»), причастия
# стадий («посыпанный») и роль ингредиента в процессе («подъём даёт сода»).
# v9 — после 48 канареек и решений автора: впечатление от вкуса ингредиента —
# стиль; вид блюда не обязан быть в названии; заправка в любой формулировке и
# «нужен, чтобы…» (роль при готовке) — ошибка. Примеры выдуманные.
# v10 — решение автора о canary-043: зачем нужен общеизвестный продукт
# («разрыхлитель — для пышности») — не ошибка; ошибка — что и как с ним делают в
# рецепте («соду гасят уксусом»). canary-043 устарела (evals/dataset.py, RETIRED_CANARIES).
JUDGE_PROMPT = "judge_v10"
# v4: обращение на «ты» и лишние эмодзи — проблема, живой тон и эмодзи в начале — нет.
CLARITY_PROMPT = "clarity_v4"
PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_TOKENS = 2000

JUDGE_SCHEMA: dict[str, object] = {
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
                "additionalProperties": False,
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

CLARITY_SCHEMA: dict[str, object] = {
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
                    "problems": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["number", "problems"],
            },
        },
    },
    "required": ["statements"],
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


class _ClarityItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    problems: list[str]


class _ClarityAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    statements: list[_ClarityItem]


@dataclass(frozen=True)
class Claim:
    """Утверждение и рецепты (или строки рецептов), на которые оно ссылается."""

    text: str
    fragments: list[str]
    # Вступление к ответу: общая фраза без своих цитат, судье — все рецепты ответа.
    intro: bool = False


@dataclass(frozen=True)
class Verdict:
    # По утверждениям и фактам; None — судья не дал ответа на этот пункт.
    supported: list[bool | None]
    present: list[bool | None]
    comments: list[str]
    # Грамотно ли написано каждое утверждение: проблем не найдено.
    clear: list[bool | None]
    # Проблемы текста, которые нашёл судья, по утверждениям.
    problems: list[list[str]] = field(default_factory=list)


def claims_of(run: AnswerRun) -> list[Claim]:
    """Утверждения, которые увидел пользователь, с процитированными рецептами
    целиком: рецепт короткий, а по полному списку ингредиентов судья видит,
    честно ли «без лука»."""
    # С проверкой перед отправкой судят итог — то, что увидел пользователь.
    statements = (
        run.review.final
        if run.review is not None
        else [s for s in run.checked if s.kept]
    )
    claims = [claim_of(s, run.sources) for s in statements]
    if run.answer.status == "answer" and run.answer.message:
        recipes = dict.fromkeys(f for c in claims for f in c.fragments)
        claims.append(Claim(run.answer.message, list(recipes), intro=True))
    return claims


def claim_of(statement: CheckedStatement, sources: Sequence[SourceText]) -> Claim:
    """Утверждение и рецепты, на которые ведут его подтверждённые цитаты, целиком."""
    by_number = {s.number: s for s in sources}
    cited = dict.fromkeys(c.source for c in statement.citations if c.valid)
    fragments = [f"Рецепт целиком:\n{by_number[n].text}" for n in cited]
    return Claim(statement.text, fragments)


def _statements(claims: Sequence[Claim]) -> list[str]:
    parts = []
    for number, claim in enumerate(claims, start=1):
        kind = ' kind="intro"' if claim.intro else ""
        parts += [
            "",
            f'<statement number="{number}"{kind}>',
            f"<text>{claim.text}</text>",
        ]
        parts += [
            f"<fragment>\n{fragment}\n</fragment>" for fragment in claim.fragments
        ]
        parts.append("</statement>")
    return parts


def judge_message(question: str, claims: Sequence[Claim], facts: Sequence[str]) -> str:
    parts = [
        f"Вопрос пользователя: {question}",
        "",
        "Утверждения ответа и их фрагменты:",
    ]
    parts += _statements(claims)
    if facts:
        parts += ["", "Факты, которые должны быть в ответе:"]
        parts += [f"{number}. {fact}" for number, fact in enumerate(facts, start=1)]
    else:
        parts += ["", "Фактов для проверки нет: верни пустой список facts."]
    return "\n".join(parts)


def clarity_message(claims: Sequence[Claim]) -> str:
    """Только тексты утверждений. С рецептами рядом судья (GigaChat-3-Ultra) оценивал
    грамотность строк рецепта, а не ответа: «Масло оливковое — неверное согласование»."""
    parts = ["Утверждения:"]
    for number, claim in enumerate(claims, start=1):
        parts += [
            "",
            f'<statement number="{number}">',
            f"<text>{claim.text}</text>",
            "</statement>",
        ]
    return "\n".join(parts)


class Judge:
    def __init__(
        self,
        llm: ChatModel,
        *,
        model: str = JUDGE_MODEL,
        prompt_version: str = JUDGE_PROMPT,
        clarity_version: str = CLARITY_PROMPT,
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_version = prompt_version
        self._prompt = (PROMPTS_DIR / f"{prompt_version}.md").read_text(
            encoding="utf-8"
        )
        self.clarity_version = clarity_version
        self._clarity_prompt = (PROMPTS_DIR / f"{clarity_version}.md").read_text(
            encoding="utf-8"
        )
        # Ответы, которые не разобрались: в отчёт, чтобы видеть, что судья ломается.
        self.failures = 0
        # Чем закончились ответы (stop, length…): у сбоя разбора видна причина.
        self.finish_reasons: Counter[str] = Counter()

    def judge(
        self,
        question: str,
        claims: Sequence[Claim],
        facts: Sequence[str],
        *,
        clarity: bool = True,
    ) -> Verdict:
        """Смысл и факты одним вызовом, грамотность — вторым (clarity=False — без него)."""
        if not claims and not facts:
            return Verdict([], [], [], [])
        supported, present, comments = self._support(question, claims, facts)
        clear, problems = (
            self.clarity(claims) if clarity else ([None] * len(claims), [])
        )
        return Verdict(supported, present, comments, clear, problems)

    def _ask(self, prompt: str, message: str, schema: dict[str, object]) -> str:
        completion = self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            model=self.model,
            temperature=DETERMINISTIC_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_schema", "schema": schema, "strict": True},
        )
        self.finish_reasons[completion.finish_reason] += 1
        return completion.content

    def _support(
        self, question: str, claims: Sequence[Claim], facts: Sequence[str]
    ) -> tuple[list[bool | None], list[bool | None], list[str]]:
        content = self._ask(
            self._prompt, judge_message(question, claims, facts), JUDGE_SCHEMA
        )
        try:
            parsed = _JudgeAnswer.model_validate_json(content)
        except ValidationError:
            self.failures += 1
            return [None] * len(claims), [None] * len(facts), []
        statements = {v.number: v for v in parsed.statements}
        present = {v.number: v.present for v in parsed.facts}
        numbers = range(1, len(claims) + 1)
        return (
            [v.supported if (v := statements.get(n)) else None for n in numbers],
            [present.get(n) for n in range(1, len(facts) + 1)],
            [v.comment if (v := statements.get(n)) else "" for n in numbers],
        )

    def clarity(
        self, claims: Sequence[Claim]
    ) -> tuple[list[bool | None], list[list[str]]]:
        """Грамотность: судья перечисляет проблемы, а «грамотно» — это «проблем нет»."""
        if not claims:
            return [], []
        content = self._ask(
            self._clarity_prompt, clarity_message(claims), CLARITY_SCHEMA
        )
        try:
            parsed = _ClarityAnswer.model_validate_json(content)
        except ValidationError:
            self.failures += 1
            return [None] * len(claims), []
        found = {item.number: item.problems for item in parsed.statements}
        numbers = range(1, len(claims) + 1)
        return (
            [None if n not in found else not found[n] for n in numbers],
            [found.get(n, []) for n in numbers],
        )


def facts_to_judge(result: AnswerCase) -> list[str]:
    """У semantic факты сформулированы своими словами — их проверяет судья.

    У follow_up это точные количества («50 г»), их проверяет код подстрокой.
    """
    return result.case.must_include if result.case.category == "semantic" else []


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
                verdict = Verdict([], [False] * len(facts), [], [])
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
    clarity_canaries: list[ClarityCanary] = field(default_factory=list)
    clarity_marks: list[bool | None] = field(default_factory=list)


def run_judge(
    results: Sequence[AnswerCase],
    judge: Judge,
    canaries: Sequence[Canary],
    clarity_canaries: Sequence[ClarityCanary] = (),
) -> Judging:
    verdicts = judge_results(results, judge)
    marks = judge_canaries(canaries, judge)
    clarity_marks = judge_clarity_canaries(clarity_canaries, judge)
    return Judging(
        model=judge.model,
        prompt_version=f"{judge.prompt_version}, грамотность {judge.clarity_version}",
        verdicts=verdicts,
        canaries=list(canaries),
        canary_marks=marks,
        stats=judge.llm.stats,
        failures=judge.failures,
        clarity_canaries=list(clarity_canaries),
        clarity_marks=clarity_marks,
    )


def judge_canaries(canaries: Sequence[Canary], judge: Judge) -> list[bool | None]:
    return [
        judge.judge("—", [Claim(c.statement, [c.quote])], [], clarity=False).supported[
            0
        ]
        for c in canaries
    ]


def judge_clarity_canaries(
    canaries: Sequence[ClarityCanary], judge: Judge
) -> list[bool | None]:
    return [judge.clarity([Claim(c.statement, [c.quote])])[0][0] for c in canaries]


# --- Метрики судьи ---


def groundedness(verdicts: Verdicts) -> Rate:
    marks = [m for v in verdicts.values() for m in v.supported if m is not None]
    return Rate(sum(marks), len(marks))


def clarity(verdicts: Verdicts) -> Rate:
    """Доля утверждений, написанных грамотно и понятно (по оценке судьи)."""
    marks = [m for v in verdicts.values() for m in v.clear if m is not None]
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


def clarity_canary_scores(
    canaries: Sequence[ClarityCanary], marks: Sequence[bool | None]
) -> tuple[Rate, Rate]:
    """(пойманы плохие тексты, приняты хорошие). Без ответа судьи — промах."""
    bad = [m for c, m in zip(canaries, marks, strict=True) if not c.clear]
    good = [m for c, m in zip(canaries, marks, strict=True) if c.clear]
    return (
        Rate(sum(m is False for m in bad), len(bad)),
        Rate(sum(m is True for m in good), len(good)),
    )
