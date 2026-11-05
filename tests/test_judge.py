import json
from collections.abc import Mapping, Sequence
from typing import Any

from eval_helpers import CHUNK, RUN, SOURCES, TEXT, case

from assistant.generation.answer import assemble
from assistant.generation.gigachat import (
    Completion,
    FunctionCallMode,
    LlmStats,
    Message,
    Usage,
)
from assistant.retrieval.store import Hit
from evals.answer_metrics import AnswerCase, AnswerRun, Rate
from evals.dataset import Canary, ClarityCanary
from evals.judge import (
    CLARITY_SCHEMA,
    JUDGE_MODEL,
    JUDGE_SCHEMA,
    Claim,
    Judge,
    Verdict,
    canary_scores,
    claims_of,
    clarity,
    clarity_canary_scores,
    groundedness,
    judge_canaries,
    judge_clarity_canaries,
    judge_message,
    judge_results,
    must_include_judged,
)
from evals.relevance import RELEVANCE_SCHEMA
from evals.run import answer_records, load_answer_cases


class FakeJudgeLlm:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[Message]] = []
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
        assert model == JUDGE_MODEL
        assert temperature > 0
        assert response_format is not None
        self.calls.append(list(messages))
        return Completion(
            self.replies.pop(0), "stop", f"{model}:1", Usage(10, 5, 0, 15)
        )


def reply(statements: list[bool], facts: list[bool] | None = None) -> str:
    """Ответ судьи на вызов «смысл»."""
    return json.dumps(
        {
            "statements": [
                {"number": n, "comment": "…", "supported": ok}
                for n, ok in enumerate(statements, start=1)
            ],
            "facts": [
                {"number": n, "present": ok}
                for n, ok in enumerate(facts or [], start=1)
            ],
        }
    )


def clarity_reply(*problems: list[str]) -> str:
    """Ответ судьи на вызов «грамотность»: проблемы по утверждениям."""
    return json.dumps(
        {
            "statements": [
                {"number": n, "problems": found}
                for n, found in enumerate(problems, start=1)
            ]
        }
    )


def test_claims_are_kept_statements_with_whole_cited_recipes() -> None:
    """Судья видит процитированный рецепт целиком: так проверяемо «без лука»."""
    claims = claims_of(RUN)

    assert [c.text for c in claims] == [
        "Почта забирается из IMAP.",
        "Проверка раз в десять минут.",
    ]
    assert claims[0].fragments == [f"Рецепт целиком:\n{TEXT}"]
    assert claims[1].fragments == [f"Рецепт целиком:\n{TEXT}"]


def test_judge_message_numbers_statements_and_facts() -> None:
    message = judge_message("как настроить почту", claims_of(RUN), ["почтовый аккаунт"])

    assert '<statement number="2">' in message
    assert "<text>Проверка раз в десять минут.</text>" in message
    assert message.endswith("Факты, которые должны быть в ответе:\n1. почтовый аккаунт")


def test_canary_scores_count_missing_verdict_as_miss() -> None:
    canaries = [
        Canary(id="canary-001", statement="a", quote="b", supported=False),
        Canary(id="canary-002", statement="a", quote="b", supported=False),
        Canary(id="canary-003", statement="a", quote="b", supported=True),
    ]

    caught, accepted = canary_scores(canaries, [False, None, True])

    assert caught == Rate(1, 2)
    assert accepted == Rate(1, 1)


def test_rescore_rebuilds_answers_from_raw_data() -> None:
    """Пересчёт собирает ответ тем же кодом, что и живой путь: результат совпадает."""
    original = AnswerCase(case("by_name-001"), [RUN])
    raw: list[dict[str, Any]] = json.loads(
        json.dumps(answer_records([original]), ensure_ascii=False)
    )

    [rebuilt] = load_answer_cases(raw, {"by_name-001": original.case})

    assert rebuilt.runs[0].answer == RUN.answer
    assert rebuilt.runs[0].checked == RUN.checked
    assert rebuilt.runs[0].cited_source_ids() == {"recipe/10"}


def test_verdicts_are_matched_by_number_and_missing_is_none() -> None:
    llm = FakeJudgeLlm(
        [
            json.dumps(
                {
                    "statements": [{"number": 2, "comment": "нет", "supported": False}],
                    "facts": [],
                }
            ),
            clarity_reply([], ["обрывок разметки"]),
        ]
    )

    verdict = Judge(llm).judge("?", claims_of(RUN), [])

    assert verdict.supported == [None, False]
    assert verdict.clear == [True, False]
    assert verdict.problems == [[], ["обрывок разметки"]]
    # Два вызова с разными рубриками: смысл и грамотность.
    assert llm.calls[0][0]["content"].startswith("Ты проверяешь ответы ассистента")
    assert llm.calls[1][0]["content"].startswith("Ты редактор русского текста")


def test_unparsed_judge_reply_is_counted_and_gives_no_verdict() -> None:
    judge = Judge(FakeJudgeLlm(["не JSON", "тоже не JSON"]))

    verdict = judge.judge("?", [Claim("a", ["b"])], ["факт"])

    assert verdict == Verdict([None], [None], [], [None])
    assert judge.failures == 2


def test_refusal_means_facts_absent_without_calling_judge() -> None:
    llm = FakeJudgeLlm([reply([True, True], [True]), clarity_reply([], [])])
    refused = AnswerRun.from_result(
        assemble(
            [Hit(CHUNK, 0.6)],
            SOURCES,
            Completion(
                '{"status": "not_found", "statements": []}',
                "stop",
                "m",
                Usage(1, 1, 0, 2),
            ),
            prompt_version="answer_v1",
            ref="v3.4.2",
        )
    )
    # Факты «своими словами» судья проверяет у semantic.
    semantic = case("semantic-001", "semantic", must_include=["почтовый аккаунт"])

    verdicts = judge_results([AnswerCase(semantic, [RUN, refused])], Judge(llm))

    assert verdicts[("semantic-001", 0)].present == [True]
    assert verdicts[("semantic-001", 1)] == Verdict([], [False], [], [])
    assert len(llm.calls) == 2  # смысл и грамотность только для ответа по существу
    assert groundedness(verdicts) == Rate(2, 2)
    assert clarity(verdicts) == Rate(2, 2)
    assert must_include_judged(verdicts) == Rate(1, 2)


def test_support_canaries_skip_the_clarity_call() -> None:
    llm = FakeJudgeLlm([reply([False])])
    canaries = [Canary(id="canary-001", statement="a", quote="b", supported=False)]

    assert judge_canaries(canaries, Judge(llm)) == [False]
    assert len(llm.calls) == 1


def test_clarity_canaries_are_scored_by_found_problems() -> None:
    canaries = [
        ClarityCanary(
            id="clarity-001", statement="Docker-объём", quote="volume", clear=False
        ),
        ClarityCanary(
            id="clarity-002", statement="том Docker", quote="volume", clear=True
        ),
    ]
    llm = FakeJudgeLlm(
        [clarity_reply(["объём — том"]), clarity_reply(["лишняя придирка"])]
    )

    marks = judge_clarity_canaries(canaries, Judge(llm))

    assert marks == [False, False]
    assert clarity_canary_scores(canaries, marks) == (Rate(1, 1), Rate(0, 1))


def test_judge_schemas_are_strict_for_claude() -> None:
    """Судьёй может быть Claude: у каждого объекта схемы — additionalProperties false."""

    def objects(schema: Any) -> list[dict[str, Any]]:
        found = [schema] if schema.get("type") == "object" else []
        for value in schema.get("properties", {}).values():
            found += objects(value)
        if "items" in schema:
            found += objects(schema["items"])
        return found

    for schema in (JUDGE_SCHEMA, CLARITY_SCHEMA, RELEVANCE_SCHEMA):
        assert all(o.get("additionalProperties") is False for o in objects(schema))
