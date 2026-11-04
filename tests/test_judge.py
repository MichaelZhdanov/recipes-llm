import json
from collections.abc import Mapping, Sequence
from typing import Any

from eval_helpers import CHUNK, RUN, SOURCES, case

from assistant.generation.answer import assemble
from assistant.generation.gigachat import Completion, LlmStats, Message, Usage
from assistant.generation.glossary import parse_glossary
from assistant.retrieval.store import Hit
from evals.answer_metrics import AnswerCase, AnswerRun, Rate
from evals.dataset import Canary, ClarityCanary
from evals.judge import (
    JUDGE_MODEL,
    Claim,
    Judge,
    Verdict,
    canary_scores,
    claims_of,
    clarity,
    clarity_canary_scores,
    groundedness,
    judge_clarity_canaries,
    judge_message,
    judge_results,
    must_include_judged,
)
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
    ) -> Completion:
        assert model == JUDGE_MODEL
        assert temperature > 0
        assert response_format is not None
        self.calls.append(list(messages))
        return Completion(
            self.replies.pop(0), "stop", f"{model}:1", Usage(10, 5, 0, 15)
        )


def reply(statements: list[bool], facts: list[bool] | None = None) -> str:
    return json.dumps(
        {
            "statements": [
                {"number": n, "comment": "…", "supported": ok, "clear": ok}
                for n, ok in enumerate(statements, start=1)
            ],
            "facts": [
                {"number": n, "present": ok}
                for n, ok in enumerate(facts or [], start=1)
            ],
        }
    )


def test_claims_are_kept_statements_with_quoted_lines() -> None:
    claims = claims_of(RUN)

    assert [c.text for c in claims] == [
        "Почта забирается из IMAP.",
        "Проверка раз в десять минут.",
    ]
    assert claims[0].fragments == [
        "docs/usage.md:12-12\nPaperless can fetch attachments from IMAP mail accounts."
    ]
    assert claims[1].fragments[0].startswith("docs/usage.md:13-13\nIt checks the mail")


def test_judge_message_numbers_statements_and_facts() -> None:
    message = judge_message("как настроить почту", claims_of(RUN), ["почтовый аккаунт"])

    assert '<statement number="2">' in message
    assert "<text>Проверка раз в десять минут.</text>" in message
    assert message.endswith("Факты, которые должны быть в ответе:\n1. почтовый аккаунт")


def test_verdicts_are_matched_by_number_and_missing_is_none() -> None:
    llm = FakeJudgeLlm(
        [
            json.dumps(
                {
                    "statements": [
                        {
                            "number": 2,
                            "comment": "нет",
                            "supported": False,
                            "clear": True,
                        }
                    ],
                    "facts": [],
                }
            )
        ]
    )

    verdict = Judge(llm).judge("?", claims_of(RUN), [])

    assert verdict.supported == [None, False]
    assert verdict.clear == [None, True]
    assert llm.calls[0][0]["content"].startswith("Ты проверяешь ответы ассистента")


def test_unparsed_judge_reply_is_counted_and_gives_no_verdict() -> None:
    judge = Judge(FakeJudgeLlm(["не JSON"]))

    verdict = judge.judge("?", [Claim("a", ["b"])], ["факт"])

    assert verdict == Verdict([None], [None], [], [None])
    assert judge.failures == 1


def test_refusal_means_facts_absent_without_calling_judge() -> None:
    llm = FakeJudgeLlm([reply([True, True], [True])])
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
    howto = case("howto-001", ["docs/usage.md#mail"]).model_copy(
        update={"must_include": ["почтовый аккаунт"]}
    )

    verdicts = judge_results([AnswerCase(howto, [RUN, refused])], Judge(llm))

    assert verdicts[("howto-001", 1)] == Verdict([], [False], [], [])
    assert len(llm.calls) == 1
    assert groundedness(verdicts) == Rate(2, 2)
    assert must_include_judged(verdicts) == Rate(1, 2)


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
    original = AnswerCase(case("howto-001", ["docs/usage.md#mail"]), [RUN])
    raw: list[dict[str, Any]] = json.loads(
        json.dumps(answer_records([original]), ensure_ascii=False)
    )

    [rebuilt] = load_answer_cases(raw, {"howto-001": original.case})

    assert rebuilt.runs[0].answer == RUN.answer
    assert rebuilt.runs[0].checked == RUN.checked
    assert rebuilt.runs[0].cited_source_ids() == {"docs/usage.md#mail"}


def test_clarity_is_judged_separately_from_support() -> None:
    llm = FakeJudgeLlm(
        [
            json.dumps(
                {
                    "statements": [
                        {
                            "number": 1,
                            "comment": "ок",
                            "supported": True,
                            "clear": True,
                        },
                        {
                            "number": 2,
                            "comment": "обрывок разметки",
                            "supported": True,
                            "clear": False,
                        },
                    ],
                    "facts": [],
                }
            )
        ]
    )

    verdict = Judge(llm).judge("?", claims_of(RUN), [])

    assert verdict.supported == [True, True]
    assert verdict.clear == [True, False]
    assert clarity({("howto-001", 0): verdict}) == Rate(1, 2)


def test_judge_sees_glossary_terms_from_fragments() -> None:
    glossary = parse_glossary(
        "volume\tтом\t-\tтом Docker, не «объём»\nimage\tобраз\t-\t\n"
    )
    llm = FakeJudgeLlm([reply([True])])
    claim = Claim(
        "Документы лежат в Docker-объёме.",
        ["docs/faq.md:19-19\nstored inside the docker volume"],
    )

    Judge(llm, glossary=glossary).judge("где файлы", [claim], [])

    message = llm.calls[0][1]["content"]
    assert (
        "Термины (как их нужно переводить):\n- volume — том (том Docker, не «объём»)"
        in message
    )
    # «image» во фрагменте нет — и в блоке его нет.
    assert "образ" not in message
    assert llm.calls[0][0]["content"].startswith("Ты проверяешь ответы")


def test_clarity_canaries_are_scored_by_clear_verdict() -> None:
    canaries = [
        ClarityCanary(
            id="clarity-001", statement="Docker-объём", quote="volume", clear=False
        ),
        ClarityCanary(
            id="clarity-002", statement="том Docker", quote="volume", clear=True
        ),
    ]
    llm = FakeJudgeLlm(
        [
            json.dumps(
                {
                    "statements": [
                        {"number": 1, "comment": "", "supported": True, "clear": False}
                    ],
                    "facts": [],
                }
            ),
            json.dumps(
                {
                    "statements": [
                        {"number": 1, "comment": "", "supported": True, "clear": False}
                    ],
                    "facts": [],
                }
            ),
        ]
    )

    marks = judge_clarity_canaries(canaries, Judge(llm))

    assert marks == [False, False]
    assert clarity_canary_scores(canaries, marks) == (Rate(1, 1), Rate(0, 1))
