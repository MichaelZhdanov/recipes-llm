import json
from collections.abc import Mapping, Sequence
from typing import Any

from eval_helpers import case

from assistant.generation.answer import assemble
from assistant.generation.citations import SourceText
from assistant.generation.gigachat import Completion, LlmStats, Message, Usage
from assistant.indexing.chunks import Chunk
from assistant.retrieval.store import Hit
from evals.answer_metrics import AnswerCase, AnswerRun, Rate
from evals.dataset import Canary
from evals.judge import (
    JUDGE_MODEL,
    Claim,
    Judge,
    Verdict,
    canary_scores,
    claims_of,
    groundedness,
    judge_message,
    judge_results,
    must_include_judged,
)
from evals.run import answer_records, load_answer_cases

TEXT = """\
## Mail

Paperless can fetch attachments from IMAP mail accounts.
It checks the mail every ten minutes by default."""

CHUNK = Chunk(
    kind="doc",
    path="docs/usage.md",
    start_line=10,
    end_line=13,
    title="Mail",
    breadcrumb="Usage > Mail",
    source_id="docs/usage.md#mail",
    url="https://x/blob/v3.4.2/docs/usage.md?plain=1#L10-L13",
    site_url=None,
    ref="v3.4.2",
    text=TEXT,
)
SOURCES = [SourceText(1, CHUNK, TEXT)]


def answered(
    statements: list[tuple[str, str]], finish_reason: str = "stop"
) -> AnswerRun:
    content = json.dumps(
        {
            "status": "answer",
            "statements": [
                {"text": text, "citations": [{"source": 1, "quote": quote}]}
                for text, quote in statements
            ],
        }
    )
    completion = Completion(
        content, finish_reason, "GigaChat-2-Max:1", Usage(100, 20, 0, 120)
    )
    result = assemble(
        [Hit(CHUNK, 0.6)], SOURCES, completion, prompt_version="answer_v1", ref="v3.4.2"
    )
    return AnswerRun.from_result(result)


RUN = answered(
    [
        ("Почта забирается из IMAP.", "fetch attachments from IMAP mail accounts"),
        ("Выдумка.", "this quote is not in the source"),
        ("Проверка раз в десять минут.", "checks the mail every ten minutes"),
    ]
)


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
                {"number": n, "comment": "…", "supported": ok}
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
                    "statements": [{"number": 2, "comment": "нет", "supported": False}],
                    "facts": [],
                }
            )
        ]
    )

    verdict = Judge(llm).judge("?", claims_of(RUN), [])

    assert verdict.supported == [None, False]
    assert llm.calls[0][0]["content"].startswith("Ты проверяешь ответы ассистента")


def test_unparsed_judge_reply_is_counted_and_gives_no_verdict() -> None:
    judge = Judge(FakeJudgeLlm(["не JSON"]))

    verdict = judge.judge("?", [Claim("a", ["b"])], ["факт"])

    assert verdict == Verdict([None], [None], [])
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

    assert verdicts[("howto-001", 0)].present == [True]
    assert verdicts[("howto-001", 1)] == Verdict([], [False], [])
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
