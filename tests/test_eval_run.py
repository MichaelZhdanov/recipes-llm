from eval_helpers import result

from evals.run import raw_records


def test_raw_data_has_one_record_per_case_and_run() -> None:
    records = raw_records([result("howto-001", [1, None]), result("howto-002", [3])])

    assert [(r["id"], r["run"], r["rank"]) for r in records] == [
        ("howto-001", 0, 1),
        ("howto-001", 1, None),
        ("howto-002", 0, 3),
    ]
    assert records[0]["top"] == [("docs/b.md#y", 0.5)]
