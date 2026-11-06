"""Сравнение двух судей на одних и тех же ответах: `python -m evals.compare_judges A B`.

A и B — сырые данные двух пересчётов одного прогона (`make eval
ARGS="--rescore RAW --judge-provider …"`): ответы модели одни и те же,
разные только судьи. Скрипт печатает:
- канарейки: сколько неверных поймал и верных принял каждый судья, и на
  каких канарейках они разошлись;
- согласие по утверждениям ответов (опора на рецепт и грамотность): доля
  совпавших вердиктов и каппа Коэна — согласие с поправкой на случайность
  (0 — как случайно, 1 — полное). По 0.6 и выше согласие обычно считают
  существенным;
- утверждения, где судьи разошлись, с комментариями обоих.

Утверждение без вердикта у любого из судей в согласие не идёт.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def kappa(pairs: Sequence[tuple[bool, bool]]) -> float:
    """Каппа Коэна для двух судей с ответом да/нет."""
    if not pairs:
        return math.nan
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    yes_a = sum(a for a, _ in pairs) / n
    yes_b = sum(b for _, b in pairs) / n
    expected = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    return math.nan if expected == 1 else (observed - expected) / (1 - expected)


def verdicts(raw: dict[str, Any], field: str) -> dict[tuple[str, int, int], bool]:
    found = {}
    for record in raw["records"]:
        judge = record.get("judge") or {}
        for number, mark in enumerate(judge.get(field, [])):
            if mark is not None:
                found[(record["id"], record["run"], number)] = mark
    return found


def _statements(raw: dict[str, Any]) -> dict[tuple[str, int, int], tuple[str, str]]:
    """Текст утверждения и комментарий судьи по ключу (кейс, прогон, номер)."""
    texts = {}
    for record in raw["records"]:
        judge = record.get("judge") or {}
        kept = [s["text"] for s in record["checked"] if _kept(s)]
        comments = judge.get("comments", [])
        for number, text in enumerate(kept):
            comment = comments[number] if number < len(comments) else ""
            texts[(record["id"], record["run"], number)] = (text, comment)
    return texts


def _kept(statement: dict[str, Any]) -> bool:
    return not statement["unknown_numbers"] and any(
        c["error"] is None for c in statement["citations"]
    )


def _canary_lines(
    a: dict[str, Any], b: dict[str, Any], kind: str, title: str
) -> list[str]:
    left = {c["id"]: c for c in a.get("judging", {}).get(kind, [])}
    right = {c["id"]: c for c in b.get("judging", {}).get(kind, [])}
    if not left or not right:
        return [f"- {title}: нет данных у одного из судей"]

    def score(canaries: dict[str, dict[str, Any]]) -> str:
        bad = [c for c in canaries.values() if not c["expected"]]
        good = [c for c in canaries.values() if c["expected"]]
        caught = sum(c["mark"] is False for c in bad)
        accepted = sum(c["mark"] is True for c in good)
        return (
            f"поймал неверных {caught}/{len(bad)}, принял верных {accepted}/{len(good)}"
        )

    lines = [f"- {title}: A — {score(left)}; B — {score(right)}"]
    for canary_id in sorted(left.keys() & right.keys()):
        if left[canary_id]["mark"] != right[canary_id]["mark"]:
            lines.append(
                f"  - {canary_id} (ожидалось {left[canary_id]['expected']}): "
                f"A {left[canary_id]['mark']}, B {right[canary_id]['mark']}"
            )
    return lines


def compare(a: dict[str, Any], b: dict[str, Any], limit: int = 15) -> str:
    name_a = a.get("judging", {}).get("model", "A")
    name_b = b.get("judging", {}).get("model", "B")
    lines = [f"A — {name_a}, B — {name_b}", "", "Канарейки:"]
    lines += _canary_lines(a, b, "canaries", "опора на рецепт")
    lines += _canary_lines(a, b, "clarity_canaries", "грамотность")
    texts = _statements(a)
    comments_b = _statements(b)
    for field, title in (("supported", "опора на рецепт"), ("clear", "грамотность")):
        left, right = verdicts(a, field), verdicts(b, field)
        common = sorted(left.keys() & right.keys())
        pairs = [(left[key], right[key]) for key in common]
        agree = sum(x == y for x, y in pairs)
        lines += [
            "",
            f"Утверждения, {title}: совпало {agree} из {len(pairs)}, "
            f"каппа Коэна {kappa(pairs):.2f}; «да» у A {sum(x for x, _ in pairs)}, "
            f"у B {sum(y for _, y in pairs)}",
        ]
        differ = [key for key in common if left[key] != right[key]]
        for key in differ[:limit]:
            text, comment_a = texts.get(key, ("?", ""))
            comment_b = comments_b.get(key, ("", ""))[1]
            lines.append(
                f"- {key[0]} (прогон {key[1]}): «{text}» — A {left[key]}, B {right[key]}"
                + (f"; A: {comment_a}; B: {comment_b}" if field == "supported" else "")
            )
        if len(differ) > limit:
            lines.append(f"- … и ещё {len(differ) - limit}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.compare_judges")
    parser.add_argument("a", type=Path, help="сырые данные пересчёта с первым судьёй")
    parser.add_argument("b", type=Path, help="сырые данные пересчёта со вторым судьёй")
    args = parser.parse_args(argv)
    a, b = (json.loads(path.read_text(encoding="utf-8")) for path in (args.a, args.b))
    print(compare(a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
