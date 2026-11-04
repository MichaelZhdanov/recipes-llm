"""Маскирование персональных данных перед отправкой текста во внешние API.

Ловится то, у чего устойчивый формат: e-mail, телефоны, паспорт РФ, СНИЛС,
ИНН и номера карт. У СНИЛС, ИНН и карт есть контрольные суммы, и число
маскируется, только если сумма сходится. Иначе любое длинное число в коде
(таймстемп, идентификатор) превращалось бы в [INN].

ФИО и адреса регулярными выражениями надёжно не ловятся — это этап 7.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

# Число не должно быть частью более длинного слова или токена: «a1b2...» в хеше,
# «...-426614174000» в UUID. Точка после числа допустима (конец предложения).
_START = r"(?<![\w-])"
_END = r"(?![\w-])"


def _digits(text: str) -> list[int]:
    return [int(char) for char in text if char.isdigit()]


def _luhn_ok(text: str) -> bool:
    digits = _digits(text)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _snils_ok(text: str) -> bool:
    digits = _digits(text)
    if len(digits) != 11:
        return False
    total = sum(
        digit * weight
        for digit, weight in zip(digits[:9], range(9, 0, -1), strict=True)
    )
    check = total % 101 if total > 101 else total
    if check in (100, 101):
        check = 0
    return check == digits[9] * 10 + digits[10]


def _inn_check(digits: list[int], weights: tuple[int, ...]) -> int:
    return sum(d * w for d, w in zip(digits, weights, strict=False)) % 11 % 10


def _inn_ok(text: str) -> bool:
    digits = _digits(text)
    if len(digits) == 10:
        return _inn_check(digits, (2, 4, 10, 3, 5, 9, 4, 6, 8)) == digits[9]
    if len(digits) == 12:
        return (
            _inn_check(digits, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)) == digits[10]
            and _inn_check(digits, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)) == digits[11]
        )
    return False


def _phone_ok(text: str) -> bool:
    return 10 <= len(_digits(text)) <= 15


@dataclass(frozen=True)
class _Rule:
    label: str
    pattern: re.Pattern[str]
    is_valid: Callable[[str], bool] | None = None


# Порядок важен: сначала более длинные и строгие форматы, чтобы номер карты
# не разобрали на «телефон» и «ИНН».
_RULES = (
    _Rule("EMAIL", re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[^\W\d_]{2,}")),
    _Rule("CARD", re.compile(_START + r"\d(?:[ -]?\d){12,18}" + _END), _luhn_ok),
    _Rule(
        "SNILS",
        re.compile(_START + r"\d{3}[- ]?\d{3}[- ]?\d{3}[- ]?\d{2}" + _END),
        _snils_ok,
    ),
    _Rule(
        "PASSPORT",
        re.compile(
            r"(?i:серия)\s*\d{2}\s?\d{2}\s*(?:№|(?i:номер))?\s*\d{6}"
            + _END
            + r"|"
            + _START
            + r"\d{2}\s\d{2}\s\d{6}"
            + _END
            + r"|"
            + _START
            + r"\d{4}\s(?:№\s?)?\d{6}"
            + _END
        ),
    ),
    _Rule(
        "PHONE",
        re.compile(
            r"(?<![\w+])\+\d[\d\s()-]{8,18}\d"
            + _END
            + r"|"
            + _START
            + r"8[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}"
            + _END
        ),
        _phone_ok,
    ),
    _Rule("INN", re.compile(_START + r"(?:\d{12}|\d{10})" + _END), _inn_ok),
)


@dataclass(frozen=True)
class MaskResult:
    text: str
    counts: Counter[str]


def mask_pii(text: str) -> MaskResult:
    """Заменяет ПДн на метки вида [EMAIL] и считает, сколько чего заменено."""
    counts: Counter[str] = Counter()
    for rule in _RULES:

        def replace(match: re.Match[str], rule: _Rule = rule) -> str:
            if rule.is_valid is not None and not rule.is_valid(match.group()):
                return match.group()
            counts[rule.label] += 1
            return f"[{rule.label}]"

        text = rule.pattern.sub(replace, text)
    return MaskResult(text=text, counts=counts)
