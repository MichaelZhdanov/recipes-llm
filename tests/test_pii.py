import pytest

from assistant.security.pii import mask_pii


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("пишите на ivan.petrov@example.com", "EMAIL"),
        ("почта: test+tag@почта.рф", "EMAIL"),
        ("звоните +7 (999) 123-45-67", "PHONE"),
        ("тел. 8 999 123 45 67", "PHONE"),
        ("+79991234567", "PHONE"),
        ("офис в Берлине +49 30 1234567", "PHONE"),
        ("паспорт 45 06 123456", "PASSPORT"),
        ("паспорт 4506 123456", "PASSPORT"),
        ("серия 4506 номер 123456", "PASSPORT"),
        ("Серия 45 06 № 123456", "PASSPORT"),
        # Учебный пример СНИЛС с верной контрольной суммой.
        ("СНИЛС 112-233-445 95", "SNILS"),
        ("СНИЛС 11223344595", "SNILS"),
        # ИНН ПАО Сбербанк (публичный, контрольная сумма верная).
        ("ИНН 7707083893.", "INN"),
        # Часто цитируемый пример ИНН физлица; обе контрольные цифры проверены вручную.
        ("ИНН 500100732259", "INN"),
        # Тестовая карта Visa, проходит проверку Луна.
        ("карта 4111 1111 1111 1111", "CARD"),
        ("4111-1111-1111-1111", "CARD"),
    ],
)
def test_masks_personal_data(text: str, label: str) -> None:
    result = mask_pii(text)

    assert f"[{label}]" in result.text
    assert result.counts[label] == 1


@pytest.mark.parametrize(
    "text",
    [
        # Неверные контрольные суммы: похожие на ПДн числа не трогаем.
        "СНИЛС 112-233-445 96",
        "ИНН 7707083894",
        "ИНН 500100732258",
        "карта 4111 1111 1111 1112",
        # Типичные числа в коде и документации Paperless.
        "Paperless-ngx 3.4.2, порт 8000, 127.0.0.1:6333",
        "дата 2026-11-04 10:31:05",
        "uuid 123e4567-e89b-12d3-a456-426614174000",
        "commit d939d873e11dd14896d1bfe23f65b1baa6513ba7",
        "os.open(path, flags, 0o600); timeout=300",
        "@app.get('/health') и root@localhost",
        "PAPERLESS_OCR_LANGUAGE=rus+eng",
    ],
)
def test_leaves_non_personal_data_untouched(text: str) -> None:
    result = mask_pii(text)

    assert result.text == text
    assert not result.counts


def test_masks_several_kinds_in_one_text_and_counts_them() -> None:
    text = "Иван, ivan@example.com, +7 999 123-45-67, второй адрес ivan2@example.com"

    result = mask_pii(text)

    assert result.text == "Иван, [EMAIL], [PHONE], второй адрес [EMAIL]"
    assert result.counts == {"EMAIL": 2, "PHONE": 1}


def test_card_number_is_not_split_into_phone_and_inn() -> None:
    result = mask_pii("4111111111111111")

    assert result.text == "[CARD]"
    assert dict(result.counts) == {"CARD": 1}
