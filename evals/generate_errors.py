"""Кейсы error из исходников Paperless: `python -m evals.generate_errors`.

Правильное место ошибки известно автоматически: это строка кода, где задан
её текст. Генератор берёт шаблоны сообщений тем же извлечением, что и индекс
(`assistant/indexing/messages.py`), заполняет подстановки правдоподобными
значениями и оборачивает текст так, как его присылают пользователи.

Слабое место. Генератор и поиск видят одни и те же шаблоны. Если извлечение
пропускает какой-то вид сообщений, генератор такой кейс просто не создаст.
Поэтому рядом есть кейсы независимого субагента (`error_reports.yaml`),
который читает код сам, а в паспорте индекса — покрытие извлечения.

Что выбирается:
- видимые пользователю сообщения: `_fail` обработчика, error и exception
  логгера, warning, raise; без debug и info;
- шаблоны, которые встречаются в коде один раз: иначе «правильное место»
  неоднозначно;
- поровну по видам, не больше MAX_PER_FILE из одного файла;
- dev и отложенная выборка не пересекаются.

Наборы пишутся в `evals/dataset/errors.yaml` и `evals/dataset/holdout/errors.yaml`
и вручную не правятся. При смене тега их нужно сгенерировать заново.
"""

import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from assistant.config import Settings
from assistant.indexing.code import chunk_code
from evals.dataset import DATASET_DIR, HOLDOUT_DIR

SEED = 20261104
DEV_SIZE = 30
HOLDOUT_SIZE = 15
MAX_PER_FILE = 3
# Виды сообщений, которые видит пользователь, по группам: из каждой поровну.
GROUPS = {
    "_fail": "обработка документа",
    "log.error": "ошибка в логе",
    "log.exception": "ошибка в логе",
    "log.critical": "ошибка в логе",
    "log.warning": "предупреждение в логе",
    "raise": "исключение",
}
LEVELS = {"log.warning": "WARNING", "log.critical": "CRITICAL"}
FORMS = ("сообщение", "строка лога", "ошибка задачи", "вопрос", "обрезано")


@dataclass(frozen=True)
class Template:
    path: str
    line: int
    kind: str
    parts: tuple[str, ...]
    head_hole: bool
    tail_hole: bool


def collect_templates(root: Path, ref: str, repo_url: str) -> list[Template]:
    """Шаблоны видимых сообщений, которые встречаются в коде ровно один раз."""
    found = [
        Template(
            chunk.path,
            message.line,
            message.kind,
            tuple(message.parts),
            message.head_hole,
            message.tail_hole,
        )
        for chunk in chunk_code(root, ref=ref, repo_url=repo_url)
        for message in chunk.messages
        if message.kind in GROUPS
    ]
    counts = Counter(t.parts for t in found)
    return sorted(
        (t for t in found if counts[t.parts] == 1), key=lambda t: (t.path, t.line)
    )


# Значения для подстановок: выбор по тексту перед дыркой, иначе — по кругу.
_HINTS = (
    ("asn", "42"),
    ("mime type", "application/octet-stream"),
    ("language", "rus"),
    ("version", "3.4.2"),
    ("page", "3"),
    ("id", "17"),
    ("user", "alice"),
)
_FALLBACK = (
    "scan_0042.pdf",
    "/usr/src/paperless/media/documents/originals/0000042.pdf",
    "[Errno 13] Permission denied",
    "42",
)


def _value(before: str, rng: random.Random) -> str:
    before = before.lower().rstrip(" :")
    return next(
        (v for hint, v in _HINTS if before.endswith(hint)), rng.choice(_FALLBACK)
    )


def fill(template: Template, rng: random.Random) -> str:
    """Склеивает шаблон, подставляя в дырки правдоподобные значения."""
    text = _value("", rng) if template.head_hole else ""
    for index, part in enumerate(template.parts):
        if index:
            text += _value(text, rng)
        text += part
    if template.tail_hole:
        text += _value(text, rng)
    return text.strip()


def logger_name(path: str) -> str:
    """src/documents/consumer.py → paperless.consumer: так называются логгеры Paperless."""
    return f"paperless.{Path(path).stem}"


def render(template: Template, form: str, rng: random.Random) -> str:
    message = fill(template, rng)
    if form == "строка лога":
        level = LEVELS.get(template.kind, "ERROR")
        return f"[2026-11-04 10:15:02,123] [{level}] [{logger_name(template.path)}] [3f2a9c1d] {message}"
    if form == "ошибка задачи":
        return f"scan_0042.pdf: {message}"
    if form == "вопрос":
        return f"В логах paperless вот такое: «{message}». Что это значит и что делать?"
    if form == "обрезано":
        # Пользователь скопировал не всё: оставляем первые две трети, но не меньше 30 символов.
        return message[: max(30, len(message) * 2 // 3)]
    return message


def pick(
    templates: Sequence[Template], size: int, rng: random.Random, taken: set[Template]
) -> list[Template]:
    """size шаблонов поровну из групп, не больше MAX_PER_FILE на файл, без уже взятых."""
    groups: dict[str, list[Template]] = {}
    for t in templates:
        if t not in taken:
            groups.setdefault(GROUPS[t.kind], []).append(t)
    queues = [rng.sample(g, len(g)) for _, g in sorted(groups.items())]
    per_file: Counter[str] = Counter()
    picked: list[Template] = []
    while len(picked) < size and any(queues):
        for queue in queues:
            while queue and len(picked) < size:
                candidate = queue.pop()
                if per_file[candidate.path] < MAX_PER_FILE:
                    per_file[candidate.path] += 1
                    picked.append(candidate)
                    break
    return picked


def cases(
    templates: Sequence[Template], prefix: str, rng: random.Random
) -> list[dict[str, object]]:
    return [
        {
            "id": f"{prefix}-{number:03}",
            "category": "error",
            # Исключения из глубины кода чаще разбирает инженер.
            "role": "engineer" if t.kind == "raise" else "user",
            "question": render(t, FORMS[number % len(FORMS)], rng),
            "expected_behavior": "answer",
            "expected_line": f"{t.path}:{t.line}",
            "notes": f"{GROUPS[t.kind]} ({t.kind}), форма: {FORMS[number % len(FORMS)]}",
        }
        for number, t in enumerate(templates, start=1)
    ]


def write(path: Path, items: list[dict[str, object]], ref: str) -> None:
    header = (
        f"# Сгенерировано evals/generate_errors.py из Paperless-ngx {ref}, сид {SEED}.\n"
        "# Вручную не править: при смене тега сгенерировать заново.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        header + yaml.safe_dump(items, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    settings = Settings()
    root = settings.paperless_source_path / settings.paperless_ref
    if not (root / "src").exists():
        print(f"нет исходников в {root}: выполните make sources")
        return 2
    templates = collect_templates(
        root, settings.paperless_ref, settings.paperless_repo_url
    )
    rng = random.Random(SEED)
    dev = pick(templates, DEV_SIZE, rng, set())
    holdout = pick(templates, HOLDOUT_SIZE, rng, set(dev))
    write(DATASET_DIR / "errors.yaml", cases(dev, "error", rng), settings.paperless_ref)
    write(
        HOLDOUT_DIR / "errors.yaml",
        cases(holdout, "error_holdout", rng),
        settings.paperless_ref,
    )
    print(
        f"шаблонов: {len(templates)}; dev {len(dev)}, отложенная выборка {len(holdout)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
