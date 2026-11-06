# Отчёт eval — 2026-11-07

Ответы со ссылками. Категории: by_name, chit_chat, exclude, follow_up, how_to_cook, include, out_of_scope, semantic; наборы в `evals/dataset/`,
кто и как их составлял — `evals/dataset/README.md`.

| Параметр | Значение |
|---|---|
| Коммит ассистента | 4484a3a |
| Пересчёт | ответы модели из `повтор ответа по разбору из evals/reports/raw/2026-11-06-kimi-low.json`, ссылки и судья — заново |
| Данные | rogozinushka/povarenok-recipes, ревизия ffaf49ee |
| Модель эмбеддингов | cloud.ru: ai-sage/Giga-Embeddings-instruct-3B, размерность 2048 |
| Инструкция к вопросу | нет, вопрос уходит как есть |
| Индекс | 145954 рецептов (весь датасет), векторы int8; построен 2026-11-06T06:47:31+00:00 |
| LLM | deepseek-ai/DeepSeek-V4-Pro; по ответам API: deepseek-ai/DeepSeek-V4-Pro |
| Режим | always — «всегда ищем»: разбор вопроса, поиск, ответ |
| Температура | 0.0001 (GigaChat не принимает 0) |
| Промпт | `recipes_v3` — 76 |
| Судья | не запускался |
| Кейсов | 76 (by_name 10, chit_chat 5, exclude 10, follow_up 10, how_to_cook 5, include 12, out_of_scope 8, semantic 16), прогонов 1 |

## Метрики

Доли — «k из n» по всем прогонам, ДИ — 95% интервал Уилсона.

| Метрика | Значение | Порог | Статус |
|---|---|---|---|
| tool correctness: искал, где нужно, и не искал на болтовне | 0.91 (53/58, ДИ 0.81–0.96)  | ≥ 0.90 | ✅ |
| citation accuracy | 1.00 (866/866, ДИ 1.00–1.00)  | ≥ 0.90 | ✅ |
| abstention (out_of_scope) | 1.00 (8/8, ДИ 0.68–1.00)  | ≥ 0.90 | ✅ |
| false refusal (вопросы из области) | 0.02 (1/63, ДИ 0.00–0.08)  | ≤ 0.10 | ✅ |
| болтовня: короткий ответ без утверждений (chit_chat) | 1.00 (5/5, ДИ 0.57–1.00) | показать | |
| покрытие утверждений цитатами | 1.00 (182/182, ДИ 0.98–1.00) | показать | |
| негативный контроль (цитата у соседнего источника) | 0.06 (52/866, ДИ 0.05–0.08) | ≈ 0 | |
| out_of_scope: отказ дал фильтр, а не модель | 0.00 (0/8, ДИ 0.00–0.32) | показать | |
| must_include (подстрокой) | 0.67 (2/3, ДИ 0.21–0.94) | показать | |
| must_not_include нарушен | 0.00 (0/5, ДИ 0.00–0.43) | 0 | |
| доля кириллицы в ответе вне `кода` | 1.00 | показать | |
| поиск p50 / p95 | 0.70 / 1.69 с | показать | |
| генерация p50 / p95 | 29.05 / 53.38 с | показать | |
| токены LLM: промпт / ответ / из кэша | 168207 / 156620 / 99072 | показать | |
| стоимость по платному тарифу | — (на Freemium бесплатно) | показать | |
| срабатывания маскера ПДн (эмбеддинги / LLM) | 0 / 0 | — | |

## Чем закончились прогоны

| Группа | ответ | model_not_found | filtered_out | format_error | blacklist | length |
|---|---|---|---|---|---|---|
| вопросы из области | 62 | 1 | 0 | 0 | 0 | 0 |
| out_of_scope | 0 | 8 | 0 | 0 | 0 | 0 |

## Score топ-1 поиска

Порога по score нет. Распределение — чтобы решать о нём осознанно.

| Группа | min | медиана | max |
|---|---|---|---|
| вопросы из области | 0.331 | 0.640 | 0.831 |
| out_of_scope | 0.205 | 0.490 | 0.590 |

## По кейсам

| id | категория | итог по прогонам | цитаты: подтверждено / всего |
|---|---|---|---|
| include-001 | include | ответ | 12 / 12 |
| include-002 | include | ответ | 16 / 16 |
| include-003 | include | ответ | 13 / 13 |
| include-004 | include | ответ | 14 / 14 |
| include-005 | include | ответ | 8 / 8 |
| include-006 | include | ответ | 17 / 17 |
| include-007 | include | ответ | 12 / 12 |
| include-008 | include | ответ | 15 / 15 |
| include-009 | include | ответ | 21 / 21 |
| include-010 | include | ответ | 14 / 14 |
| include-011 | include | ответ | 19 / 19 |
| include-012 | include | ответ | 12 / 12 |
| exclude-001 | exclude | ответ | 11 / 11 |
| exclude-002 | exclude | ответ | 10 / 10 |
| exclude-003 | exclude | ответ | 17 / 17 |
| exclude-004 | exclude | ответ | 14 / 14 |
| exclude-005 | exclude | ответ | 19 / 19 |
| exclude-006 | exclude | ответ | 14 / 14 |
| exclude-007 | exclude | ответ | 16 / 16 |
| exclude-008 | exclude | ответ | 13 / 13 |
| exclude-009 | exclude | ответ | 10 / 10 |
| exclude-010 | exclude | ответ | 19 / 19 |
| by_name-001 | by_name | ответ | 15 / 15 |
| by_name-002 | by_name | ответ | 19 / 19 |
| by_name-003 | by_name | model_not_found | 0 / 0 |
| by_name-004 | by_name | ответ | 13 / 13 |
| by_name-005 | by_name | ответ | 18 / 18 |
| by_name-006 | by_name | ответ | 14 / 14 |
| by_name-007 | by_name | ответ | 9 / 9 |
| by_name-008 | by_name | ответ | 15 / 15 |
| by_name-009 | by_name | ответ | 11 / 11 |
| by_name-010 | by_name | ответ | 28 / 28 |
| semantic-001 | semantic | ответ | 15 / 15 |
| semantic-002 | semantic | ответ | 14 / 14 |
| semantic-003 | semantic | ответ | 16 / 16 |
| semantic-004 | semantic | ответ | 14 / 14 |
| semantic-005 | semantic | ответ | 15 / 15 |
| semantic-006 | semantic | ответ | 16 / 16 |
| semantic-007 | semantic | ответ | 12 / 12 |
| semantic-008 | semantic | ответ | 18 / 18 |
| semantic-009 | semantic | ответ | 14 / 14 |
| semantic-010 | semantic | ответ | 18 / 18 |
| how_to_cook-001 | how_to_cook | ответ | 30 / 30 |
| how_to_cook-002 | how_to_cook | ответ | 12 / 12 |
| how_to_cook-003 | how_to_cook | ответ | 25 / 25 |
| how_to_cook-004 | how_to_cook | ответ | 12 / 12 |
| how_to_cook-005 | how_to_cook | ответ | 16 / 16 |
| out_of_scope-001 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-002 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-003 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-004 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-005 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-006 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-007 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-008 | out_of_scope | model_not_found | 0 / 0 |
| chit_chat-001 | chit_chat | ответ | 0 / 0 |
| chit_chat-002 | chit_chat | ответ | 0 / 0 |
| chit_chat-003 | chit_chat | ответ | 0 / 0 |
| chit_chat-004 | chit_chat | ответ | 0 / 0 |
| chit_chat-005 | chit_chat | ответ | 0 / 0 |
| follow_up-001 | follow_up | ответ | 10 / 10 |
| follow_up-002 | follow_up | ответ | 12 / 12 |
| follow_up-003 | follow_up | ответ | 1 / 1 |
| follow_up-004 | follow_up | ответ | 1 / 1 |
| follow_up-005 | follow_up | ответ | 12 / 12 |
| follow_up-006 | follow_up | ответ | 12 / 12 |
| follow_up-007 | follow_up | ответ | 1 / 1 |
| follow_up-008 | follow_up | ответ | 16 / 16 |
| semantic-011 | semantic | ответ | 13 / 13 |
| semantic-012 | semantic | ответ | 13 / 13 |
| semantic-013 | semantic | ответ | 6 / 6 |
| semantic-014 | semantic | ответ | 13 / 13 |
| semantic-015 | semantic | ответ | 6 / 6 |
| semantic-016 | semantic | ответ | 20 / 20 |
| follow_up-009 | follow_up | ответ | 14 / 14 |
| follow_up-010 | follow_up | ответ | 11 / 11 |

## Отказы на вопросах из области (1)

- by_name-003 (model_not_found) — «аливье рецепт»

## Ответы вне области (0)

Нет.

## Поломки: format_error, blacklist, length (0)

Нет.

## Неподтверждённые цитаты (0)

Нет.
