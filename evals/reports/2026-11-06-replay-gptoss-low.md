# Отчёт eval — 2026-11-06

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
| LLM | openai/gpt-oss-120b; по ответам API: openai/gpt-oss-120b |
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
| citation accuracy | 0.99 (663/667, ДИ 0.98–1.00)  | ≥ 0.90 | ✅ |
| abstention (out_of_scope) | 1.00 (8/8, ДИ 0.68–1.00)  | ≥ 0.90 | ✅ |
| false refusal (вопросы из области) | 0.02 (1/63, ДИ 0.00–0.08)  | ≤ 0.10 | ✅ |
| болтовня: короткий ответ без утверждений (chit_chat) | 1.00 (5/5, ДИ 0.57–1.00) | показать | |
| покрытие утверждений цитатами | 0.99 (175/177, ДИ 0.96–1.00) | показать | |
| негативный контроль (цитата у соседнего источника) | 0.07 (48/667, ДИ 0.05–0.09) | ≈ 0 | |
| out_of_scope: отказ дал фильтр, а не модель | 0.00 (0/8, ДИ 0.00–0.32) | показать | |
| must_include (подстрокой) | 0.00 (0/3, ДИ 0.00–0.56) | показать | |
| must_not_include нарушен | 0.00 (0/5, ДИ 0.00–0.43) | 0 | |
| доля кириллицы в ответе вне `кода` | 1.00 | показать | |
| поиск p50 / p95 | 0.71 / 2.35 с | показать | |
| генерация p50 / p95 | 1.93 / 2.85 с | показать | |
| токены LLM: промпт / ответ / из кэша | 157176 / 32810 / 98432 | показать | |
| стоимость по платному тарифу | — (на Freemium бесплатно) | показать | |
| срабатывания маскера ПДн (эмбеддинги / LLM) | 0 / 0 | — | |

## Чем закончились прогоны

| Группа | ответ | model_not_found | filtered_out | format_error | blacklist | length |
|---|---|---|---|---|---|---|
| вопросы из области | 62 | 0 | 1 | 0 | 0 | 0 |
| out_of_scope | 1 | 7 | 0 | 0 | 0 | 0 |

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
| include-002 | include | ответ | 9 / 9 |
| include-003 | include | ответ | 12 / 12 |
| include-004 | include | ответ | 9 / 9 |
| include-005 | include | ответ | 8 / 8 |
| include-006 | include | ответ | 12 / 12 |
| include-007 | include | ответ | 6 / 6 |
| include-008 | include | ответ | 12 / 12 |
| include-009 | include | ответ | 13 / 13 |
| include-010 | include | ответ | 9 / 9 |
| include-011 | include | ответ | 9 / 9 |
| include-012 | include | ответ | 12 / 12 |
| exclude-001 | exclude | ответ | 15 / 15 |
| exclude-002 | exclude | ответ | 9 / 9 |
| exclude-003 | exclude | ответ | 12 / 12 |
| exclude-004 | exclude | ответ | 9 / 9 |
| exclude-005 | exclude | ответ | 9 / 9 |
| exclude-006 | exclude | ответ | 19 / 19 |
| exclude-007 | exclude | ответ | 9 / 9 |
| exclude-008 | exclude | ответ | 11 / 11 |
| exclude-009 | exclude | ответ | 18 / 18 |
| exclude-010 | exclude | ответ | 4 / 4 |
| by_name-001 | by_name | ответ | 12 / 12 |
| by_name-002 | by_name | ответ | 10 / 10 |
| by_name-003 | by_name | ответ | 5 / 5 |
| by_name-004 | by_name | ответ | 10 / 10 |
| by_name-005 | by_name | ответ | 10 / 10 |
| by_name-006 | by_name | ответ | 9 / 9 |
| by_name-007 | by_name | ответ | 21 / 22 |
| by_name-008 | by_name | ответ | 13 / 13 |
| by_name-009 | by_name | ответ | 12 / 12 |
| by_name-010 | by_name | ответ | 6 / 6 |
| semantic-001 | semantic | ответ | 20 / 20 |
| semantic-002 | semantic | ответ | 10 / 10 |
| semantic-003 | semantic | ответ | 9 / 9 |
| semantic-004 | semantic | ответ | 6 / 6 |
| semantic-005 | semantic | ответ | 10 / 10 |
| semantic-006 | semantic | ответ | 7 / 7 |
| semantic-007 | semantic | ответ | 18 / 18 |
| semantic-008 | semantic | ответ | 11 / 11 |
| semantic-009 | semantic | ответ | 12 / 12 |
| semantic-010 | semantic | ответ | 9 / 9 |
| how_to_cook-001 | how_to_cook | ответ | 21 / 21 |
| how_to_cook-002 | how_to_cook | ответ | 12 / 12 |
| how_to_cook-003 | how_to_cook | ответ | 8 / 8 |
| how_to_cook-004 | how_to_cook | ответ | 16 / 16 |
| how_to_cook-005 | how_to_cook | ответ | 17 / 17 |
| out_of_scope-001 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-002 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-003 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-004 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-005 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-006 | out_of_scope | ответ | 0 / 0 |
| out_of_scope-007 | out_of_scope | model_not_found | 0 / 0 |
| out_of_scope-008 | out_of_scope | model_not_found | 0 / 0 |
| chit_chat-001 | chit_chat | ответ | 0 / 0 |
| chit_chat-002 | chit_chat | ответ | 0 / 0 |
| chit_chat-003 | chit_chat | ответ | 0 / 0 |
| chit_chat-004 | chit_chat | ответ | 0 / 0 |
| chit_chat-005 | chit_chat | ответ | 0 / 0 |
| follow_up-001 | follow_up | ответ | 9 / 9 |
| follow_up-002 | follow_up | ответ | 9 / 9 |
| follow_up-003 | follow_up | filtered_out | 0 / 1 |
| follow_up-004 | follow_up | ответ | 1 / 1 |
| follow_up-005 | follow_up | ответ | 9 / 9 |
| follow_up-006 | follow_up | ответ | 12 / 12 |
| follow_up-007 | follow_up | ответ | 1 / 1 |
| follow_up-008 | follow_up | ответ | 9 / 9 |
| semantic-011 | semantic | ответ | 9 / 9 |
| semantic-012 | semantic | ответ | 6 / 6 |
| semantic-013 | semantic | ответ | 11 / 11 |
| semantic-014 | semantic | ответ | 6 / 6 |
| semantic-015 | semantic | ответ | 10 / 10 |
| semantic-016 | semantic | ответ | 10 / 12 |
| follow_up-009 | follow_up | ответ | 14 / 14 |
| follow_up-010 | follow_up | ответ | 15 / 15 |

## Отказы на вопросах из области (1)

- follow_up-003 (filtered_out) — «Сколько муки во втором?»
  - «Во втором рецепте указано 50 г муки пшеничной.» — такой строки нет в рецепте; чисел нет в цитатах: 50

## Ответы вне области (0)

Нет.

## Поломки: format_error, blacklist, length (0)

Нет.

## Неподтверждённые цитаты (4)

- такой строки нет в рецепте: 3
- это часть строки, а цитата — строка целиком: 1

Первые 10:

- by_name-007, источник 5: «Тирамису» — это часть строки, а цитата — строка целиком
- follow_up-003, источник 2: «Мука пшеничная — 50 г» — такой строки нет в рецепте
- semantic-016, источник 8: «Сахар — 1 ч. л.» — такой строки нет в рецепте
- semantic-016, источник 2: «Яйцо куриное — 3 шт» — такой строки нет в рецепте
