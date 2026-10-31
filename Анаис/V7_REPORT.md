# Анаис V7 — проверенная интеграция командного v45

Полная команда: **44,70 с** по медиане двух холодных парных запусков на Kaggle T4. Цель **20–30 с не достигнута**. Число 25,94 с относится только к циклу обучения.

| Метрика | Base | Анаис V7 |
|---|---:|---:|
| Полное время, медиана, с | 83,43 | 44,70 |
| Отдельные запуски, с | 83,58 / 83,28 | 43,82 / 45,57 |
| Обучение, медиана, с | 61,88 | 25,94 |
| Пиковая память, MiB | 14506 | 7040 |
| Eval loss | 2,04963 | 2,04963 |

Оригинальные правила: **10/10 PASS**. Решение: **388 passed, 1 opt-in skipped**; Soup: **79 passed**. Доля собственных ядер — не менее 6,7% времени обучения. Отдельная opt-in проверка native metadata сохраняется в полном архиве notebook. Не складывать пересекающиеся suites.

Модель Qwen2.5-1.5B-Instruct, NF4, LoRA, данные, batch/order, 64 шага, seed, optimizer, schedule и выходы Soup сохранены. Eval loss в этой паре не вырос; скрытые данные остаются отдельной проверкой. В интерактивном процессе оставался неработающий CUDA-контекст предыдущей диагностики; оригинальная проверка idle/clean measurement прошла, внешние пики приведены без вычитания этого контекста.

## Изменения и авторство

За основу скопирован выбранный `Нұржігіт/solution` (v45, измеренный source commit `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`, snapshot репозитория `735187a88975867781d40104bb2bc91d9b582fce`). Эта папка использована только для чтения. Все записи в репозитории ограничены `Анаис/`.

Нұржігіт: точное компактное frozen tied storage, управление временем жизни временных tensors и подготовка launchers. Мерей: корректный private training path, фактический AMP gradient scale, packed dropout, batch metadata и API guards. Владимир: исходные командные ядра и интеграция unpadding/MLP/NF4. SwiGLU организаторов сохранён с исходной атрибуцией.

Анаис: run-local NF4 decode cache с инвалидацией по источникам и версиям, перенесённый в этот командный вариант; совместимость прежних cache/launcher diagnostic helpers; независимые GPU-тесты, измерения и отчёт. Нет дискового кэша между запусками, параметры NF4 не заменяются.

## Доказательства

- `solution/` побайтно совпадает с измеренным решением: digest указан в `V7_VERIFICATION.json` и оригинальном `v7-checks.json`.
- `v7-result.json`, `checks.json`, per-run metrics и logs — оригинальные bytes из Kaggle.
- `kaggle_anais_v7.ipynb` — автономный повторяемый notebook с двумя парными запусками.
- `v7-source.zip`, `V7_PROVENANCE.json`, `V7_VERIFICATION.json` — исходники и цепочка SHA256.

## Другие проверенные варианты

V8: V7 плюс causal target partitions из эксперимента Нұржігіта v51. 418 passed / 1 skipped; одиночный screening 44,37 с, 7038 MiB, eval loss 2,04964. V9: V8 плюс source-guarded lazy train CLI из v34. 425 passed / 1 skipped; screening 44,02 с, 7038 MiB, eval loss 2,04967. Эти solution-only screenings не являются полным парным допуском. Устойчивое преимущество не установлено; приоритет качества сохраняет V7.

Изменение layout frozen weights: weighted MLP speedup 0,9789x, часть округлений отличается — отклонено. Statement-level startup diagnostic: import torch 1,52 с, CUDA init 0,28 с, import Transformers 3,01 с, apply 5,00 с, Soup preparation/training 33,30 с. Диагностические времена не выдаются за соревновательные.

До верхней границы цели остаётся 14,70 с. Не заявляется достижение 20–30 с или невозможность дальнейшего ускорения. Старые `SPEED_REPORT.md`, `speed-result.json`, `fastest-source.zip`, `kaggle_fastest.ipynb` относятся к V6 и сохранены как история.
