# Submission — Soup Kernel Challenge

Публичный Kaggle T4 development запуск прошёл **10/10 checks**.
Финальная платформенная сдача ещё не оформлена.

Kaggle full version: **363837033**, [ссылка](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363837033).
Snapshot SHA256: `ddc7740aa180e81cbdafa886d4fa44c1698cd2f79ae1cc45e63dbd45bedb033b`.
Measured solution SHA256: `698b33301c9a191f3b8bfadc502a0c15d7a08870579c465cfed119a06a2d4d77`.
Result SHA256: `eecdceb5343a87742d4bf2b1b87e497cd9cd072c52d39dfeb744e532ecf8bb8b`; он проверен против checks JSON.
Рабочий code commit: [b7c406dea7dbdc379149910efe044dccf155d083](https://github.com/promaladoi/Decentrathon-6.0/commit/b7c406dea7dbdc379149910efe044dccf155d083).
Все 32 solution-файла рабочего commit побайтно сверены с измеренным snapshot; measured solution SHA256 совпадает.


## Собственная реализация

| Собственный @triton.jit | Операция | Мин. доля времени |
| --- | --- | --- |
| vocabulary_partials_kernel | Частичные max, exp-sum и момент энтропии | 0.97% |
| vocabulary_finish_kernel | Объединение статистики CE/энтропии | 0.53% |
| vocabulary_gradient_kernel | Градиент CE с исходной нормировкой | 0.52% |
| soup_rmsnorm_forward_kernel | Frozen Qwen2 RMSNorm | 0.61% |
| soup_rmsnorm_backward_kernel | Градиент RMSNorm по активациям | 0.83% |
| soup_rope_forward_kernel | Парное вращение Q/K | 0.58% |
| soup_rope_backward_kernel | Градиент RoPE | 0.45% |
| soup_nf4_unpack_kernel | NF4 и double-quant scales для MatMul4Bit backward | 2.10% |

Суммарный минимум двух solution runs — **6.60%**, steps 3+.
Восемь собственных кернелов выполняются; organizer SwiGLU исключён.
CE/entropy, frozen RMSNorm, RoPE и fused NF4 описаны в README с формулами
и первичными источниками. Новые Python paths сохраняют FP16 storage
внутренних метрик, batched rewind losses, strided accuracy и frozen MLP
приведения без дополнительной квантизации. Head recompute и compressed-only
SwiGLU присутствуют как диагностика, но в training setup отключены.

## Проверенный запуск

| Показатель | База | Решение | Отношение / проверка |
| --- | --- | --- | --- |
| Полное время soup train, с | 88.74 | 78.18 | 1.1351× |
| GPU peak, MiB, nvidia-smi | 14345 | 7381 | база/решение 1.9435× |
| Public eval loss | 2.04964 | 2.04964 | JSON Δ +0.000% |
| Собственные кернелы, steps 3+ | — | 6.60% | минимум двух запусков, ≥5% |
| Автоматические правила | — | 10/10 PASS | публичные данные |

**241 passed, 1 skipped** solution; **79 passed** Soup с патчами. Полное время включает компиляцию; медиана двух времён и
максимум пиков. [result](verification/t4-result.json), [checks](verification/t4-checks.json), [Kaggle UI log](verification/t4-run.log), [source audit](verification/source-audit.json).

Историческая v9 №363823158 имела solution `wall_s=72.91`
и peak `11181 MiB`; текущие значения — `78.18`
и `7381 MiB`. Это **разные Kaggle-сессии**, не парный
v9/new A/B. Разница не приписывается одному изменению; текущий speedup
вычислен неизменённым harness относительно базы той же новой сессии.


## Артефакты и оставшаяся сдача

| Артефакт | Статус |
| --- | --- |
| Личная рабочая копия | [Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9), commit b7c406dea7dbdc379149910efe044dccf155d083 |
| Saved full development run | [№363837033](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363837033), private embedded snapshot |
| Clone working notebook | [Подготовлен](templates/kaggle_working_notebook.ipynb), ещё не исполнен |
| Платформенный final commit | Не подготовлен; [репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) не изменялся |
| Public notebook из шаблона с final commit | Не подготовлен |
| Данные команды, демо до 3 минут, презентация | Не подготовлены |

Работа ведётся в [личной папке Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
По указанию пользователя [платформенный репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) и папки
других участников не изменяются. Для сдачи ещё нужны финальный commit в
требуемом платформой репозитории, публичный сохранённый notebook из исходного
шаблона с этим commit и успешным Run All, данные команды, демо до трёх минут
и презентация. Скрытые строки и другие seed не проверены; финальный допуск
и конкурсный результат определяются отдельно. Текущая development версия private.

