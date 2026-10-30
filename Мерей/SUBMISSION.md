# Submission — vocabulary reductions, RMSNorm, RoPE и NF4

На публичных данных Kaggle T4 пройдены **10/10 автоматических проверок**.
Финальная сдача ещё не оформлена: работа ведётся в личной папке `Мерей`,
а Kaggle development notebook пока private и использует снимок исходников.
Репозиторий платформы по указанию пользователя не изменялся.

## Ссылки и артефакты

| Артефакт | Ссылка / статус |
|---|---|
| Рабочая папка | [promaladoi/Decentrathon-6.0 → Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9) |
| Коммит кода в личном репозитории | [3c389e9f1259031123026c99057f78b3ad57b671](https://github.com/promaladoi/Decentrathon-6.0/commit/3c389e9f1259031123026c99057f78b3ad57b671) |
| Проверенная сохранённая Kaggle-версия | [development №363806584](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363806584); private, embedded snapshot |
| Репозиторий платформы | [BAITC-Hacks/hack-a5a4072f-wither-ai](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai); оставлен без изменений |
| Финальный commit в репозитории платформы | Не подготовлен |
| Публичный notebook из шаблона с финальным commit | Не подготовлен |
| Название команды, состав, контакт капитана | Не предоставлены |
| Демо до 3 минут и презентация | Не подготовлены |

Основа: starter kit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
Проверенный source snapshot SHA256:
`eaa4e00672e0f2133bcbb1452ae73652c8e9dbaa3c1dc89b280725d0e9f80308`.
`harness/`, `tests/`, `task/` сохранены без изменений.

## Восемь собственных кернелов

| Имя `@triton.jit` | Файл | Операция | Мин. доля времени |
|---|---|---|---|
| `vocabulary_partials_kernel` | `solution/vocab.py` | Частичные max, exp-sum и момент энтропии | 1.51% |
| `vocabulary_finish_kernel` | `solution/vocab.py` | Объединение частей: CE или энтропия | 0.72% |
| `vocabulary_gradient_kernel` | `solution/vocab.py` | Градиент CE с `ignore_index` | 0.96% |
| `soup_rmsnorm_forward_kernel` | `solution/rmsnorm.py` | Qwen2 RMSNorm | 0.56% |
| `soup_rmsnorm_backward_kernel` | `solution/rmsnorm.py` | Градиент RMSNorm по активациям | 0.76% |
| `soup_rope_forward_kernel` | `solution/rope.py` | Вращение Q/K с RoPE | 0.52% |
| `soup_rope_backward_kernel` | `solution/rope.py` | Градиент RoPE без сохранения Q/K | 0.39% |
| `soup_nf4_unpack_kernel` | `solution/nf4.py` | NF4 и double-quant scales для MatMul4Bit backward | 1.92% |

Все восемь запускаются в обучении; суммарная доля — **7.3%**, минимум
двух запусков, шаги 3+. Доли строк посчитаны из CUDA-event `own_kernel_ms`
и `timed_loop_s`, округлены до 0.01%; минимум берётся для каждого кернела
отдельно и не складывается вместо суммарной проверки. Неизменённый пример
SwiGLU организаторов также подключён и исключён из собственной доли.
Формулы, интерфейсы и источники указаны в README; чужой Triton-код не
копировался.

## Измеренный результат, публичные строки

`run_task.py --repeats 2`, порядок база → решение → решение → база.

| Показатель | База | Решение | Отношение / разница |
|---|---|---|---|
| Полное время `soup train`, с | 86.68 | 81.82 | 1.0594× |
| Пик GPU, MiB, nvidia-smi | 14345 | 12381 | база/решение 1.1586× |
| Eval loss | 2.04963 | 2.04969 | +0.003% |
| Собственные кернелы, шаги 3+ | — | 7.3% | Требуется ≥5% |
| Автоматические правила | — | 10/10 PASS | Публичный development запуск |

Solution tests: **165 passed, 1 skipped**. Soup с патчами: **79 passed**.
Исходные артефакты: [result](verification/t4-result.json),
[checks](verification/t4-checks.json),
[отдельный microbenchmark](verification/t4-microbenchmark-final.json).
Скрытые строки жюри не проверялись. Подробности: `LOCAL_VERIFICATION.md`.

- [x] GPU unit/integration tests и 79 тестов Soup с патчами прошли на T4.
- [x] Выполнены два запуска базы и два решения; eval и 10/10 checks прошли.
- [x] README описывает алгоритмы, источники, воспроизведение и реальные результаты.
- [ ] Финальный код зафиксирован в требуемом платформой репозитории.
- [ ] Сохранён публичный notebook из шаблона с `COMMIT` финальной версии.
- [ ] Добавлены данные команды, демо и презентация.

Публикация и перенос в репозиторий платформы сейчас не выполнялись:
пользователь указал работать в личной папке `Мерей`. До финальной сдачи
необходимо отдельно завершить перечисленные артефакты и повторить запуск
с финальным commit.
