# Submission — half-input CE, RMSNorm, RoPE и NF4

На публичных данных Kaggle T4 пройдены **10/10 автоматических проверок**.
Финальная сдача не оформлена: работа ведётся в личной папке `Мерей`,
проверенный Kaggle notebook пока private и использует embedded snapshot.
Репозиторий платформы по указанию пользователя оставлен без изменений.

## Ссылки и артефакты

| Артефакт | Ссылка / статус |
|---|---|
| Рабочая папка | [promaladoi/Decentrathon-6.0 → Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9) |
| Текущий коммит кода | [60918ea61d91c9995799392dc350344a66e8af3e](https://github.com/promaladoi/Decentrathon-6.0/commit/60918ea61d91c9995799392dc350344a66e8af3e) |
| Проверенная сохранённая Kaggle-версия | [development №363823158](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363823158); private, embedded snapshot |
| Репозиторий платформы | [BAITC-Hacks/hack-a5a4072f-wither-ai](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai); не изменялся |
| Финальный commit в репозитории платформы | Не подготовлен |
| Публичный notebook из шаблона с финальным commit | Не подготовлен |
| Название команды, состав, контакт капитана | Не предоставлены |
| Демо до 3 минут и презентация | Не подготовлены |

Основа: starter kit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
Проверенный snapshot SHA256:
`c1b9cba94e65421fc0b204c097b725509edd32587c54db5a4b1a9d28961cc7d4`.
Измеренный `solution_sha256`:
`9d65e00dce6240bdc5fa6de66d6df83c338941585e443ccda57b277efed59f52`.
Хеш относится к измеренным байтам снимка; для GitHub-копии отдельно учтены
CRLF/LF. `harness/`, `tests/`, `task/` не менялись по содержимому.

## Восемь собственных кернелов

| Имя `@triton.jit` | Операция | Мин. доля времени |
|---|---|---|
| `vocabulary_partials_kernel` | Частичные max, exp-sum и момент энтропии | 1.80% |
| `vocabulary_finish_kernel` | Объединение частей CE/энтропии | 1.12% |
| `vocabulary_gradient_kernel` | Градиент CE, включая FP16 вход/выход | 0.58% |
| `soup_rmsnorm_forward_kernel` | Qwen2 RMSNorm | 0.68% |
| `soup_rmsnorm_backward_kernel` | Градиент RMSNorm по активациям | 0.91% |
| `soup_rope_forward_kernel` | Вращение Q/K с RoPE | 0.62% |
| `soup_rope_backward_kernel` | Градиент RoPE | 0.49% |
| `soup_nf4_unpack_kernel` | NF4/double-quant scales для MatMul4Bit backward | 2.32% |

Все восемь реально запускаются; суммарный минимум двух solution-запусков,
шаги 3+, — **8.5%** при требовании ≥5%. Доли строк вычислены из CUDA-event
`own_kernel_ms / timed_loop_s`, округлены до 0.01%; минимум каждой строки
берётся отдельно. Пример SwiGLU организаторов исключён из собственной доли.

`half_loss.py` устанавливает instance-only hook исходной Qwen2 loss_function.
FP16 logits читаются с FP32 арифметикой в тех же трёх vocabulary-кернелах;
FP32 CE-input/gradient копии не создаются, выходной градиент — FP16 как
после исходного CastBackward. Forward модели, глобальный LOSS_MAPPING и
FP32 преобразование outputs в Accelerate сохранены. Источники и остальные
алгоритмы описаны в README; чужой Triton-код не копировался.

## Измеренный результат: публичные строки T4

`run_task.py --repeats 2`, порядок база → решение → решение → база.
Медиана времени, максимальный пик каждой стороны; компиляция включена.

| Показатель | База | Решение | Отношение / разница |
|---|---|---|---|
| Полное время `soup train`, с | 81.38 | 72.91 | 1.1162× |
| Пик GPU, MiB, nvidia-smi | 14345 | 11181 | база/решение 1.283× |
| Public eval loss | 2.04964 | 2.04965 | JSON Δ округлено до +0.000%; не точное равенство |
| Собственные кернелы, шаги 3+ | — | 8.5% | Минимум двух запусков |
| Автоматические правила | — | 10/10 PASS | Публичный development запуск |

**194 passed, 1 skipped** solution tests; **79 passed** Soup tests.
Исходные артефакты: [result](verification/half-t4-result.json),
[checks](verification/half-t4-checks.json),
[подтверждение запуска в Kaggle UI](verification/half-t4-run.log),
[source audit](verification/half-source-audit.json),
[half-CE диагностика](verification/half-ce-microbenchmark.json),
[другие микрозамеры](verification/half-t4-microbenchmark.json).

Времена base_1/base_2: **86.09/76.66 с**, solution_1/solution_2:
**72.96/72.85 с**. Разброс базы учтён исходной медианой harness.
Предыдущий пик решения 12381 MiB снизился до 11181 MiB, но старый и новый
полные запуски не являются парным A/B в одной сессии; причинная доля
half CE в разнице времени не заявлена. Скрытые данные жюри не проверялись.

- [x] GPU unit/integration tests и 79 тестов Soup с патчами прошли на T4.
- [x] Два запуска базы и два решения, public eval и все 10 checks завершены.
- [x] README содержит алгоритмы, источники, запуск, реальные результаты и ограничения.
- [ ] Финальный код зафиксирован в требуемом платформой репозитории.
- [ ] Сохранён публичный notebook из шаблона с `COMMIT` финальной версии.
- [ ] Добавлены данные команды, демо и презентация.

Рабочий clone-notebook для личной папки подготовлен отдельно; измеренные
результаты относятся к development snapshot. Текущий `kaggle_notebook_dev.ipynb`
содержит этот же embedded snapshot v9. UI-журнал сохраняет строки таблицы
Kaggle Logs; численные значения сверены с исходными JSON. По указанию пользователя
платформенный репозиторий сейчас не изменяется. Финальная сдача требует
завершить оставшиеся артефакты и повторить запуск с её commit.
