# Проверенные результаты — 30 октября 2026

## Успешный полный запуск на Kaggle T4

[Сохранённая development-версия №363806584](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363806584)
пока private. Она содержит проверяемый ZIP исходников, а не clone финального
commit репозитория платформы. Source snapshot SHA256:
`eaa4e00672e0f2133bcbb1452ae73652c8e9dbaa3c1dc89b280725d0e9f80308`.

Окружение: Python 3.13, Tesla T4, torch `2.11.0+cu128`, Triton `3.6.0`,
Soup `0.77.0`, Transformers `5.16.1`, PEFT `0.20.0`, TRL `0.29.1`,
bitsandbytes `0.50.2`, Accelerate `1.14.0`.
Предустановленный несовместимый optional `torchao==0.10.0` удалён до
тестов и замеров; одинаковое окружение использовано для базы и решения.

- `python -m pytest solution -q`: **165 passed, 1 skipped**, 71.20 с.
- `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`: **79 passed**, 8.43 с.
- `python harness/run_task.py --repeats 2`: все четыре запуска завершились.
- `python harness/check_rules.py`: **ALL CHECKS PASSED**, 10/10.
- `python harness/score.py runs/result.json`: все проверки привязаны к этому
  результату; итоговые баллы зависят от результатов других команд.

Порядок и прямые значения из лога:

| Запуск | Полное время, с | Пик nvidia-smi, MiB |
|---|---|---|
| base_1 | 88.79 | 14345 |
| solution_1 | 81.82 | 12381 |
| solution_2 | 81.82 | 12381 |
| base_2 | 84.56 | 14345 |

Итоговый `summary` из [реального result.json](verification/t4-result.json):

| Показатель | База | Решение |
|---|---|---|
| `wall_s` | 86.68 | 81.82 |
| `gpu_peak_mib` | 14345 | 12381 |
| `train_loop_s` | 59.79 | 59.22 |
| `eval_loss` на публичных строках | 2.04963 | 2.04969 |

`speedup = 1.0594`, `memory_ratio = 1.1586`,
`quality_delta_pct = 0.003`. Полное время включает компиляцию.
Сводное время — округлённая медиана двух запусков каждой стороны;
память — максимум пиков. Это реальные результаты, не цифры базы из ТЗ.

## Все десять проверок

| № | Правило | Результат |
|---|---|---|
| 1 | Все запуски завершились | PASS: 2 base и 2 solution |
| 2 | Нет запрещённых kernel libraries и установки пакетов во время замеров | PASS |
| 3 | Свои `@triton.jit` реально запускаются | PASS: 8 собственных; пример исключён |
| 4 | Доля своих кернелов ≥5% | PASS: 7.3%, минимум двух запусков, шаги 3+ |
| 5 | Та же задача и выходы Soup | PASS: в пределах исходных допусков |
| 6 | Кривая loss | PASS: первый шаг 0.00%, среднее расхождение 0.01% |
| 7 | Eval loss не хуже более чем на 0.5% | PASS: +0.003% |
| 8 | Время и память не хуже более чем на 10% | PASS: время ×0.944, память ×0.863 базы |
| 9 | Чистый замер на T4, одна GPU, исходный allocator | PASS |
| 10 | Регрессионные тесты Soup с патчами | PASS: 79 passed, 14 warnings, 8.20 с |

Исходный [t4-checks.json](verification/t4-checks.json) связан с результатом
по SHA256 `5297c7c570ede337722148f59aeed47f65ea27f66b503662fc6f1f1b6fe193d1`.
Хеш исходников solution из того же отчёта:
`68a2eb35f8d32e68f0ae45efc30a1d01d87daf9c7363667caf4f8c98a0009878`.
Автоматические проверки на публичных строках не являются проверкой скрытых
строк с другим seed; итоговую оценку и авторство проверяет жюри.

## Синтетический microbenchmark того же запуска

[Исходный JSON](verification/t4-microbenchmark-final.json): 256 строк,
словарь 151936, 20 повторов; CUDA events после трёх прогревов, без времени
компиляции. Память — дополнительный PyTorch allocated peak. Это отдельный
показатель, не полное время обучения и не nvidia-smi peak.

| Операция | База, median ms | Решение, median ms | База, extra MiB | Решение, extra MiB |
|---|---|---|---|---|
| CE forward/backward | 5.700400114059448 | 2.3114880323410034 | 445.1259765625 | 148.37841796875 |
| Энтропия | 8.905744075775146 | 0.7046239972114563 | 296.7509765625 | 0.1142578125 |
| RMSNorm forward/backward | 0.5905760228633881 | 0.4042239934206009 | 9.0 | 3.0009765625 |

База энтропии здесь вычисляет dense
`logsumexp(logits) - sum(softmax(logits) * logits)`. Это не отдельное
сравнение с chunked entropy из TRL. Microbenchmark не измеряет RoPE/NF4
и не позволяет приписать каждому кернелу долю общего ускорения.

## Предыдущие проверки

CPU, изолированное Windows-окружение Python 3.12.14 / torch 2.11.0+cpu:
79 тестов Soup с `apply()` прошли; прежняя solution-версия дала
11 passed, 76 skipped. Tiny PEFT Qwen CPU forward/backward и `pip check`
прошли. Протокол: `verification/cpu-checks-current.json`.

Первые T4 smoke tests выявили несовместимый предустановленный torchao и
слишком строгий fp32 absolute tolerance в fp16 smoke test. После очистки
окружения и отдельного mixed-precision допуска GPU-тесты прошли; проверки
loss, градиентов и float32 остались строгими.

Первая полная версия с пятью собственными кернелами, snapshot
`33f542713e1801f499ecf6fade05a7fb0f40eedd8d4c7a327510212fa7ba6ec7`,
прошла 9/10 правил: доля собственных кернелов составила 4.4%, ниже 5%.
После добавления полезных RoPE и NF4 операций повторён полный цикл;
успешный результат восьми кернелов приведён выше.
Старый `verification/t4-microbenchmark.json` относится к предыдущей версии
и не заменяет `t4-microbenchmark-final.json`. `verification/local_report.json`
— историческая статическая проверка, не последний GPU-результат.

`harness/`, `tests/`, `task/` не изменялись. В обучении и замере
использовалась целевая T4; локальная RTX 5060 не использовалась.

## Проверка рабочей GitHub-копии

GitHub-копия `solution/` проверена отдельно: все 14 Python-файлов совпадают
с измеренным snapshot после нормализации CRLF/LF; 11 совпадают побайтно.
В GitHub-копиях трёх файлов (`__init__.py`, `swiglu.py`, `test_kernels.py`)
используются LF, а в измеренном snapshot — CRLF. Поэтому хеши
snapshot и `solution_sha256` выше относятся к измеренным исходникам,
а не объявляются побайтным хешем всей GitHub-копии. [Аудит копии](verification/github-working-copy-audit.json).

JSON и лог последнего запуска сохранены без изменения байтов; SHA256
`verification/t4-result.json` совпадает с `result_sha256` отчёта проверок.
Локальный [final_source_audit.json](verification/final_source_audit.json)
подтверждает совпадение локальных исходников с измеренным snapshot.

## Что осталось для финальной сдачи

Рабочий код расположен по указанию пользователя в
[личной папке Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9),
[коммит кода](https://github.com/promaladoi/Decentrathon-6.0/commit/3c389e9f1259031123026c99057f78b3ad57b671).
Репозиторий платформы оставлен без изменений. Для сдачи нужны финальный
commit в требуемом платформой репозитории, публичный notebook из исходного
шаблона с этим commit и успешным Run All, данные команды, демо и презентация.
Текущая private development-версия с embedded snapshot эти требования не
закрывает.
