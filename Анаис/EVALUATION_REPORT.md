# Итоговая оценка обучения — Анаис

Дата: 30 октября 2026, UTC+5. Источник: собственные запуски в Kaggle на Tesla T4.

## Основной результат

Оптимизированный кандидат: **59.34 с**, ускорение **1.369×**, экономия времени **26.98%** относительно base в той же паре.
Пиковая память **5641 MiB**, экономия **60.68%**. Eval loss **2.04964**, изменение **-0.00049%**.
Цель ≤60 с: **достигнута**. Официальные проверки: **10/10 PASS**; привязка checks к исходным байтам result.json: **совпадает**.

## Методика и результаты

| Группа | Вариант | Обучений | Время, с | Peak MiB | Eval loss |
|---|---|---:|---:|---:|---:|
| starter | base | 2 | 84.34 | 14345 | 2.04966 |
| starter | solution | 2 | 85.32 | 12385 | 2.04967 |
| candidate | base | 1 | 81.26 | 14345 | 2.04965 |
| candidate | solution | 1 | 59.34 | 5641 | 2.04964 |

Starter проверен двумя повторами (base → solution → solution → base), оптимизированный кандидат — одним полным парным запуском (base → solution) для быстрого отбора. Отношения считаются внутри соответствующей группы, а не между разными baseline. Время — полный soup train с компиляцией, не только 64 шага. Память — внешний nvidia-smi peak, не оценка размеров тензоров.

Все обучения: исходная конфигурация, seed 1, 256 training и 256 public eval строк. Eval выполняется plain transformers/peft без патчей кандидата. Исходные harness/, task/ и tests/ сохранены побайтно. Использована одна GPU; корректность условий измерения отражена в проверке 9.

## Качество и loss

- base_1: 64 шагов; первый loss 2.058933, средний 2.056945; первый разрыв 0.000000%, средний относительный разрыв 0.000000%.
- solution_1: 64 шагов; первый loss 2.059355, средний 2.057031; первый разрыв 0.020496%, средний относительный разрыв 0.012542%.

Training loss и held-out loss измеряют разные данные. Допуски: первый loss 1%, кривая 3%, eval loss +0,5%.

## Официальные проверки

- PASS — 1. every base and solution run finished: 1 base and 1 solution runs of 1 each
- PASS — 2. no ready-made kernel libraries: loaded: none; imported in solution/: none; packages changed during the run: none
- PASS — 3. own @triton.jit kernels in solution/ run in training: add_rmsnorm_bwd_kernel (solution/rmsnorm.py), add_rmsnorm_fwd_kernel (solution/rmsnorm.py), heads_rows_kernel (solution/unpad.py), lm_ce_kernel (solution/lmhead.py), nf4_dequant_kernel (solution/nf4.py), rmsnorm_bwd_kernel (solution/rmsnorm.py), rmsnorm_fwd_kernel (solution/rmsnorm.py), rope_pad_bwd_kernel (solution/unpad.py), rope_pad_fwd_kernel (solution/unpad.py); not launched in every solution run: _rotate_heads, rope_kernel; the example's kernels do not count, under any name: example_swiglu_fwd_kernel, example_swiglu_bwd_kernel
- PASS — 4. own kernels together take >= 5% of the training time: 12.6% of the training time (lowest of 1 runs, steps 3+)
- PASS — 5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind): identical within tolerance
- PASS — 6. loss curve follows the base: solution_1: first step 0.02%, mean of all steps 0.01% (max 1% and 3%)
- PASS — 7. quality: eval loss at most 0.5% above the base: -0.000%
- PASS — 8. not worse than the base on time or memory (10% margin): time x0.730, memory x0.393 of the base
- PASS — 9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree): clean
- PASS — 10. Soup regression tests with your patches: 79 passed, 14 warnings in 1.12s

## Воспроизводимость и авторство

- Кандидат: реализация Владимира, `Владимир/saved/v2-d080f2b/solution` из commit `2d77f774725a18c2b16f86d76fe6ec481fe61562`.
- Анаис: независимый запуск, тестирование, сравнение, анализ качества и отчётность. Ядра Владимира не выдаются за авторскую работу Анаис.
- [Notebook кандидата](kaggle_optimized_candidate.ipynb) содержит закреплённый snapshot; [исходники](optimized-candidate-source.zip) и [происхождение/хеши](CANDIDATE_PROVENANCE.json).
- [Полный отчёт](CANDIDATE_REPORT.md), [result.json](candidate-result.json), [checks.json](candidate-checks.json), [SHA256](EVIDENCE_SHA256.json).
- SHA256 result кандидата: `b7d5b8303a117c936d993752731dd8b4a66eb5e10e9316e4e2015f4336e6d1b2`.
- Для финальной проверки выполнить notebook кандидата с двумя повторами, как в его шаблоне. Быстрый одноповторный результат не доказывает устойчивость на другом seed и скрытых данных.
- Проверки harness не заменяют ручное решение жюри и проверку авторства. Финальная отправка на платформу не выполнялась.
