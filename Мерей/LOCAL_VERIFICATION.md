# Verification — полный Kaggle T4 development run

Saved Kaggle full development version: [№363884710](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363884710), private.
Snapshot ZIP SHA256: `1916a1eab83d5ad80877937c6dc1203b63c10e7a151a64d49ca9beb89401403c`.
Measured solution SHA256: `60216a16a1f993117f90e6028e93e4096fbacc4fb21afefbccd3f7f3a2e3a7f0`.
Result SHA256: `53b72beee7141c1b18cb2b7d6b6ea60f796ecbdd23dbb8b05c1b945b9c8b8a7b`; совпадает с checks и audit.
Рабочий code commit: [66bcd9b6585b027561a158448d87c2699f975931](https://github.com/promaladoi/Decentrathon-6.0/commit/66bcd9b6585b027561a158448d87c2699f975931).
Все 27 solution-файла этого commit
побайтно сверены с измеренным source snapshot по GitHub readback receipt.
`kaggle_notebook_dev.ipynb` — точная побайтовая копия входного notebook
этой saved версии, а не новый исполненный clone notebook.


## Окружение и тесты

python `3.13.15`, torch `2.11.0+cu128`, triton `3.6.0`, soup-cli `0.77.0`, transformers `5.16.1`, peft `0.20.0`, trl `0.29.1`, bitsandbytes `0.50.2`, accelerate `1.14.0`. Optional `torchao` удалён до всех тестов и измерений.
**181 passed, 0 skipped** solution; **79 passed** Soup с патчами. Solution pytest: 63.85 с; начальный Soup
pytest: 1.05 с; rule 10: 0.85 с.
Числа взяты **из полного final-run.log**, по отдельным маркерам
`SOLUTION_UNIT_TESTS_BEGIN` и `SOUP_REGRESSION_TESTS_BEGIN`.
Число solution tests не выбирается как максимум pytest summaries;
числовые колонки TIME/LINE не соединяются с BODY.

Предварительный packed smoke №363875832 имел 180 solution PASS и 79 Soup
PASS; source после него получил порог dropout 384 и соответствующий
integration case. Итоговое число выше взято из финального полного запуска,
а не перенесено из smoke. Ранний core smoke №363860917 имел ошибочный
double-cast reference одного RMS теста; исправлен именно reference по
pinned Qwen2 single-cast, с zero tolerance, без ослабления kernel assertion.

## Полное обучение и качество

| Показатель | Базовый Soup | Решение | Отношение / проверка |
| --- | --- | --- | --- |
| Полное время soup train, с | 82.19 | 56.09 | 1.4653× |
| GPU peak, MiB (nvidia-smi) | 14345 | 5275 | база/решение 2.7194× |
| Public eval loss | 2.04962 | 2.04965 | JSON Δ +0.001% |
| Собственные kernels, steps 3+ | — | 15.08% | минимум двух запусков, ≥5% |
| Автоматические правила | — | 10/10 PASS | публичные данные |

| Запуск | Полное время, с | GPU peak, MiB |
| --- | --- | --- |
| base_1 | 80.04 | 14345 |
| solution_1 | 55.97 | 5275 |
| solution_2 | 56.22 | 5275 |
| base_2 | 84.35 | 14345 |

Полное время включает инициализацию, обучение, компиляцию и сохранение
адаптера. Official summary берёт median двух времён (для двух чисел это
равно среднему) и maximum пиков каждой стороны. Порядок official run:
base → solution → solution → base. Loss и `quality_delta_pct` в JSON
округлены; одинаковые напечатанные loss не доказывают побитное равенство.
Проверка допуска +0.5% пройдена на публичных данных.

Все 15 `harness/`, `tests/`, `task/` файла побайтно совпадают с проверенным
родителем. Исходные task, batch/order, шаги, optimizer, scheduler,
LoRA/dropout, settings, tokens, метрики и rewind сохранены в пределах
исходных проверок. Собственная доля исключает оба organizer SwiGLU kernel
и неактивные fallback JIT. CUDA graphs, torch.compile и готовые kernel
libraries не подключаются. Результат автоматических правил — предварительная
проверка; source review жюри остаётся отдельным этапом.


## Все десять правил

| Проверка | Статус | Исходное detail |
| --- | --- | --- |
| 1. every base and solution run finished | PASS | 2 base and 2 solution runs of 2 each |
| 2. no ready-made kernel libraries | PASS | loaded: none; imported in solution/: none; packages changed during the run: none |
| 3. own @triton.jit kernels in solution/ run in training | PASS | add_rmsnorm_bwd_kernel (solution/rmsnorm.py), add_rmsnorm_fwd_kernel (solution/rmsnorm.py), lm_ce_kernel (solution/lmhead.py), lm_scale_gradient_kernel (solution/lmhead.py), nf4_dequant_kernel (solution/nf4.py), real_dropout_backward_kernel (solution/dropout.py), real_dropout_gather_kernel (solution/dropout.py), rope_pad_bwd_kernel (solution/unpad.py), rope_pad_fwd_kernel (solution/unpad.py); not launched in every solution run: _rotate_heads, heads_rows_kernel, rmsnorm_bwd_kernel, rmsnorm_fwd_kernel, rope_kernel; the example's kernels do not count, under any name: example_swiglu_fwd_kernel, example_swiglu_bwd_kernel |
| 4. own kernels together take >= 5% of the training time | PASS | 15.1% of the training time (lowest of 2 runs, steps 3+) |
| 5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind) | PASS | identical within tolerance |
| 6. loss curve follows the base | PASS | base_2: first step 0.00%, mean of all steps 0.01%; solution_1: first step 0.01%, mean of all steps 0.01%; solution_2: first step 0.01%, mean of all steps 0.01% (max 1% and 3%) |
| 7. quality: eval loss at most 0.5% above the base | PASS | +0.001% |
| 8. not worse than the base on time or memory (10% margin) | PASS | time x0.682, memory x0.368 of the base |
| 9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree) | PASS | clean |
| 10. Soup regression tests with your patches | PASS | 79 passed, 14 warnings in 0.85s |

## Source и собственная доля

| Собственный командный @triton.jit | Source | Операция | Мин. доля |
| --- | --- | --- | --- |
| add_rmsnorm_bwd_kernel | solution/rmsnorm.py | Residual/RMSNorm backward с native casts | 1.29% |
| add_rmsnorm_fwd_kernel | solution/rmsnorm.py | Residual add + RMSNorm forward | 1.15% |
| lm_ce_kernel | solution/lmhead.py | Online CE, entropy, argmax и FP32 LSE | 0.58% |
| lm_scale_gradient_kernel | solution/lmhead.py | Actual-scale CE gradient перед FP16 GEMM | 0.31% |
| nf4_dequant_kernel | solution/nf4.py | NF4 decode и double-quant scales | 7.77% |
| real_dropout_backward_kernel | solution/dropout.py | Real-row dropout backward | 1.26% |
| real_dropout_gather_kernel | solution/dropout.py | Native full mask → real-row dropout | 1.71% |
| rope_pad_bwd_kernel | solution/unpad.py | Градиенты packed RoPE и base/LoRA projections | 0.45% |
| rope_pad_fwd_kernel | solution/unpad.py | Q/K RoPE + прямой packed Q/K/V producer | 0.56% |

| Запуск | timed_loop_s (steps 3+) | Сумма own kernel ms | Доля |
| --- | --- | --- | --- |
| solution_1 | 27.07 | 4123.507 | 15.23% |
| solution_2 | 27.223 | 4105.414 | 15.08% |

Активные symbols определены по rule 3, AST измеренного source и ненулевому
timing **в каждом** solution run; их число — **9**, а не число
всех `@triton.jit` в папке. Для каждой строки указан независимый минимум
`100 * kernel_ms / (timed_loop_s * 1000)` за два запуска.
Суммарный минимум считается отдельно:
`min_run(100 * sum(active_own_kernel_ms) / (timed_loop_s * 1000))` =
**15.08%**, steps 3+. Складывать независимые минимумы строк для
получения суммарной доли нельзя.


| JIT вне активной доли | Source | Причина |
| --- | --- | --- |
| _rotate_heads | solution/rope.py | component/fallback; rule 3 не подтверждает запуск в каждом измерении |
| heads_rows_kernel | solution/unpad.py | component/fallback; rule 3 не подтверждает запуск в каждом измерении |
| rmsnorm_bwd_kernel | solution/rmsnorm.py | component/fallback; rule 3 не подтверждает запуск в каждом измерении |
| rmsnorm_fwd_kernel | solution/rmsnorm.py | component/fallback; rule 3 не подтверждает запуск в каждом измерении |
| rope_kernel | solution/rope.py | component/fallback; rule 3 не подтверждает запуск в каждом измерении |

Прочие JIT остаются component/fallback helpers и не добавляются к active
own share. Organizer SwiGLU byte-identical source исключён. Source audit
проверяет командное авторство; hash-сверка сама по себе его не доказывает.
ZIP, embedded notebook, solution hash, result/checks hash и code-commit
readback согласованы. Все 15 protected files сверены побайтно с reference.

## Same-session team reference A/B

| Одна сессия, два запуска каждого | Владимир pinned | Новое решение | Отношение reference/candidate |
| --- | --- | --- | --- |
| Полное время, с (median) | 59.655 | 56.095 | 1.0635× |
| GPU peak, MiB (maximum) | 5641 | 5275 | 1.0694× |
| Training loop, с (median) | 30.79 | 31.78 | диагностика |

Это отдельный **same-session performance diagnostic**, не конкурсный score.
Порядок: reference A1 → official base 1 → candidate B1 → candidate B2 →
official base 2 → reference A2. Между завершёнными дочерними процессами
меняется только статический `solution/`; перед official harness и после
диагностики восстанавливается candidate. Используется неизменённый
`harness.train`: те же task, seed, данные, allocator, monitor, cold-cache
правила и сохранение адаптера. Official quality и все десять правил
проверяют новый candidate; отдельная reference eval этим диагностическим
сравнением не заявляется. [Raw comparison](verification/final-comparison.json).

Предел вывода: это **два запуска каждого варианта**. Полное время нового
варианта ниже в этой сессии, но median самого training loop составляет
31.78 с против 30.79 с
reference. Ускорение именно training loop этим сравнением не подтверждено.
Разница полного времени в основном связана с более долгой частью вне
training loop у reference A1; причины отдельно не измерялись.

| Вариант | Запуск | wall − train_loop, с |
| --- | --- | --- |
| reference | reference_1 | 33.42 |
| reference | reference_2 | 24.31 |
| candidate | solution_1 | 24.33 |
| candidate | solution_2 | 24.3 |

Для оценки устойчивости выигрыша полного времени нужны дополнительные
чередующиеся запуски. Сравнение не доказывает новый абсолютный рекорд
ниже 52.32 с, полученных выбранной базой в другой сессии.


| Вариант | Запуск | Полное время, с | GPU peak, MiB |
| --- | --- | --- | --- |
| reference | reference_1 | 62.54 | 5641 |
| reference | reference_2 | 56.77 | 5641 |
| candidate | solution_1 | 55.97 | 5275 |
| candidate | solution_2 | 56.22 | 5275 |

Выбранная пользователем база — pinned source Владимира, чей проверенный
reference run №363851998 дал 52.32 с и 5641 MiB против
81.83 с и 14345 MiB базового Soup; оба напечатанных eval loss — 2.04968,
все 10 правил PASS. Это другой запуск; его значения не подставляются
в текущий official score и сами по себе не доказывают эффект изменений.
Изменение относительно team reference оценивается в same-session A/B выше.


## Границы проверки и воспроизведение

Решение рассчитано на зафиксированные Qwen2.5-1.5B/NF4, plain frozen MLP,
LoRA q/v, fp16, PyTorch 2.11 и одну T4. Native packed attention — внутренний
version-pinned оператор PyTorch. Поддержка других моделей, layouts,
distributed training, higher-order derivatives и другой версии стека
не заявляется. Guards сохраняют исходные paths при неподдерживаемом
вызове. Whole-run выигрыш каждого семейства отдельно не измерялся;
микробенчмарки не заменяют полный training run. Native dropout full mask
и временный результат всё ещё выделяются. Public Tensor logits и
`return_outputs=True` сохраняют исходный путь с его расходом памяти.


1. Импортируйте [pinned working notebook](templates/kaggle_working_notebook.ipynb)
   в Kaggle. Включите **GPU T4 x2**, **Internet** и выполните **Run All** при
   свободных GPU. Из двух карт harness использует одну.
2. Notebook берёт commit `66bcd9b6585b027561a158448d87c2699f975931` из личного репозитория,
   переходит в `Мерей` и проверяет measured solution SHA256 **до тестов**.
   Для закрытого репозитория добавьте read-only `GITHUB_TOKEN` в Kaggle
   Secrets. `HF_TOKEN` необязателен; токены в исходники не записываются.
3. Он ставит pinned packages и удаляет несовместимый optional `torchao`
   **до всех тестов и измерений**, одинаково для базы и решения.
   Внутри измерений пакеты не меняются. Нужен recorded Kaggle image с
   PyTorch/Triton versions ниже; notebook проверяет их перед тестами.
4. Working clone notebook подготовлен, **но сам ещё не исполнен**.
   Доказательство текущих чисел — [saved embedded run №363884710](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363884710)
   и приложенные raw артефакты. Из личного checkout сначала выполните
   `cd Мерей`; исходный platform template остаётся в
   `templates/kaggle_notebook.ipynb`.

```bash
pip uninstall -y torchao
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" \
  "peft==0.20.0" "trl==0.29.1" "bitsandbytes==0.50.2" \
  "accelerate==1.14.0" pytest
python task/prepare_data.py
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python harness/score.py runs/result.json
```

Окружение измеренного запуска: python `3.13.15`, torch `2.11.0+cu128`, triton `3.6.0`, soup-cli `0.77.0`, transformers `5.16.1`, peft `0.20.0`, trl `0.29.1`, bitsandbytes `0.50.2`, accelerate `1.14.0`.


[result](verification/final-result.json), [checks](verification/final-checks.json), [full Kaggle log](verification/final-run.log), [A/B](verification/final-comparison.json), [source audit](verification/source-audit.json).

Рабочее место — [личная папка Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
Финальная платформенная сдача ещё не оформлена. По указанию пользователя
[платформенный репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) и папки других участников не
изменяются. Для сдачи остаются final commit в репозитории платформы,
публичный saved notebook из исходного шаблона с этим commit и успешным
Run All, данные команды, демо до 3 минут и презентация. Скрытые строки и
другие seed не проверены; окончательный допуск и баллы определяет жюри.

