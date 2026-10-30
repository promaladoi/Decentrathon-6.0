# Submission — Soup Kernel Challenge

**Development checkpoint: 10/10 automatic checks PASS.** Финальная
платформенная сдача ещё не оформлена; этот файл не заявляет допуск жюри.

Saved Kaggle full development version: [№363884710](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363884710), private.
Snapshot ZIP SHA256: `1916a1eab83d5ad80877937c6dc1203b63c10e7a151a64d49ca9beb89401403c`.
Measured solution SHA256: `60216a16a1f993117f90e6028e93e4096fbacc4fb21afefbccd3f7f3a2e3a7f0`.
Result SHA256: `53b72beee7141c1b18cb2b7d6b6ea60f796ecbdd23dbb8b05c1b945b9c8b8a7b`; совпадает с checks и audit.
Рабочий code commit: [66bcd9b6585b027561a158448d87c2699f975931](https://github.com/promaladoi/Decentrathon-6.0/commit/66bcd9b6585b027561a158448d87c2699f975931).
Все 27 solution-файла этого commit
побайтно сверены с измеренным source snapshot по GitHub readback receipt.
`kaggle_notebook_dev.ipynb` — точная побайтовая копия входного notebook
этой saved версии, а не новый исполненный clone notebook.


## Результат на публичных данных

| Показатель | Базовый Soup | Решение | Отношение / проверка |
| --- | --- | --- | --- |
| Полное время soup train, с | 82.19 | 56.09 | 1.4653× |
| GPU peak, MiB (nvidia-smi) | 14345 | 5275 | база/решение 2.7194× |
| Public eval loss | 2.04962 | 2.04965 | JSON Δ +0.001% |
| Собственные kernels, steps 3+ | — | 15.08% | минимум двух запусков, ≥5% |
| Автоматические правила | — | 10/10 PASS | публичные данные |

**181 passed, 0 skipped** solution; **79 passed** Soup с патчами. Полное время включает инициализацию, обучение, компиляцию и сохранение
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


## Командный вклад

Основа Владимира из той же команды: pinned `987da36e1efa90652e1a7f8c48afef50685dbc40`.
Мерей добавил private streaming scope, actual-scale head gradient,
direct packed RoPE/native attention, real-row native-RNG dropout,
точные NF4/MLP dtype contracts и cast-aware normalization backward.
Формулы, ограничения и primary sources приведены в [README](README.md).

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

Активные symbols определены по rule 3, AST измеренного source и ненулевому
timing **в каждом** solution run; их число — **9**, а не число
всех `@triton.jit` в папке. Для каждой строки указан независимый минимум
`100 * kernel_ms / (timed_loop_s * 1000)` за два запуска.
Суммарный минимум считается отдельно:
`min_run(100 * sum(active_own_kernel_ms) / (timed_loop_s * 1000))` =
**15.08%**, steps 3+. Складывать независимые минимумы строк для
получения суммарной доли нельзя.


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


## Артефакты и оставшиеся действия

| Артефакт | Статус |
| --- | --- |
| Личный code commit | [66bcd9b6585b027561a158448d87c2699f975931](https://github.com/promaladoi/Decentrathon-6.0/commit/66bcd9b6585b027561a158448d87c2699f975931), только папка Мерей |
| Saved full development notebook | [№363884710](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363884710), private, immutable embedded source |
| Working clone notebook | [Подготовлен](templates/kaggle_working_notebook.ipynb), ещё не исполнен |
| Raw verification | [result](verification/final-result.json), [checks](verification/final-checks.json), [full Kaggle log](verification/final-run.log), [A/B](verification/final-comparison.json), [source audit](verification/source-audit.json) |
| Platform final commit | Не оформлен; платформенный репозиторий не изменялся |
| Public saved notebook с platform commit | Не оформлен |
| Данные команды | Не заполнены |
| Демо до 3 минут и презентация | Не подготовлены |

Рабочее место — [личная папка Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
Финальная платформенная сдача ещё не оформлена. По указанию пользователя
[платформенный репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) и папки других участников не
изменяются. Для сдачи остаются final commit в репозитории платформы,
публичный saved notebook из исходного шаблона с этим commit и успешным
Run All, данные команды, демо до 3 минут и презентация. Скрытые строки и
другие seed не проверены; окончательный допуск и баллы определяет жюри.

