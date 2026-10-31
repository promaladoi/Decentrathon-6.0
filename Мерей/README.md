# Проверенный лимит GPU: ≤827 MiB

Оптимизированный вариант при 827 MiB: полный пик **827 MiB**;
все 10 автоматических проверок ТЗ пройдены.
Основное сравнение: **Первый вариант при 827 MiB → Оптимизированный вариант при 827 MiB**.
Сумма шагов 3–64: **42.853 с**
(-5.93% к этому варианту сравнения);
вся команда: **70.840 с** (-4.12%).
Положительный процент времени означает замедление.

Ранее опубликованная версия с пиком 5039 MiB обучает шаги быстрее: её одиночный контроль — 21.561 с за шаги 3–64. Новая версия при 827 MiB всё ещё тратит примерно 1.99× этого времени. [Старый быстрый вариант](https://github.com/promaladoi/Decentrathon-6.0/tree/bcbe3e82e1c9483b8d24b77a166e39b9cd787b0e/Мерей/solution) остаётся доступен в Git.

Новая версия использует оперативную память CPU для собственного управления
весами и сохранёнными тензорами; приведённый лимит относится к полному пику GPU.

[Полный отчёт и исходные результаты](verification/codex_memory827/optimization-report.md).
[Подтверждение исходного официального harness/tests](verification/codex_memory827/official-upstream-source-proof.json).
Runtime: `28f8a3426582444fb493e74574caf74eea719c9b`; точный источник solution: `6f4752b5bbd4478410b7724bfe05e475314a6e5c`.

---

# Мерей — Soup Kernel Challenge

## Этап 3: шаги быстрее на 4.834%, память ≤ 5039 MiB

Текущий Мерей 766 → выбранный вариант E на основе Анаис: **23.191 → 22.07 с для шагов 3–64**, медианы четырёх запусков каждого варианта, два блока ABBA. Полная команда: **43.51 → 42.46 с**. Во всех парных и официальных solution runs пик≤5175 MiB. **10/10 правил PASS**, 428 solution tests + отдельный native parity 1 PASS; Soup 79 PASS. Это ускорение самих повторяющихся шагов; секунды прошлых сессий и обязательного native admission с этой парой не смешиваются.

Runtime [`ce2c4e4ab547088abd17dde87997e16f798a4695`](https://github.com/promaladoi/Decentrathon-6.0/commit/ce2c4e4ab547088abd17dde87997e16f798a4695).
[Отчёт и все восемь запусков](verification/codex_stage3/optimization-report.md) · [Checks](verification/codex_stage3/final-checks.json) · [Происхождение](verification/codex_stage3/attribution.md) · [Source proof](verification/codex_stage3/final-source-proof.json) · [Чистый reproduction notebook](templates/stage3_reproduction.ipynb).

Публикуется точный измеренный solution tree `1791f95ed4509df573e61a0230163482caac440d`; отчёт и notebook находятся вне solution. Анаис, остальные папки участников и исходные harness/task/tests сохранены. Ниже — история предыдущих этапов.

## Этап 2: ещё −3.7% полного времени при прежних 5175 MiB

В новой парной серии Kaggle T4 текущий Мерей (`ae2e088`) и выбранный
runtime `766d0a39` дали **48.005 → 46.21 с по медиане**, GPU peak —
**5175 → 5175 MiB**. Старые запуски: 46.28 / 49.73 с; новые:
45.74 / 46.68 с. Порядок: старый → новый → новый → старый.
Timed steps 3–64: 24.895 → 24.9475 с — ускорение самих шагов не установлено.
Выигрыш получен на подготовке полного запуска: повторный поиск метаданных
TRL заменён независимой копией уже вычисленной штатной таблицы Transformers
в том же процессе. Проверки доступности пакетов и версий сохранены.

**352 solution tests PASS + отдельный native setup-prefix test PASS;
79 Soup tests PASS.** Официальные проверки: **10/10 PASS**.

Измеренный и опубликованный runtime:
[766d0a39fe8394be8a1cd280401af1adde6d5206](https://github.com/promaladoi/Decentrathon-6.0/commit/766d0a39fe8394be8a1cd280401af1adde6d5206).
[Отчёт этапа 2](verification/codex_stage2/optimization-report.md) ·
[A/B](verification/codex_stage2/merey-stage2-snapshot_late-ab.json) ·
[Official checks](verification/codex_stage2/final-checks.json) ·
[Source proof](verification/codex_stage2/source-proof.json) ·
[Чистый reproduction notebook](templates/stage2_optimization_validation.ipynb).

Весь `solution/` публикуется побайтно из измеренного runtime. Новые отчёты
и notebook находятся снаружи `solution/`, поэтому его официальный SHA256
не меняется после замера. `harness/`, `task/` и `tests/` сохранены.
Исследованные более медленные GPU-варианты в runtime не включены.
Ниже сохранены результаты предыдущих этапов; их секунды не смешиваются
с новой парной серией. Платформенная сдача остаётся отдельным этапом.

## Этап 1 — история: прежний Мерей → runtime 29419f43

В одной серии Kaggle T4 медиана полного запуска уменьшилась **53.38 →
49.17 с (−7.9%)**, GPU peak — **5275 → 5175 MiB**. Два старых запуска:
52.30 / 54.46 с; два новых: 48.50 / 49.84 с. Timed steps 3–64:
25.707 → 25.737 с; ускорение steady-state обучения этим A/B не заявляется.
**339 solution tests + 79 Soup tests PASS.**

Runtime source: [29419f43d9b9173d1ba5d6824a31589ad3b84df7](https://github.com/promaladoi/Decentrathon-6.0/commit/29419f43d9b9173d1ba5d6824a31589ad3b84df7).
[Что изменено, границы сравнений](solution/OPTIMIZATION_NOTES.md) ·
[Pinned reproduction notebook](solution/optimization_validation.ipynb) ·
[Official admission/quality checks](verification/codex_optimization/final-checks.json).
Обязательный допуск проверяется отдельно от A/B. Документация внутри
solution может публиковаться позднее без изменения runtime `.py`;
финальный checks hash связывает опубликованные bytes.

## Предыдущее подтверждённое состояние — история

Ниже сохранены прежние измерения и ссылки; они относятся к указанным
там старым code commits, а не к runtime 29419f43.

**Kaggle T4: 82.19 → 56.09 с; GPU peak
14345 → 5275 MiB; 10/10 checks PASS.**
**181 passed, 0 skipped** solution; **79 passed** Soup с патчами. Финальная платформенная сдача ещё не оформлена.

Продолжение **командной работы Владимира** с дополнениями Мерея для
фиксированного Soup 0.77.0 / Decentrathon 6.0. [ТЗ](https://docs.google.com/document/d/1ws-AXhXpcx2pBBVSf5KO7GB5ZZEkBt3L2BPIuT4owWY/edit),
[saved Kaggle run](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363884710), рабочий code commit [66bcd9b6585b027561a158448d87c2699f975931](https://github.com/promaladoi/Decentrathon-6.0/commit/66bcd9b6585b027561a158448d87c2699f975931).

## Что изменено и почему

Основа — работа **Владимира из той же команды**, закреплённая commit
`987da36e1efa90652e1a7f8c48afef50685dbc40`. Пользователь подтвердил общую
команду. Удаление padding внутри батча, chunked LM head с онлайн-статистикой,
residual RMSNorm и интеграция frozen NF4 MLP происходят из этой командной
работы. Здесь сохранено её авторство; дополнения Мерея описаны отдельно.
Два `example_swiglu_*` — неизменённый пример организаторов и не входят в
собственную долю команды.

### Реальные токены и attention

Поддерживаемый training path собирает реальные токены **внутри исходного
батча**. Строки разных батчей не смешиваются; состав, порядок и causal
позиции сохраняются. До embeddings и dropout проверяются бинарная
prefix-mask, непустые строки и `labels == -100` на удаляемых позициях.
Неподдерживаемый вызов передаётся исходному forward.

Дополнение Мерея: Triton RoPE producer сразу записывает Q/K/V в packed
layout `[1, heads, real_tokens, head_dim]`, складывая base и LoRA значения,
выполняя RoPE и повторяя K/V до числа Q heads. Поэтому не нужны dense
padding и отдельные gathers вокруг attention. Встроенный оператор
PyTorch 2.11 `aten._efficient_attention_forward` получает Q/K/V,
`int32` cumulative sequence lengths для Q и KV, **обе** максимальные длины
и causal mask type. Equal heads сохраняют native efficient/Cutlass path
на T4. Это разрешённый оператор PyTorch; собственный attention-кернел
или сторонняя kernel library здесь не заявляются. Native autograd
вычисляет attention backward; Triton backward возвращает градиенты Q/K/V
и LoRA-проекций. Ветка padded attention остаётся для component fallback.

### Dropout с исходным RNG

Базовый full-shape CUDA dropout остаётся источником маски: тот же размер
`[batch, padded_sequence, hidden]`, порядок вызовов и продвижение native
генератора. Triton `real_dropout_gather_kernel` читает только реальные
значения и соответствующую часть маски; backward применяет сохранённую
real-row mask и исходный scale. Параметр LoRA dropout не меняется, RNG
не сбрасывается внутри обучения.

Этот путь включается только при `batch * padded_sequence * hidden >=
4 * 384 * 1536`, на pinned PyTorch и с исходным ненаблюдаемым `nn.Dropout`.
Для меньших тензоров сохранён dense native путь: отдельный T4 benchmark
показал, что дополнительные запуски Triton на малых размерах не окупаются.
Dropout всё ещё создаёт full-shape native mask и временный результат;
оптимизация сокращает работу с реальными данными и сохранённое состояние
autograd, а не отменяет исходную генерацию случайных чисел.

### Chunked LM head, loss и метрики

Frozen LM head обрабатывает токены порциями по 512; vocabulary — плитками
по 4096. Командный online-softmax кернел одновременно получает CE,
энтропию, argmax и FP32 log-sum-exp. Для следующей плитки с максимумом
`b`, значениями `z` и предыдущими статистиками `(m, s, u)`:

```text
M = max(m, b)
alpha = exp(m - M)
e = exp(z - M)
u_new = alpha * (u + s * (m - M)) + sum(e * (z - M))
s_new = alpha * s + sum(e)
m_new = M
LSE = m + log(s)
CE = LSE - z[target]
entropy = log(s) - u / s
```

Prompt tokens всё ещё участвуют в исходных entropy/accuracy readers;
`ignore_index=-100` исключает их только из supervised loss и его градиента.
Сохранены causal shift, нормировка `num_items_in_batch`, row losses для
rewind и исходный порядок логирования. Полный `[batch, sequence, vocab]`
тензор в этом приватном training path не материализуется.

Дополнение Мерея исправляет место AMP scale. Истинный upstream scale
должен применяться **до FP16 округления dlogits и head-gradient GEMM**:

```text
g32 = (exp(logits.float() - LSE) - one_hot(target))
g16 = (g32 * (actual_upstream / denominator)).half()
dx16 = g16 @ frozen_weight16
```

В узком поддерживаемом Soup scope считывается текущий native `GradScaler`.
Ожидаемый `dx16` считается при обработке chunk в forward, и этот временный
logits buffer переиспользуется для `g16`. Сохраняются half hidden states,
half hidden gradient и FP32 LSE. Backward сравнивает **фактический** upstream
с ожидаемым: при совпадении возвращает сохранённый gradient, при отличии
пересчитывает supervised logits из собственной half-копии hidden states и
применяет настоящий scale. В общем component path сохраняются исходные
supervised FP16 logits и FP32 LSE для actual-scale backward. Фиксированное
умножение на 1024 и последующая компенсация после half GEMM не используются.

Streaming `LazyLogits` разрешён только внутри внешнего Soup
`compute_loss(return_outputs=False)`. `ContextVar` привязан к точным
объектам модели и input tensors и сбрасывается при любом выходе, включая
исключение. Публичный вызов модели и `return_outputs=True` сохраняют
исходный Tensor-logits forward. Guards отклоняют custom loss, hooks,
другие backend/settings, cache/embeddings/position overrides и
неподдерживаемые модель, layout или AMP конфигурацию.

### NF4, MLP и нормализация

Командный NF4 кернел объединяет nibble/codebook decode и восстановление
double-quant scales. Дополнение Мерея повторяет native rounding
`QuantState.dtype -> requested dtype`, сохраняет native small-M
bitsandbytes dispatch, а для остальных поддерживаемых форм использует
собственный decode и native linear. Матрицы остаются frozen; новые
обучаемые параметры не добавляются.

Frozen NF4 MLP хранит gate/up/SwiGLU активации в FP16 и складывает две
ветви input gradient в FP32. Public output dtype возвращается к исходному
input dtype; `save_for_backward` сохраняет проверку версий и повторный
backward с `retain_graph=True`. Узкие guards сохраняют fallback для
адаптеров, hooks и неподдерживаемых проекций. SwiGLU вычисляет исходный
пример организаторов; он не считается собственным kernel contribution.

RMSNorm копит статистику в FP32, затем выполняет исходный cast
нормализованного значения перед умножением на вес. Дополнение Мерея
сохраняет промежуточное округление градиента перед CastBackward. В fused
residual-add RMSNorm также сохранены promoted dtype суммы, обе ветви
градиента и порядок их округления. Статистика на строку:

```text
r = rsqrt(mean(x.float() ** 2) + eps)
y = weight * (x.float() * r).to(x.dtype)
g = (dy.float() * weight.float()).to(x.dtype).float()
dx = g * r - x * sum(g * x) * r**3 / hidden
```

Последняя формула описывает математику; код отдельно повторяет native cast
boundaries и порядок операций, важные для FP16 backward.

Compile-only warmup запускается в том же измеряемом процессе во время
инициализации Soup. Компиляция входит в полное время; готовые кэши не
поставляются. Warmup dropout не генерирует маску и не продвигает RNG.


## Измеренный результат: native Soup против решения

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


[result](verification/final-result.json), [checks](verification/final-checks.json), [full Kaggle log](verification/final-run.log), [A/B](verification/final-comparison.json), [source audit](verification/source-audit.json). [Все проверки](LOCAL_VERIFICATION.md).

## Сравнение с кодом Владимира в той же сессии

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


Выбранная пользователем база — pinned source Владимира, чей проверенный
reference run №363851998 дал 52.32 с и 5641 MiB против
81.83 с и 14345 MiB базового Soup; оба напечатанных eval loss — 2.04968,
все 10 правил PASS. Это другой запуск; его значения не подставляются
в текущий official score и сами по себе не доказывают эффект изменений.
Изменение относительно team reference оценивается в same-session A/B выше.


## Собственные командные kernels в измеренном обучении

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


## Как воспроизвести

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


## Привязка source и границы результата

Saved Kaggle full development version: [№363884710](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363884710), private.
Snapshot ZIP SHA256: `1916a1eab83d5ad80877937c6dc1203b63c10e7a151a64d49ca9beb89401403c`.
Measured solution SHA256: `60216a16a1f993117f90e6028e93e4096fbacc4fb21afefbccd3f7f3a2e3a7f0`.
Result SHA256: `53b72beee7141c1b18cb2b7d6b6ea60f796ecbdd23dbb8b05c1b945b9c8b8a7b`; совпадает с checks и audit.
Рабочий code commit: [66bcd9b6585b027561a158448d87c2699f975931](https://github.com/promaladoi/Decentrathon-6.0/commit/66bcd9b6585b027561a158448d87c2699f975931).
Все 27 solution-файла этого commit
побайтно сверены с измеренным source snapshot по GitHub readback receipt.
`kaggle_notebook_dev.ipynb` — точная побайтовая копия входного notebook
этой saved версии, а не новый исполненный clone notebook.


Решение рассчитано на зафиксированные Qwen2.5-1.5B/NF4, plain frozen MLP,
LoRA q/v, fp16, PyTorch 2.11 и одну T4. Native packed attention — внутренний
version-pinned оператор PyTorch. Поддержка других моделей, layouts,
distributed training, higher-order derivatives и другой версии стека
не заявляется. Guards сохраняют исходные paths при неподдерживаемом
вызове. Whole-run выигрыш каждого семейства отдельно не измерялся;
микробенчмарки не заменяют полный training run. Native dropout full mask
и временный результат всё ещё выделяются. Public Tensor logits и
`return_outputs=True` сохраняют исходный путь с его расходом памяти.


По ТЗ speedup `s = base_time / solution_time`, memory ratio
`m = base_peak / solution_peak`. Жюри нормирует скорость как
`35 * ln(s) / ln(s_best)` и память как `30 * ln(m) / ln(m_best)`;
значения ниже 1.05 дают 0 соответствующих баллов. Допуск, лучший результат
других команд и README score ещё неизвестны, поэтому итоговые баллы здесь
не прогнозируются.


Рабочее место — [личная папка Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
Финальная платформенная сдача ещё не оформлена. По указанию пользователя
[платформенный репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) и папки других участников не
изменяются. Для сдачи остаются final commit в репозитории платформы,
публичный saved notebook из исходного шаблона с этим commit и успешным
Run All, данные команды, демо до 3 минут и презентация. Скрытые строки и
другие seed не проверены; окончательный допуск и баллы определяет жюри.


## Источники и авторство

- [Командный source Владимира, pinned commit](https://github.com/promaladoi/Decentrathon-6.0/tree/987da36e1efa90652e1a7f8c48afef50685dbc40/%D0%92%D0%BB%D0%B0%D0%B4%D0%B8%D0%BC%D0%B8%D1%80): unpadding, online head statistics, residual RMSNorm, frozen NF4 MLP и compile scheduling. Пользователь подтвердил ту же команду; это командное авторство сохранено.
- [Starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge/tree/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb) и [неизменённый SwiGLU example](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/solution/swiglu.py): два example kernel исключены из собственной доли.
- [Online normalizer calculation, Milakov/Gimelshein](https://arxiv.org/abs/1805.02867): online softmax statistics; формулы реализованы командой.
- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html) и [LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html): интерфейс и идея fusion, без копирования tutorial kernel.
- [Qwen2, Transformers 5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py) и [causal loss](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py): RoPE, RMSNorm, causal shift, casts и нормировка.
- [bitsandbytes 0.50.2 functional](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py), [MatMul4Bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py), [Linear4bit](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py) и [CUDA dispatch](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py): QuantState, NF4 и native dispatch.
- [PyTorch 2.11 CUDA dropout](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu), [native dropout](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Dropout.cpp) и [GradScaler](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/amp/grad_scaler.py): RNG mapping и место loss scale.
- [PyTorch 2.11 native efficient attention](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/attention.cu) и [derivatives](https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml): packed operator arguments и native autograd.
- [Accelerate 1.14.0 output conversion](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py), [AMP wrapper](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py), [TRL 0.29.1](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py) и [Soup rewind 0.77.0](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/rewind_hf.py): внешний consumer, метрики и rewind.

Лицензия — Apache-2.0, см. `LICENSE` и `NOTICE`. Модель и данные скачиваются
исходными скриптами и не включены в репозиторий. Чужие готовые Triton kernels
не выдаются за код команды; вклад Владимира и пример организаторов указаны отдельно.

