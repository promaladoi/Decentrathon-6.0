# Проверенные результаты — полный DDC запуск

Kaggle full version: **363837033**, [ссылка](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363837033).
Snapshot SHA256: `ddc7740aa180e81cbdafa886d4fa44c1698cd2f79ae1cc45e63dbd45bedb033b`.
Measured solution SHA256: `698b33301c9a191f3b8bfadc502a0c15d7a08870579c465cfed119a06a2d4d77`.
Result SHA256: `eecdceb5343a87742d4bf2b1b87e497cd9cd072c52d39dfeb744e532ecf8bb8b`; он проверен против checks JSON.
Рабочий code commit: [b7c406dea7dbdc379149910efe044dccf155d083](https://github.com/promaladoi/Decentrathon-6.0/commit/b7c406dea7dbdc379149910efe044dccf155d083).
Все 32 solution-файла рабочего commit побайтно сверены с измеренным snapshot; measured solution SHA256 совпадает.


## Окружение и выполненные тесты

python `3.13.15`, torch `2.11.0+cu128`, triton `3.6.0`, soup-cli `0.77.0`, transformers `5.16.1`, peft `0.20.0`, trl `0.29.1`, bitsandbytes `0.50.2`, accelerate `1.14.0`. T4; optional torchao удалён до всех тестов/замеров.
**241 passed, 1 skipped** solution; **79 passed** Soup с патчами. Solution pytest: 76.39 с;
начальный Soup pytest: 8.52 с.
Smoke №363836042 имел 241 passed, 1 skipped и 79 passed Soup;
текущие числа выше извлечены именно из полного [журнала](verification/t4-run.log).

Первый smoke №363834978 завершился с 236 passed, 1 failed, 1 skipped:
повторная установка converter теряла ссылку на исходное преобразование.
Обёртка теперь сохраняет и восстанавливает эту ссылку; добавлен regression
test. Обучение в том неудачном запуске не выполнялось.
[Журнал ошибки](verification/experiments/smoke-363834978-failed.log) сохранён.
Финальная локальная CPU-проверка: 28 passed, 214 skipped; GPU-пропуски
покрыты завершёнными T4 тестами выше.

## Полное обучение и качество

| Показатель | База | Решение | Отношение / проверка |
| --- | --- | --- | --- |
| Полное время soup train, с | 88.74 | 78.18 | 1.1351× |
| GPU peak, MiB, nvidia-smi | 14345 | 7381 | база/решение 1.9435× |
| Public eval loss | 2.04964 | 2.04964 | JSON Δ +0.000% |
| Собственные кернелы, steps 3+ | — | 6.60% | минимум двух запусков, ≥5% |
| Автоматические правила | — | 10/10 PASS | публичные данные |

| Запуск | Полное время, с | GPU peak, MiB |
| --- | --- | --- |
| base_1 | 90.42 | 14345 |
| solution_1 | 78.63 | 7381 |
| solution_2 | 77.73 | 7381 |
| base_2 | 87.06 | 14345 |

Полное время включает компиляцию. Summary — медиана двух времён и максимум
NVML/nvidia-smi пиков. Public eval и округлённый quality_delta_pct взяты
из result JSON; это не заявление точного равенства loss.

## Все десять правил

| Проверка | Результат | Исходное detail |
| --- | --- | --- |
| 1. every base and solution run finished | PASS | 2 base and 2 solution runs of 2 each |
| 2. no ready-made kernel libraries | PASS | loaded: none; imported in solution/: none; packages changed during the run: none |
| 3. own @triton.jit kernels in solution/ run in training | PASS | soup_nf4_unpack_kernel (solution/nf4.py), soup_rmsnorm_backward_kernel (solution/rmsnorm.py), soup_rmsnorm_forward_kernel (solution/rmsnorm.py), soup_rope_backward_kernel (solution/rope.py), soup_rope_forward_kernel (solution/rope.py), vocabulary_finish_kernel (solution/vocab.py), vocabulary_gradient_kernel (solution/vocab.py), vocabulary_partials_kernel (solution/vocab.py); the example's kernels do not count, under any name: example_swiglu_fwd_kernel, example_swiglu_bwd_kernel |
| 4. own kernels together take >= 5% of the training time | PASS | 6.6% of the training time (lowest of 2 runs, steps 3+) |
| 5. same task and Soup outputs (steps, adapter, settings, tokens, metrics, rewind) | PASS | identical within tolerance |
| 6. loss curve follows the base | PASS | base_2: first step 0.00%, mean of all steps 0.01%; solution_1: first step 0.00%, mean of all steps 0.01%; solution_2: first step 0.00%, mean of all steps 0.01% (max 1% and 3%) |
| 7. quality: eval loss at most 0.5% above the base | PASS | +0.000% |
| 8. not worse than the base on time or memory (10% margin) | PASS | time x0.881, memory x0.515 of the base |
| 9. clean measurement (a T4, idle GPUs, one GPU, harness allocator, peaks agree) | PASS | clean |
| 10. Soup regression tests with your patches | PASS | 79 passed, 14 warnings in 8.08s |

## Доля собственных кернелов

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

| Запуск | timed_loop_s, шаги 3+ | Сумма собственных кернелов |
| --- | --- | --- |
| solution_1 | 47.497 | 6.60% |
| solution_2 | 47.107 | 6.63% |

Формула: `100 * sum(own_kernel_ms) / (timed_loop_s * 1000)` только для
восьми символов из source audit. Fully-qualified имена сопоставлены по
последнему символу; organizer SwiGLU исключён. Итоговый минимум — 6.60%.

## Дополнительные диагностические сравнения

[MLP diagnostic](verification/mlp-diagnostic.json) и
[head diagnostic](verification/head-diagnostic.json) используют одинаковые
случайные веса, три сбалансированных AB/BA раунда, CUDA events и wall time
после прогрева. Forward/input gradients совпали побитно; у head совпали
также logits и loss. Память — дополнительный PyTorch allocated peak одного
вызова, а не NVML peak обучения. Сводные baseline MLP median объединяют
оба сравнения; внутри исходного JSON сохранены отдельные парные раунды.
`narrow` выбран по устойчивому выигрышу обоих размеров. Compressed-only
SwiGLU и пересчёт frozen LM head **не установлены в обучении**: первый
уступает narrow, второй медленнее с большим измеренным extra peak.


| Синтетическая операция | Вариант | База ms | Вариант ms | База extra bytes | Вариант extra bytes |
| --- | --- | --- | --- | --- | --- |
| MLP [4, 128, 1536] | compressed | 5.94047999382019 | 6.156304121017456 | 95043584 | 113393664 |
| MLP [4, 128, 1536] | narrow | 5.94047999382019 | 4.771568059921265 | 95043584 | 59768832 |
| MLP [4, 512, 1536] | compressed | 20.009967803955078 | 20.028400421142578 | 379584512 | 365838336 |
| MLP [4, 512, 1536] | narrow | 20.009967803955078 | 16.696640014648438 | 379584512 | 196083712 |
| LM head rows=256 | recompute | 14.494607925415039 | 19.51755142211914 | 623117312 | 624690176 |
| LM head rows=2048 | recompute | 92.56632232666016 | 96.0726547241211 | 1717699584 | 1730282496 |

Историческая v9 №363823158 имела solution `wall_s=72.91`
и peak `11181 MiB`; текущие значения — `78.18`
и `7381 MiB`. Это **разные Kaggle-сессии**, не парный
v9/new A/B. Разница не приписывается одному изменению; текущий speedup
вычислен неизменённым harness относительно базы той же новой сессии.


## Источники и границы проверки

[result](verification/t4-result.json), [checks](verification/t4-checks.json), [Kaggle UI log](verification/t4-run.log), [source audit](verification/source-audit.json). Result SHA256, solution SHA256 и snapshot SHA256
взаимно сверены. Source audit подтверждает восемь собственных kernels,
отсутствие запрещённых импортов/копий и неизменность 15 защищённых файлов.
Все изменения кода относительно v9 находятся только в solution/.
Тесты охватывают реальные Qwen/PEFT/Accelerate paths, публичные outputs,
метрики, rewind и LoRA gradients, а также отдельные MLP и head контракты.
Численная точность полного обучения проверена исходными допусками harness.
Авторство и финальный допуск не выводятся из одного public запуска.

Работа ведётся в [личной папке Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
По указанию пользователя [платформенный репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) и папки
других участников не изменяются. Для сдачи ещё нужны финальный commit в
требуемом платформой репозитории, публичный сохранённый notebook из исходного
шаблона с этим commit и успешным Run All, данные команды, демо до трёх минут
и презентация. Скрытые строки и другие seed не проверены; финальный допуск
и конкурсный результат определяются отдельно. Текущая development версия private.

