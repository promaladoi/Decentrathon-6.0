# Мерей — Soup Kernel Challenge

Kaggle T4-де **10/10 ереже өтті**: solution — **194 passed, 1 skipped**, Soup — **79 passed**. Толық уақыт **81.38 → 72.91 с**, GPU peak **14345 → 11181 MiB**, жеке кернелдер үлесі кемінде **8.5%**. Public eval loss **2.04964 → 2.04965**. [Толық күй және дәлелдер](WORK_STATUS.md).

Жұмысты [Kaggle working notebook](templates/kaggle_working_notebook.ipynb) арқылы жалғастырыңыз; ол жеке репозиторийдің бекітілген кодын `Мерей` ішінен іске қосады. Жергілікті жұмыс үшін репозиторий түбірінен `cd Мерей` жасаңыз. [Тексерілген development нұсқасы №363823158](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363823158) әзірге private және embedded snapshot қолданады. Финалдық платформалық commit, public notebook, видео және презентация әлі дайын емес. Басқа қатысушылардың қалталары мен платформа репозиторийі өзгертілмейді.

---

# Soup Kernel Challenge — half-input CE, RMSNorm, RoPE и NF4

Собственные Triton-кернелы для фиксированного обучения Soup 0.77.0 на
Decentrathon 6.0. На публичных данных Kaggle T4 пройдены **все 10 проверок**:
ускорение **1.1162×**, отношение памяти база/решение **1.283×**.
Сдача ещё не оформлена: текущая Kaggle-версия private и использует снимок
исходников, а работа по указанию пользователя ведётся в личной папке `Мерей`.

Основа: [starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge),
commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`;
[ТЗ](https://docs.google.com/document/d/1ws-AXhXpcx2pBBVSf5KO7GB5ZZEkBt3L2BPIuT4owWY/edit).
Рабочий [коммит кода 60918ea61d91c9995799392dc350344a66e8af3e](https://github.com/promaladoi/Decentrathon-6.0/commit/60918ea61d91c9995799392dc350344a66e8af3e)
находится в [личной папке Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).

## Что изменено

`solution.apply()` устанавливает идемпотентный hook после построения
настоящей PyTorch/PEFT-модели в `SFTTrainerWrapper._setup_transformers`.

| Операция | Подключение | Реализация |
|---|---|---|
| Training CE с FP16 входом | `Qwen2ForCausalLM.instance.loss_function` | FP32 вычисления из FP16 загрузок; градиент сразу в FP16 |
| FP32 CE | `transformers.loss.loss_utils.fixed_cross_entropy` | Блочные log-sum-exp, пересчёт вероятностей в backward |
| Энтропия TRL | `sft_trainer.entropy_from_logits` | Частичные суммы вместо полной матрицы softmax |
| Потери строк rewind | `soup_cli.trainer.rewind_hf.row_losses` | Исходный causal shift, маска, token counts и средние |
| RMSNorm | `Qwen2RMSNorm.forward` | Слитые forward/backward для frozen weights |
| RoPE | `modeling_qwen2.apply_rotary_pos_emb` | Парное вращение Q/K с исходными strides |
| NF4 | `bitsandbytes.functional.dequantize_4bit` | NF4 и double-quant scales для `MatMul4Bit.backward` |

Восемь собственных `@triton.jit` находятся в `vocab.py` (3), `rmsnorm.py`
(2), `rope.py` (2) и `nf4.py` (1). Неизменённый пример SwiGLU организаторов
также подключён для активаций MLP; его два кернела **не считаются нашими**.
Half-input CE использует те же три vocabulary-кернела и не добавляет новых.

`vocab.py` делит словарь на плитки по 4096 элементов. Для каждой вычисляются
максимум `m`, сумма `s = sum(exp(z-m))` и для энтропии момент
`t = sum(exp(z-m)*(z-m))`. Объединение частей с общим максимумом `M`:

```text
S = sum(exp(m-M) * s)
CE = (M - z[target]) + log(S)
T = sum(exp(m-M) * (t + s*(m-M)))
H = log(S) - T/S
```

Сохраняются входные логиты, targets, count и два FP32 значения на токен;
полная матрица log-softmax не сохраняется. Вероятности пересчитываются в
backward; временная статистика имеет объём `O(tokens * ceil(vocab/4096))`.

`half_loss.py` заменяет `loss_function` только у подходящего экземпляра
Qwen2 внутри PEFT. Исходный forward и глобальный `LOSS_MAPPING` не меняются.
Вместо полного `logits.float()` данные FP16 загружаются и приводятся к FP32
внутри кернела. Loss и вся арифметика нормализации остаются FP32. Градиент
записывается сразу в FP16, как исходный `CastBackward` после FP32 CE.
Сохраняются causal shift, `ignore_index`, `num_items_in_batch` и kwargs-семантика HF.

Возвращаемые logits не заменяются: Accelerate выполняет прежнее преобразование
outputs в FP32, поэтому TRL и rewind получают тот же dtype и значения.
Полные logits для метрик и FP16 вход для backward по-прежнему существуют.
Кэша логитов и состояния между шагами нет. Полностью игнорируемый mean
остаётся NaN с нулевым градиентом. Отдельная ветка для бесконечного upstream
сохраняет native NLL/log-softmax Inf/NaN-семантику, включая нулевой denominator.

RMSNorm сохраняет FP32 накопление и cast нормализованного значения к dtype
входа перед умножением на вес. RoPE читает transposed Q/K без dense-копий,
сохраняет только cos/sin и повторяет промежуточные fp16 округления.
NF4 объединяет nibble/codebook и восстановление сжатых scales; forward
bitsandbytes и последующий matmul остаются исходными.

Модель, LoRA/dropout, батчи и их порядок, число шагов, оптимизатор, schedule
и выходы Soup не меняются. `harness/`, `tests/`, `task/` сохранены по содержимому.
Готовые kernel libraries, `torch.compile` и CUDA graphs в обучении не используются.

## Как воспроизвести

1. Импортируйте [working notebook](templates/kaggle_working_notebook.ipynb) в
   Kaggle, включите **GPU T4 x2** и **Internet**. Он клонирует личный
   репозиторий с commit `60918ea61d91c9995799392dc350344a66e8af3e` и работает
   из `/kaggle/working/repo/Мерей`. Локально корень проекта — папка `Мерей`.
2. Выполните Run All без параллельной GPU-работы. Несовместимый optional
   `torchao==0.10.0` удаляется до тестов и замеров, одинаково для базы и решения.
3. Подтверждённые ниже результаты получены из embedded snapshot версии
   №363823158; этот же текущий снимок находится в
   `kaggle_notebook_dev.ipynb`. Подготовленный clone-notebook сам ещё не запускался.
4. Для финальной сдачи нужен исходный `templates/kaggle_notebook.ipynb`
   с commit **репозитория, созданного платформой**, и публичная сохранённая
   версия с успешным Run All. Сейчас платформа оставлена без изменений.

Python 3.13, torch `2.11.0+cu128`, Triton `3.6.0`; подготовка и команды:

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

Для закрытого репозитория используйте read-only `GITHUB_TOKEN` через Kaggle
Secrets; токены в notebook не записывайте. Диагностика отдельно:
`python harness/run_task.py --only solution --out runs_try --profile`.

## Подтверждённый результат на T4

30 октября 2026, публичные строки, порядок база → решение → решение → база.
Полное время включает компиляцию. Harness берёт медиану двух времён и
максимальный пик памяти каждой стороны.

| Показатель | База | Решение | Результат |
|---|---|---|---|
| Полное время `soup train`, с | 81.38 | 72.91 | 1.1162×; на 10.4% меньше времени |
| Пик GPU, MiB, nvidia-smi | 14345 | 11181 | база/решение 1.283×; на 22.1% меньше памяти |
| Public eval loss | 2.04964 | 2.04965 | Δ в JSON округлено до +0.000%; предел +0.5% |
| Собственные кернелы, шаги 3+ | — | 8.5% | Минимум двух запусков; требуется ≥5% |
| Rule checks | — | 10/10 PASS | Включая выходы Soup, качество и чистоту замера |

**194 passed, 1 skipped** solution tests; **79 passed** Soup tests с патчами.
Округлённый `quality_delta_pct=0.0` не означает равенство loss: значения выше
различаются. Конкурсные баллы зависят от других команд и решения жюри.

| Запуск | Время, с | Пик, MiB |
|---|---|---|
| base_1 | 86.09 | 14345 |
| solution_1 | 72.96 | 11181 |
| solution_2 | 72.85 | 11181 |
| base_2 | 76.66 | 14345 |

У базы заметный разброс; для итогового сравнения использована именно медиана
harness. Предыдущая версия имела пик 12381 MiB, новая — 11181 MiB
(ещё на 1200 MiB, 9.7%, ниже). Старый и новый замеры не являются парным A/B
в одной сессии; разницу их времён нельзя целиком приписать одному изменению.

Доказательства: [result](verification/half-t4-result.json),
[checks](verification/half-t4-checks.json),
[журнал из таблицы Kaggle Logs](verification/half-t4-run.log),
[аудит источников](verification/half-source-audit.json),
[подробный отчёт](LOCAL_VERIFICATION.md).
[Сохранённая Kaggle-версия №363823158](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363823158)
пока **private**. Журнал содержит отображённые строки таблицы Kaggle Logs;
численные значения в отчёте сверены с исходными result/checks JSON.
Snapshot SHA256:
`c1b9cba94e65421fc0b204c097b725509edd32587c54db5a4b1a9d28961cc7d4`.
Измеренный `solution_sha256`:
`9d65e00dce6240bdc5fa6de66d6df83c338941585e443ccda57b277efed59f52`.
Это хеши измеренного снимка; GitHub-копия может отличаться окончаниями CRLF/LF.

## Ограничения и оставшаяся сдача

- Все проверки относятся к публичным строкам и указанному seed. Скрытая
  выборка жюри и другие seed не проверялись; итоговый допуск определяет жюри.
- Half CE ускоряется только для поддерживаемых CUDA FP16 logits и int64 labels.
  Custom loss, CPU, FP32 и explicit `shift_labels` сохраняют исходный путь.
  RMSNorm требует frozen weights; неподдерживаемые layouts идут через eager.
- Полная матрица logits остаётся нужна метрикам. Отдельный
  [half-CE microbenchmark](verification/half-ce-microbenchmark.json) для
  256/1024/2048 строк — диагностика, не замена полного обучения.
  [Обычный microbenchmark](verification/half-t4-microbenchmark.json) исключает
  компиляцию и измеряет дополнительную PyTorch allocated память.
- Неизменённый пример SwiGLU участвует в общем результате, но исключён из
  собственной доли. Эффект каждого семейства на полный запуск отдельно не измерялся.
- Для финальной сдачи ещё нужны commit платформенного репозитория, публичный
  notebook из шаблона с этим commit, данные команды, демо до трёх минут и презентация.
  По указанию пользователя сейчас изменяется только личная папка `Мерей`.

## Источники и авторство

Triton-код написан самостоятельно; источники задают формулы, интерфейсы и
порядок dtype conversions. Чужие реализации Triton не копировались.

- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html).
- [Qwen2RMSNorm/RoPE, Transformers 5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py).
- [ForCausalLMLoss/fixed CE 5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py): shift, FP32 upcast и нормировка.
- [Instance loss_function setter](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py).
- [Accelerate 1.14.0 output conversion](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py) и [AMP wrapper](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py).
- [TRL SFTTrainer 0.29.1](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py) и [Soup rewind 0.77.0](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/rewind_hf.py).
- [bitsandbytes 0.50.2 functional](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py) и [MatMul4Bit backward](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py): NF4/QuantState.

Лицензия — Apache-2.0, см. `LICENSE` и `NOTICE`. Модель и датасет загружаются
исходными скриптами и не включены в репозиторий. Оригинальные README команды
и личной папки сохранены в `docs/`.
