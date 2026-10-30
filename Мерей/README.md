# Мерей — Soup Kernel Challenge

Kaggle T4: **10/10 ереже PASS**; **241 passed, 1 skipped** solution; **79 passed** Soup с патчами.
Толық уақыт **88.74 → 78.18 с**, GPU peak
**14345 → 7381 MiB**, жеке кернелдер үлесі
кемінде **6.60%**. [Күй және қалған қадамдар](WORK_STATUS.md).

Собственные Triton-кернелы и оптимизация времени жизни активаций для
фиксированного Soup 0.77.0 / Decentrathon 6.0. Основа —
[starter kit](https://github.com/MakazhanAlpamys/soup-kernel-challenge),
commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`; [ТЗ](https://docs.google.com/document/d/1ws-AXhXpcx2pBBVSf5KO7GB5ZZEkBt3L2BPIuT4owWY/edit).
Рабочий commit: [b7c406dea7dbdc379149910efe044dccf155d083](https://github.com/promaladoi/Decentrathon-6.0/commit/b7c406dea7dbdc379149910efe044dccf155d083). Финальная сдача ещё не оформлена.

## Что выполняет решение

`solution.apply()` устанавливает идемпотентный hook после построения
реальной PyTorch/PEFT модели в Soup `SFTTrainerWrapper._setup_transformers`.
Восемь собственных Triton-кернелов находятся в `vocab.py` (3), `rmsnorm.py`
(2), `rope.py` (2), `nf4.py` (1). Half-input loss и новые внутренние метрики
переиспользуют три vocabulary-кернела. Неизменённые два SwiGLU-кернела
организаторов подключены, но исключены из собственной доли.

`vocab.py` делит словарь на плитки по 4096 элементов. Частичные максимум
`m`, сумма `s = sum(exp(z-m))` и момент `t = sum(exp(z-m)*(z-m))`
объединяются с общим максимумом `M`:

```text
S = sum(exp(m-M) * s)
CE = (M - z[target]) + log(S)
T = sum(exp(m-M) * (t + s*(m-M)))
H = log(S) - T/S
gradient = exp(z-M)/S - one_hot(target)
```

Сохранены `none/sum/mean`, ignore mask и нормировка; полностью игнорируемый
mean остаётся NaN с нулевым градиентом. Вероятности пересчитываются в backward,
полная log-softmax не сохраняется. Временная статистика имеет размер
`O(tokens * ceil(vocab/4096))`; исходные logits и их градиент остаются нужны.

`half_loss.py` устанавливает instance-only Qwen loss hook: FP16 данные
загружаются с FP32 арифметикой, loss остаётся FP32, градиент записывается
сразу в FP16, как после исходного CastBackward. Сохранены causal shift,
`ignore_index`, `num_items_in_batch`, kwargs и native Inf/NaN семантика,
включая нулевой denominator. Глобальный LOSS_MAPPING и
`Qwen2ForCausalLM.forward` не заменяются; custom loss и неподдерживаемые
вызовы используют исходный путь. Отдельный MLP hook описан ниже.

RMSNorm сохраняет FP32 накопление и cast нормализованного значения к dtype
входа перед умножением на вес. RoPE читает transposed Q/K без dense-копий,
сохраняет cos/sin и повторяет промежуточные FP16 округления. NF4 объединяет
nibble/codebook и восстановление double-quant scales в backward frozen
bitsandbytes MatMul4Bit; его forward и последующий native matmul сохранены.


## Время жизни логитов внутри расчёта метрик

В исходном пути Accelerate переводит возвращённые моделью FP16 логиты в
FP32. Затем TRL вычисляет энтропию и token accuracy, а внешний Soup recorder
считает per-row cross-entropy. Во время обычного training step наружу
возвращается только loss; полный tensor логитов нужен внутренним метрикам.

`private_logits.py` ограничивает изменение этим внутренним участком.
Обёртка устанавливается на экземпляр фактического внешнего Rewind trainer,
созданного Soup. На время его `compute_loss` используется ContextVar.
Только внутри подходящего вызова Accelerate сохраняет FP16 storage поля
`CausalLMOutputWithPast.logits`; остальные поля проходят прежнюю конвертацию.
Обычные публичные вызовы модели сохраняют исходное FP32 преобразование.
Если внешний вызывающий код запрашивает `return_outputs=True`, исходный
converter восстанавливает FP32 outputs перед возвратом из обёртки.

`half_metrics.py` читает эти FP16 значения с FP32 арифметикой. Для энтропии
используются существующие блочные max, exp-sum и центрированный момент.
Для rewind сохраняются causal shift, `ignore_index=-100`, число supervised
токенов и FP32 среднее каждой строки. Per-token CE считается одним батчем;
добавленная последняя игнорируемая позиция исключается перед исходной
маскированной редукцией. Средние и counts передаются на CPU небольшим
пакетом; целые counts допускаются только в диапазоне точного FP32 представления.

Маски и агрегация TRL, token accuracy, порядок recorder и значения,
возвращаемые пользователю, сохраняют исходный контракт. Кэш логитов между
шагами не создаётся. ContextVar сбрасывается в `finally`, а вложенный
неподходящий вызов временно отключает внешний private scope.

Этот путь ограничен обычным NLL-обучением подходящей Qwen2/LoRA модели с
установленным half-input CE и native AMP FP16. Eval, пользовательский loss,
label smoothing, prompt tuning, context parallelism, дополнительные
поля outputs, наблюдающие forward hooks и `past_index >= 0` сохраняют
исходный путь. Перед пропуском FP32 копии отдельно проверяется, что оба
внутренних обработчика поддерживают полученные logits и labels.

## Перенос приведений типов в frozen NF4 MLP

`narrow_mlp.py` применим только к исходным frozen `bitsandbytes.Linear4bit`
без bias, адаптеров и изменяющих projection hooks, с NF4 весами и явно
выбранным FP16 compute dtype. Для таких слоёв bitsandbytes сначала получает
FP16 результат GEMM, а затем расширяет его до dtype входа. Поэтому FP32
результат проекции в этом конкретном пути состоит из точно представимых
FP16 значений.

Исходный MLP расширяет gate/up до FP32, выполняет SwiGLU и снова приводит
активацию к FP16 перед frozen down projection. Новый путь переносит эти
приведения: gate/up остаются FP16, неизменённый SwiGLU-кернел организаторов
по-прежнему загружает значения в FP32 и выполняет ту же арифметику, а его
выход сразу записывается в FP16 перед down projection. Округление соответствует
исходному FP16 cast перед этим GEMM. Финальный выход MLP расширяется до
исходного FP32 dtype входа.

Для gate и up сохранены **два отдельных** `x.to(torch.float16)`. Объединение
их в один общий tensor изменило бы место сложения двух градиентных ветвей:
накопление могло бы происходить в FP16 вместо исходного FP32. В текущем
пути градиенты проходят те же промежуточные half округления и расширяются
перед исходным сложением ветвей.

Дополнительной квантизации произвольных FP32 активаций здесь нет:
предпосылка о представимости gate/up обеспечивается строгим guard их
производителя. PEFT-обёртки MLP, trainable weights, BF16 compute, другие
типы слоёв, изменённый `Linear4bit.forward` и наблюдатели получают исходный
MLP. `compressed_swiglu.py` содержит общий guard и отдельный диагностический
вариант; текущий setup устанавливает `narrow_mlp`.

## Token accuracy без промежуточной contiguous-копии

В обычной causal ветви TRL создаёт
`outputs.logits[..., :-1, :].contiguous()` перед `argmax(dim=-1)`.
Argmax поддерживает исходный strided view. Само копирование не участвует
в формуле accuracy, маске labels или её распределённой агрегации.

`strided_accuracy.py` клонирует pinned `SFTTrainer.compute_loss` и делает
ровно одно AST-изменение: убирает `.contiguous()` у этого временного
`shift_logits`. Все остальные ветви, masks, gather calls, loss и outputs
сохраняются. Клон вызывается только в private training scope; остальные
вызовы используют исходный метод.

Установка требует TRL 0.29.1 и точный SHA256 исходного метода:
`2b060ef7c0acc470d38c72a7a2e468a0d66e7f33a9008c947522245943723562`.
Отдельная проверка AST допускает только известные потребители
`shift_logits`: prompt re-slice и argmax. При изменении версии, исходника
или потребителей установка отклоняется и остаётся исходная реализация.

Чтобы сохранить семантику `super()`, изменённый метод компилируется внутри
временного класса, после чего его code object связывается с **исходными
globals и исходным `__class__` closure**. Tensor subclass и глобальные
подмены `Tensor.contiguous` не используются. Тесты T4 подтвердили
argmax на strided данных, включая ties, NaN и infinities, и совпадение
полного Soup/TRL/PEFT пути вместе с метриками, rewind и градиентами LoRA.



## Измеренный результат на публичных данных T4

| Показатель | База | Решение | Отношение / проверка |
| --- | --- | --- | --- |
| Полное время soup train, с | 88.74 | 78.18 | 1.1351× |
| GPU peak, MiB, nvidia-smi | 14345 | 7381 | база/решение 1.9435× |
| Public eval loss | 2.04964 | 2.04964 | JSON Δ +0.000% |
| Собственные кернелы, steps 3+ | — | 6.60% | минимум двух запусков, ≥5% |
| Автоматические правила | — | 10/10 PASS | публичные данные |

**241 passed, 1 skipped** solution; **79 passed** Soup с патчами. Полное время включает компиляцию; harness использует медиану
двух времён и максимум пиков каждой стороны. Порядок: base → solution →
solution → base. Округлённый quality_delta_pct не означает точного равенства
loss; проверка допуска +0.5% пройдена.

| Запуск | Полное время, с | GPU peak, MiB |
| --- | --- | --- |
| base_1 | 90.42 | 14345 |
| solution_1 | 78.63 | 7381 |
| solution_2 | 77.73 | 7381 |
| base_2 | 87.06 | 14345 |

Модель, данные, LoRA/dropout, батчи и порядок, число шагов, optimizer,
scheduler, training settings и выходы Soup сохранены. Все 15 файлов
`harness/`, `tests/`, `task/` побайтно совпадают с проверенным родителем.
Готовые kernel libraries, torch.compile, CUDA graphs, activation offloading
и изменение gradient checkpointing в обучении не используются.

[result](verification/t4-result.json), [checks](verification/t4-checks.json), [Kaggle UI log](verification/t4-run.log), [source audit](verification/source-audit.json). [Подробные проверки](LOCAL_VERIFICATION.md).

## Собственное исполнение и авторство

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

Суммарный минимум двух запусков — **6.60%**, шаги 3+.
Доля каждой строки — минимум `100 * kernel_ms / (timed_loop_s * 1000)`;
минимумы строк берутся независимо. Суммарный минимум рассчитывается отдельно.
Неизменённые SwiGLU-кернелы организаторов исключены, включая narrow MLP.

## Как воспроизвести

1. Импортируйте [working clone notebook](templates/kaggle_working_notebook.ipynb)
   в Kaggle, включите **GPU T4 x2** и **Internet**. Он закрепляет commit
   `b7c406dea7dbdc379149910efe044dccf155d083` и работает из `/kaggle/working/repo/Мерей`.
2. Выполните Run All при свободных GPU. Notebook сверяет measured solution
   SHA256 до тестов; `torchao==0.10.0` удаляется до тестов и всех измерений,
   одинаково для базы и решения. Изменения пакетов внутри замеров не делаются.
3. Текущий подтверждённый результат получен из [private development версии
   №363837033](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363837033) с embedded DDC snapshot.
   `kaggle_notebook_dev.ipynb` — точная копия её полного входного notebook.
   Отдельный clone notebook подготовлен, но ещё не исполнен.
4. Локально из корня личного репозитория выполните `cd Мерей`. Для закрытого
   репозитория используйте read-only GITHUB_TOKEN в Kaggle Secrets; токены в
   исходники не записывайте. Исходный платформенный notebook сохранён отдельно.

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

## Происхождение результата и ограничения

Kaggle full version: **363837033**, [ссылка](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363837033).
Snapshot SHA256: `ddc7740aa180e81cbdafa886d4fa44c1698cd2f79ae1cc45e63dbd45bedb033b`.
Measured solution SHA256: `698b33301c9a191f3b8bfadc502a0c15d7a08870579c465cfed119a06a2d4d77`.
Result SHA256: `eecdceb5343a87742d4bf2b1b87e497cd9cd072c52d39dfeb744e532ecf8bb8b`; он проверен против checks JSON.
Рабочий code commit: [b7c406dea7dbdc379149910efe044dccf155d083](https://github.com/promaladoi/Decentrathon-6.0/commit/b7c406dea7dbdc379149910efe044dccf155d083).
Все 32 solution-файла рабочего commit побайтно сверены с измеренным snapshot; measured solution SHA256 совпадает.


Историческая v9 №363823158 имела solution `wall_s=72.91`
и peak `11181 MiB`; текущие значения — `78.18`
и `7381 MiB`. Это **разные Kaggle-сессии**, не парный
v9/new A/B. Разница не приписывается одному изменению; текущий speedup
вычислен неизменённым harness относительно базы той же новой сессии.


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


Guard-проверки оставляют исходный путь для неподдерживаемых dtype/layout,
адаптеров MLP, hooks/наблюдателей, custom loss и изменённого pinned TRL метода.
Public calls/return_outputs сохраняют исходный FP32 контракт; private FP16
storage используется только внутри поддерживаемого training compute_loss.
Эффект каждого семейства на полный запуск отдельно не измерялся.

Работа ведётся в [личной папке Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
По указанию пользователя [платформенный репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) и папки
других участников не изменяются. Для сдачи ещё нужны финальный commit в
требуемом платформой репозитории, публичный сохранённый notebook из исходного
шаблона с этим commit и успешным Run All, данные команды, демо до трёх минут
и презентация. Скрытые строки и другие seed не проверены; финальный допуск
и конкурсный результат определяются отдельно. Текущая development версия private.


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

- [bitsandbytes Linear4bit 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py).
- [Неизменённый пример SwiGLU](https://github.com/MakazhanAlpamys/soup-kernel-challenge/blob/c7c2fcd15972ce8b898d8d3ff732ee192c817ceb/solution/swiglu.py): оба кернела исключены из собственной доли.

