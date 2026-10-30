# Мерей — Soup Kernel Challenge

[Ағымдағы жұмыс күйі мен расталған тексерулер](WORK_STATUS.md). Қазіргі 8-кернелдік нұсқа T4-де 165 solution тестін (1 skipped) және 79 Soup тестін өткізді. Толық қайталама бенчмарк пен барлық ережелерге допуск әлі расталмаған.

Kaggle-де [жұмыс notebook-ын](templates/kaggle_working_notebook.ipynb) импорттап, GPU T4 x2 және Internet қосып, Run All орындаңыз. Ол жеке репозиторийдің бекітілген commit-ін алып, командаларды `Мерей` ішінен орындайды. Жергілікті іске қосу үшін репозиторий түбірінен `cd Мерей` жасаңыз. [Development нәтижелері](https://www.kaggle.com/code/promaladoi/soup-kaggle-development).

Басқа қатысушылардың қалталары өзгертілмейді. Платформаға тапсыруға арналған бастапқы `templates/kaggle_notebook.ipynb` сақталған.

Төменде алгоритм, іске қосу нұсқаулығы және шектеулер берілген.

---

# Soup Kernel Challenge — vocabulary reductions + RMSNorm

Собственные Triton-кернелы для фиксированного обучения Soup 0.77.0 на
Decentrathon 6.0. Статус: реализация подготовлена; окончательные показатели
обучения и допуск по всем правилам **не подтверждены запуском на Kaggle T4**.

Основа: [стартовый репозиторий](https://github.com/MakazhanAlpamys/soup-kernel-challenge),
commit `c7c2fcd15972ce8b898d8d3ff732ee192c817ceb`.
[ТЗ](https://docs.google.com/document/d/1ws-AXhXpcx2pBBVSf5KO7GB5ZZEkBt3L2BPIuT4owWY/edit).

## Что сделано

Публичная точка входа — `solution.apply()`. Она устанавливает идемпотентный
hook после `SFTTrainerWrapper._setup_transformers`. После построения настоящей
PyTorch/PEFT модели подключаются следующие операции:

| Операция | Подключение | Изменение вычислений |
|---|---|---|
| Training cross-entropy | `transformers.loss.loss_utils.fixed_cross_entropy` | Блочные log-sum-exp и CE; вероятности пересчитываются в backward |
| Энтропия TRL | `trl.trainer.sft_trainer.entropy_from_logits` | Энтропия из частичных сумм, без полной матрицы softmax |
| Потери строк rewind | `soup_cli.trainer.rewind_hf.row_losses` | Тот же сдвиг, маска, число токенов и среднее по строкам через наш CE |
| Qwen2 RMSNorm | `Qwen2RMSNorm.forward` | Один row-wise kernel вперёд и один назад для frozen weights |
| Qwen2 RoPE | `modeling_qwen2.apply_rotary_pos_emb` | Парное вращение Q/K вперёд и назад, без копирования transpose views |
| NF4 backward | `bitsandbytes.functional.dequantize_4bit` | Распаковка NF4 вместе с восстановлением double-quant scales для градиента входа |

Дополнительно подключён неизменённый пример SwiGLU организаторов для
сокращения временных активаций MLP. Его два кернела **не наши** и не
учитываются при проверке доли ≥5%; наши восемь кернелов перечислены отдельно.

`solution/vocab.py` делит словарь на блоки по 4096 элементов. Для каждого
блока вычисляются максимум `m`, сумма `s = sum(exp(z-m))` и, для энтропии,
центрированный момент `t = sum(exp(z-m)*(z-m))`. Второй кернел объединяет
блоки с общим максимумом `M`:

```text
S = sum(exp(m-M) * s)
CE = (M - z[target]) + log(S)
T = sum(exp(m-M) * (t + s*(m-M)))
H = log(S) - T/S
```

В backward используется `exp(z-M)/S - one_hot(target)`, с исходной маской
`ignore_index` и нормировкой. Сохранены `none`, `sum`, `mean` и поведение
полностью игнорируемых строк. Удерживается ссылка на исходные логиты и два
float32 значения на токен; полная матрица log-softmax не сохраняется.
Объём временной статистики — `O(tokens * ceil(vocab / 4096))`.
Полные логиты и их градиент по-прежнему нужны.

`solution/rmsnorm.py` сохраняет порядок приведений типов Qwen2: накопление
в float32, приведение нормализованного значения к типу входа **перед**
умножением на вес. Это важно для fp16 активаций и fp32 norm weights после
k-bit подготовки. Backward также учитывает промежуточное округление.
Сохраняются вход, frozen weight и один inverse RMS на строку.

`solution/rope.py` читает Q/K по исходным strides, объединяет попарное
вращение и сохраняет для backward только cos/sin. Промежуточные fp16
произведения округляются в том же порядке, что в PyTorch; FMA отключён.
`solution/nf4.py` объединяет чтение NF4 nibble, lookup codebook,
восстановление compressed absmax и offset в одном кернеле. Патч покрывает
именно `MatMul4Bit.backward`: forward bitsandbytes остаётся исходным
`gemm_4bit`. Dense weight для исходного matmul имеет прежние dtype и форму.
Эти операции добавлены после реального результата первой версии: её
полезные кернелы занимали 4.4%, то есть не достигали обязательных 5%.

Настройки модели, LoRA, dropout, батчи, расписание, оптимизатор и выходы
Soup задаются исходным кодом. TRL продолжает сам считать token accuracy,
маски и агрегацию метрик. Rewind сохраняет исходный recorder и порядок строк.
Файлы `harness/`, `tests/`, `task/` не изменены. Никаких готовых kernel
libraries, `torch.compile`, скрытых данных или CUDA graphs нет.

## Файлы

| Файл | Назначение |
|---|---|
| `solution/__init__.py` | `apply()` |
| `solution/patches.py` | Узкие hooks и eager fallbacks |
| `solution/vocab.py` | Три собственных Triton-кернела CE/энтропии |
| `solution/rmsnorm.py` | Два собственных Triton-кернела RMSNorm |
| `solution/rope.py`, `solution/nf4.py` | Два кернела RoPE и один NF4 decode; собственные GPU-тесты рядом |
| `solution/test_vocab.py`, `test_rmsnorm.py` | Численные GPU-тесты |
| `solution/test_integration.py` | Проверки реальных hooks и маленькой случайной PEFT Qwen |
| `solution/benchmark.py` | Отдельный синтетический microbenchmark на T4 |
| `solution/swiglu.py`, `test_kernels.py` | Неизменённый пример организаторов; его кернелы не считаются собственными |

## Как воспроизвести на Kaggle T4

1. Перенесите этот starter kit с решением в **репозиторий, созданный для
   команды платформой хакатона**. Произвольный личный репозиторий не подходит.
2. Импортируйте `templates/kaggle_notebook.ipynb` в Kaggle. Включите
   `GPU T4 x2` и интернет в аккаунте с подтверждённым телефоном.
3. Укажите URL репозитория и финальный commit в первой ячейке.
   Для закрытого репозитория используйте read-only `GITHUB_TOKEN` через
   Kaggle Secrets. Пароли и токены в notebook не записывайте.
4. Выполните Run All. Шаблон устанавливает pinned зависимости, готовит
   публичные данные, запускает unit tests, обучение и проверки.
5. Для повторов по порядку жюри выполните `--repeats 2`, как ниже.
   Сохраните публичную версию notebook с выводами и указанным commit.

Окружение ТЗ: Python 3.13, torch 2.11, Triton 3.6.0, Soup 0.77.0,
transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2,
accelerate 1.14.0. Установка из исходного шаблона:

В текущем Kaggle image обнаружен предустановленный `torchao==0.10.0`,
который мешает импорту PEFT 0.20.0. Он не используется в задаче NF4;
перед установкой/тестами и любыми замерами удалите его. Это подготовка
одинакового окружения для базы и решения, вне измеряемых процессов.

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

Диагностика и отдельные микрозамеры:

```bash
python harness/run_task.py --only solution --out runs_try --profile
python -m solution.benchmark --rows 256 --out microbench.json
```

Microbenchmark проверяет CE forward/backward, энтропию и RMSNorm на
синтетических тензорах. Он сообщает CUDA-event latency после прогрева и
добавочный PyTorch allocated peak; это **другие показатели**, чем полное
время команды и nvidia-smi peak из официального harness. Компиляция в
официальное время входит. Microbenchmark её исключает.

## Проверенные результаты

Локальный отчёт о выполненных проверках и ограничениях находится в
`LOCAL_VERIFICATION.md`. Статическая проверка из исходного
`harness/check_rules.py` обнаруживает пять собственных кернелов,
не находит запрещённых импортов, синтаксических ошибок и сходства с примером.
Это проверка исходников, а не прохождение всех десяти правил.

Подтверждены CPU-проверки в изолированном Python 3.12 с torch 2.11.0+cpu:
**79 тестов Soup с `apply()` прошли**; solution: **11 passed, 76 skipped**
(CUDA/Triton и неприменимый CPU fp16 autocast). Tiny PEFT Qwen CPU
forward/backward прошёл. Протокол: `verification/cpu-checks-current.json`.

| Конкурсный показатель | База | Решение | Статус |
|---|---|---|---|
| Время `soup train`, с | — | — | Не измерено |
| Пик памяти GPU, MiB | — | — | Не измерен |
| Eval loss на публичных строках | — | — | Не измерен |
| Доля собственных кернелов, шаги 3+ | — | — | Не измерена; требуется ≥5% |
| Все 10 checks на T4 | — | — | Не запускались |

Ссылка на публичную сохранённую Kaggle-версию: **ожидается доступ и запуск**.
Финальный commit в репозитории команды: **ожидается ссылка на репозиторий**.
Они должны быть добавлены сюда и в `SUBMISSION.md` перед сдачей.

## Ограничения

- Triton GPU-код ещё должен пройти компиляцию и численные тесты на целевой T4.
  Нельзя считать решение готовым к сдаче до этих запусков.
- Порог собственного времени ≥5% не гарантирован архитектурой. Если проверка
  4 не пройдёт, требуется профиль и оптимизация других полезных операций,
  а не искусственная нагрузка или задержки.
- Экономия памяти следует из устранения временных тензоров; реальный пик
  и скорость зависят от всего обучения и пока не измерены. Нет autotuning,
  который мог бы увеличить время холодного запуска; размер плитки фиксирован.
- CE/энтропия ускоряются для contiguous float32 logits и int64 targets;
  остальные входы используют исходный PyTorch. Это соответствует пути CE
  с upcast в Transformers; фактический путь метрик проверяется GPU smoke test.
  Если метрики получают fp16 logits, они сохраняют исходный путь; это
  потенциальное ограничение экономии памяти и доли собственных кернелов.
- RMSNorm ускоряется только для CUDA fp16/fp32, frozen weights и ширины
  до 8192. Trainable norm weights и неподдерживаемые layouts идут в eager.
- Не реализованы second-order gradients для собственного CE. Они не нужны
  фиксированному обучению; `once_differentiable` явно запрещает их.
- Полное обучение локально не выполнялось: текущая система — Windows с
  RTX 5060, без совместимого официального Triton runtime. Другой GPU не
  использовался для конкурсных замеров.

## Источники и авторство

Triton-код написан самостоятельно. Источники определяют математические
формулы, порядок dtype conversions и интерфейсы; код чужих кернелов не взят:

- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html):
  загрузки, редукции, экспонента и launch grid.
- [Qwen2RMSNorm, Transformers 5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py):
  точный порядок cast и epsilon.
- [Transformers loss_utils 5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/loss/loss_utils.py):
  causal shift, float32 upcast, `num_items_in_batch`.
- [TRL SFTTrainer 0.29.1](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py):
  no-grad entropy и исходная агрегация метрик.
- [Soup rewind_hf 0.77.0](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/rewind_hf.py):
  supervised-token row loss и token counts.
- [bitsandbytes 0.50.2 functional](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/functional.py)
  и [MatMul4Bit autograd](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py):
  представление QuantState, nibble order и реально используемый backward hook.

Лицензия starter kit и решения — Apache-2.0, см. `LICENSE` и `NOTICE`.
Данные загружаются исходным `task/prepare_data.py`; модель и датасет не
включены в репозиторий.
