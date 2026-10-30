# Мерей — Soup Kernel Challenge

T4-де **10/10 ереже өтті**: solution — **165 passed, 1 skipped**, Soup — **79 passed**. Қайталама өлшем: **86.68 → 81.82 с**, **14345 → 12381 MiB**, жеке кернелдер үлесі кемінде **7.3%**. [Толық күй және дәлелдер](WORK_STATUS.md).

Жұмысты [Kaggle working notebook](templates/kaggle_working_notebook.ipynb) арқылы жалғастырыңыз; ол жеке репозиторийдің бекітілген кодын `Мерей` ішінен іске қосады. Жергілікті жұмыс үшін репозиторий түбірінен `cd Мерей` жасаңыз. [Тексерілген development нұсқасы №363806584](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363806584) әзірге private; финалдық платформалық commit, public notebook, видео және презентация әлі дайын емес.

Басқа қатысушылардың қалталары және бастапқы платформалық notebook өзгертілмейді.

---

# Soup Kernel Challenge — vocabulary reductions, RMSNorm, RoPE и NF4

Собственные Triton-кернелы для фиксированного обучения Soup 0.77.0 на
Decentrathon 6.0. На Kaggle T4 пройдены **все 10 проверок harness** на
публичных данных: ускорение **1.0594×**, отношение памяти база/решение
**1.1586×**, доля восьми собственных кернелов **7.3%**. Финальная сдача
ещё не оформлена: сохранённый development notebook пока private и использует
снимок исходников, а не commit репозитория команды от платформы.

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

Рабочая копия по указанию пользователя находится в папке
[Мерей личного репозитория](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
[Коммит кода](https://github.com/promaladoi/Decentrathon-6.0/commit/3c389e9f1259031123026c99057f78b3ad57b671):
`3c389e9f1259031123026c99057f78b3ad57b671`.
Репозиторий платформы не изменялся. Для разработки корнем проекта служит
папка `Мерей`, содержащая `solution/`, `harness/`, `tests/`, `task/`.

1. Для работы в личном репозитории импортируйте
   [`templates/kaggle_working_notebook.ipynb`](templates/kaggle_working_notebook.ipynb)
   в Kaggle. Включите `GPU T4 x2` и интернет. Notebook клонирует личный
   репозиторий, выбирает проверенный commit кода `3c389e9f1259031123026c99057f78b3ad57b671`
   и переходит в `/kaggle/working/repo/Мерей`; команды запускаются оттуда.
   Локально сначала выполните `cd Мерей` из корня репозитория.
2. Выполните Run All. Рабочий notebook удаляет несовместимый optional
   `torchao` до измерений, сохраняет исходные pinned зависимости и T4-проверку,
   готовит публичные данные, запускает unit tests, обучение и проверки.
3. Для двух повторов каждой стороны используйте `--repeats 2`, как ниже.
   Пока идёт измерение, не запускайте другую работу на GPU.
4. Для финальной сдачи используйте сохранённый без изменений
   `templates/kaggle_notebook.ipynb`: укажите **репозиторий, созданный платформой**,
   и его финальный commit. Личная рабочая папка не заменяет это требование ТЗ.
   Для закрытого репозитория доступен read-only `GITHUB_TOKEN` через Kaggle Secrets;
   пароли и токены в notebook не записывайте.
5. Повторите полный запуск с финальным commit и сохраните публичную версию
   notebook с выводами. Текущий успешный development запуск использует embedded
   snapshot; рабочий clone-notebook подготовлен отдельно и сам ещё не запускался.

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

30 октября 2026 года, Kaggle Tesla T4, публичные строки задачи.
`python harness/run_task.py --repeats 2`: порядок база, решение, решение,
база; компиляция входит в полное время. Таблица повторяет `summary` из
реального [result.json](verification/t4-result.json).
Harness берёт медиану двух времён и максимум пиков памяти каждой стороны.

| Показатель | База | Решение | Результат |
|---|---|---|---|
| Полное время `soup train`, с | 86.68 | 81.82 | 1.0594×; на 5.6% меньше времени |
| Пик GPU, MiB, nvidia-smi | 14345 | 12381 | база/решение 1.1586×; на 13.7% меньше памяти |
| Eval loss, публичная выборка | 2.04963 | 2.04969 | +0.003%; предел +0.5% |
| Доля собственных кернелов, шаги 3+ | — | 7.3% | Минимум двух запусков; требуется ≥5% |
| Rule checks | — | 10/10 PASS | Включая качество, выходы Soup и чистоту замера |

GPU unit/integration tests: **165 passed, 1 skipped**. Soup с патчами:
**79 passed**. [Исходный отчёт проверок](verification/t4-checks.json)
привязан хешем к этому результату; конкурсные баллы зависят от других команд.

[Сохранённая Kaggle development-версия №363806584](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363806584)
пока **private**, со встроенным ZIP исходников. Проверенный snapshot SHA256:
`eaa4e00672e0f2133bcbb1452ae73652c8e9dbaa3c1dc89b280725d0e9f80308`.
Подробности и история: [LOCAL_VERIFICATION.md](LOCAL_VERIFICATION.md).
GitHub-копия `solution/` проверена отдельно: все 14 Python-файлов совпадают
с измеренным snapshot после нормализации CRLF/LF; 11 совпадают побайтно.
Три неизменённых starter-файла (`__init__.py`, `swiglu.py`, `test_kernels.py`)
в GitHub используют LF вместо CRLF измеренного snapshot. Поэтому хеши
snapshot и `solution_sha256` выше относятся к измеренным исходникам,
а не объявляются побайтным хешем всей GitHub-копии. [Аудит копии](verification/github-working-copy-audit.json).

Отдельные синтетические замеры:
[t4-microbenchmark-final.json](verification/t4-microbenchmark-final.json).

Перед финальной сдачей нужны публичный notebook из исходного шаблона с
commit репозитория платформы, данные команды, демо и презентация. Эти
артефакты пока не готовы; текущий успешный запуск относится к development
снимку и публичным данным.

## Ограничения

- Все 10 автоматических проверок пройдены на публичных строках. Скрытые
  строки и другой seed не проверялись; итоговый допуск определяет жюри.
- 7.3% — измеренный минимум двух solution-запусков. Порог ≥5% и остальные
  критерии нужно заново подтвердить при финальном запуске с commit.
- Результат относится ко всему решению вместе с примером SwiGLU. Вклад
  каждого кернела в полное время и пик памяти отдельно не измерялся.
  Размеры плиток фиксированы; autotuning не используется.
- CE/энтропия ускоряются для contiguous float32 logits и int64 targets;
  остальные входы используют исходный PyTorch. Это соответствует пути CE
  с upcast в Transformers; GPU smoke tests прошли.
  Если метрики получают fp16 logits, они сохраняют исходный путь; это
  потенциальное ограничение экономии памяти и доли собственных кернелов.
- RMSNorm ускоряется только для CUDA fp16/fp32, frozen weights и ширины
  до 8192. Trainable norm weights и неподдерживаемые layouts идут в eager.
- Не реализованы second-order gradients для собственного CE. Они не нужны
  фиксированному обучению; `once_differentiable` явно запрещает их.
- На Windows выполнены CPU и статические проверки; полное обучение и
  конкурсные замеры выполнены удалённо на Kaggle T4. RTX 5060 для них не
  использовалась.

## Источники и авторство

Triton-код написан самостоятельно. Источники определяют математические
формулы, порядок dtype conversions и интерфейсы; код чужих кернелов не взят:

- [Triton language API](https://triton-lang.org/main/python-api/triton.language.html):
  загрузки, редукции, экспонента и launch grid.
- [Qwen2RMSNorm и RoPE, Transformers 5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py):
  формулы, точный порядок cast, epsilon и layout Q/K.
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
