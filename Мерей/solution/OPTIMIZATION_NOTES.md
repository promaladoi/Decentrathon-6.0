# Мерей: ускорение текущего решения при прежнем ограничении памяти

Измеренный runtime source: [29419f43d9b9173d1ba5d6824a31589ad3b84df7](https://github.com/promaladoi/Decentrathon-6.0/commit/29419f43d9b9173d1ba5d6824a31589ad3b84df7).
Цель этого этапа — уменьшить полное время относительно текущего решения
Мерея, сохранив GPU peak не выше 5275 MiB. Настройки обучения остаются
исходными; время включает подготовку, компиляцию, обучение и сохранение.

## Сравнение с текущим Мереем в одной серии T4

Порядок: старое решение → новое → новое → старое. Каждый measured training
запускается отдельным процессом с исходными cold-cache и allocator правилами.

| Показатель | Текущий Мерей | Runtime 29419f43 |
| --- | ---: | ---: |
| Полное время двух запусков, с | 52.30 / 54.46 | 48.50 / 49.84 |
| Медиана полного времени, с | 53.38 | 49.17 |
| Максимальный GPU peak, MiB | 5275 | 5175 |
| Медиана timed steps 3–64, с | 25.707 | 25.737 |

Полное время уменьшилось на **7.9%**, peak — на **100 MiB**.
Ускорение steady-state обучения не установлено: время последних 62 шагов
практически одинаково. Результат нельзя описывать как ускорение всех
training steps на 7.9% или переносить на другую сессию.
[Исходный текущий-Мерей A/B](../verification/codex_optimization/current-merei-ab.json).

В отдельной прямой серии с предыдущим runtime
[7e7330b5](https://github.com/promaladoi/Decentrathon-6.0/commit/7e7330b59b693661d479a3b7f01dad6a881db406)
получены 47.69 / 51.60 с против 46.39 / 47.83 с: медианы **49.645 →
47.11 с**, peak **5175 MiB** с обеих сторон. Это другая серия, её секунды
не объединяются с таблицей выше. Более ранние 48.62 с для 7e7330b5 также
получены отдельно и не являются парным сравнением с новым runtime.
[Прямой 7e против 294](../verification/codex_optimization/direct-7e-vs-294.json) ·
[Предыдущая отдельная серия 7e](../verification/codex_optimization/previous-7e-ab.json).

## Что делает выбранный runtime

- Dropout сохраняет восемь real-token mask bits в байте. Источником
  остаётся полный native CUDA draw исходной формы; порядок q/v, scales,
  multiplication order и порог dispatch сохранены.
- Head statistics сразу записывают entropy, argmax и supervised CE в
  конечные позиции. Prompt chunks не создают ненужные CE/LSE buffers.
  Forward statistics используют одну production compilation signature;
  GEMM shapes, chunk size и actual AMP gradient scaling сохранены.
- Batch metadata строится одним проходом с исходными prefix/labels guards.
  QKV/MLP input gradients складываются после отдельных, неизменённых GEMM:
  FP16 результаты расширяются в FP32 и складываются в прежнем порядке.
  Higher-order calls сохраняют обычный Torch fallback.
- В private packed AMP path scaling LoRA q/v переносится в существующие
  RoPE kernels. Сохранены округление raw B output в FP16 после scale и
  две границы FP16 в backward. Public и padded paths остаются прежними.
- В qualified private head повторно используется тот же frozen FP16 cast.
  FP32 Parameter, adapter и public forward не заменяются. Versioned mutation,
  replacement и смена CUDA stream инвалидируют слот. Это не сохранённый
  между запусками tensor или compiler cache; peak проверяется целым run.
- Compile-only warmup дополнительно читает свойство `CompiledKernel.run`,
  чтобы загрузить binary/launcher handles внутри того же измеряемого
  процесса. Launcher **не вызывается**: арифметика и RNG не выполняются,
  готовые compiled caches из другого запуска не используются.
- Для точного native Soup text-SFT callback устраняется дорогое
  сериализующее хеширование замыкания tokenizer. Callback, formatter,
  проверки каждой строки, порядок, токенизация и Arrow writer остаются
  исходными. Только для этого qualified map задаётся новый OS-random
  fingerprint и `load_from_cache_file=False`: строки вычисляются заново,
  чужие или предыдущие результаты не подставляются. Fingerprint не влияет
  на seed Python/NumPy/Torch и числовые выходы, а посторонние Dataset.map
  вызовы сохраняют исходный путь.

Выгрыш отдельных семейств не выводится из общего A/B. Более ранние
промежуточные изменения сами по себе не дали устойчивого выигрыша;
сравнение выше относится только к закреплённой комбинации 29419f43.
Авторство командной основы и двух organizer SwiGLU kernels сохранено;
organizer example kernels не считаются собственным вкладом.

## Проверки и воспроизведение

На T4 для runtime 29419f43 прошли **339 solution tests** и **79 Soup
regression tests**. A/B служит performance diagnostic, а обязательные
quality/admission checks выполняются отдельно неизменённым harness.
Их полный результат: [final-checks.json](../verification/codex_optimization/final-checks.json).
[Measured-runtime checks](../verification/codex_optimization/measured-294-checks.json) ·
[Official result](../verification/codex_optimization/final-result.json) ·
[Source proof](../verification/codex_optimization/source-proof.json) ·
[CPU tokenization diagnostic](../verification/codex_optimization/hash-tokenization.json).
Сопутствующие логи публикуются в
[verification/codex_optimization](../verification/codex_optimization/).
Эти ссылки отделяют admission proof от performance headline; результаты
публичных данных не заменяют hidden-data run и source review жюри.

[Reproduction notebook](optimization_validation.ipynb) закрепляет runtime
29419f43, проверяет версии и source, запускает solution/Soup tests и
обязательный двухпарный official harness с проверками допуска. Отдельный
текущий-Мерей A/B включается явно и сохраняется как diagnostic.
Notebook подготовлен для воспроизведения; он не выдаётся за уже
исполненную saved Kaggle version.

Документация и notebook внутри `solution/` входят в полный solution SHA256.
Поэтому поздняя публикация только документации может иметь другой hash,
чем runtime commit 29419f43, при идентичных production `.py` файлах.
Финальный `checks.json` связывается с опубликованными bytes после отдельного
повтора check_rules; времена относятся к указанному runtime source.
Изменения остаются в личной папке Мерей. Protected `task/`, `harness/` и
`tests/` не редактируются. Окончательная платформенная сдача оформляется
отдельно.
