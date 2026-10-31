# Ускорение шагов: решение Анаис с ограниченным кэшем

В восьми парных запусках текущего Мерей и выбранного варианта медиана времени **шагов 3–64 снизилась с 23.191 до 22.07 с, на 4.834%**. Полная команда: 43.51 → 42.46 с, изменение 2.413%. Пик выбранного варианта во всех парных и официальных запусках ≤5039 MiB при установленном пределе 5175 MiB. Исходные правила: **10/10 PASS**.

Точка сравнения — текущий Мерей `766d0a39fe8394be8a1cd280401af1adde6d5206`. Выбранный измеренный runtime — `ce2c4e4ab547088abd17dde87997e16f798a4695`, вариант E. За основу взят `Анаис/solution` из `78a2ec998ca7b665d1aa70299aeb6f2f2a105a57`. Её опубликованные 44,70 с и 7040 MiB относятся к другой серии; они не используются для расчёта нашего ускорения. Полный NF4-кэш Анаис превышал наш лимит, поэтому объём удержания ограничен. [Авторство и цепочка исходников](attribution.md).

## Что изменено

Сохранены lossless frozen tied storage и раннее освобождение уже использованных logits из командного варианта Анаис. Каждый FP32-элемент frozen embedding/head проверяется на точную представимость в FP16; только после полного proof тот же связанный Parameter получает компактное хранение. FP32 embedding outputs и публичный head fallback сохранены; trainable параметры не сжимаются. Общий half weight одновременно обслуживает private head и не требует второй копии.

Кэш Анаис ограничен 768 MiB: attention резервирует 294 MiB, fixed-model MLP —474 MiB. В MLP помещается 18 матриц по 26,25 MiB; модель удерживает максимум 766,5 MiB dense NF4 data. Это лимит кэша; полный GPU peak измеряется отдельно. Sources и decoded output защищены identity/version/pointer/dtype/shape/stride checks, учётом CUDA stream и weak-reference lifetime. Miss сохраняет исходную decode-арифметику и GEMM. Заведомый miss без старой записи при полном общем или category budget сразу выполняет raw decode, обходя source signatures. Существующие cache hits сохраняют все guards. Нет межпроцессного или дискового кэша.

В прежнем large-shape dropout-пути FP32-мультипликации и native full-shape RNG сохранены; FP16 store объединён с немедленным native Linear AMP input cast. Backward читает half gradient в FP32 и возвращает FP32 dx. Generic FP32 API и fallback при custom forward/call/F.linear/hooks/неподдерживаемом AMP сохраняются. Это применяется к прежнему подмножеству батчей; отдельный microbenchmark не подменяет полные шаги.

## Текущий Мерей → E: одна парная серия

Порядок — два блока ABBA, восемь отдельных полных процессов с 64 шагами: final_old_1, final_new_1, final_new_2, final_old_2, final_old_3, final_new_3, final_new_4, final_old_4. Медианы пересчитаны из оригинальных per-run JSON. Все 64 шага выполняются; для основной метрики учитываются 62 шага 3–64.

| Метрика | Предыдущий Мерей 766 | E |
|---|---:|---:|
| Шаги 3–64 каждого запуска, с | 21.781 / 23.356 / 23.044 / 23.338 | 21.458 / 21.993 / 22.427 / 22.147 |
| Медиана шагов 3–64, с | 23.191 | 22.07 |
| Медиана цикла обучения, с | 24.905 | 23.775 |
| Полное время каждого запуска, с | 42.12 / 43.81 / 43.41 / 43.61 | 41.96 / 42.46 / 43.01 / 42.46 |
| Медиана полного времени, с | 43.51 | 42.46 |
| GPU peak каждого запуска, MiB | 5175 / 5175 / 5175 / 5175 | 5039 / 5039 / 5039 / 5039 |

| Запуск | Вариант | Шаги 3–64, с | Цикл, с | Вся команда, с | Пик, MiB |
|---|---|---:|---:|---:|---:|
| final_old_1 | previous | 21.781 | 23.39 | 42.12 | 5175 |
| final_new_1 | candidate | 21.458 | 23.16 | 41.96 | 5039 |
| final_new_2 | candidate | 21.993 | 23.72 | 42.46 | 5039 |
| final_old_2 | previous | 23.356 | 25.09 | 43.81 | 5175 |
| final_old_3 | previous | 23.044 | 24.76 | 43.41 | 5175 |
| final_new_3 | candidate | 22.427 | 24.16 | 43.01 | 5039 |
| final_new_4 | candidate | 22.147 | 23.83 | 42.46 | 5039 |
| final_old_4 | previous | 23.338 | 25.05 | 43.61 | 5175 |

Разброс отдельных запусков приведён полностью. Выигрыш относится к этой парной T4-сессии и не гарантирует те же секунды на другой нагрузке или скрытых данных. Полное время включает запуск, подготовку, компиляцию и сохранение; оно не выдаётся за время шагов. Внутренние allocated bytes и оценки размеров весов не заменяют внешний пик неизменённого harness. Контекст GPU из пика не вычитался.

## Отдельная обязательная проверка ТЗ

Native Soup сравнивается только для admission; эта серия не смешивается с current-Мерей A/B. Два solution runs: 42.61 / 42.96 с, медиана полной команды 42.785 с; медиана шагов 3–64 22.3255 с, peak 5039 / 5039 MiB. Native base median 78.3 с. Public eval loss решения 2.04963, отклонение -0.001%. Собственные kernels — 12.8% active time, минимум официальных повторов, steps 3+; организаторские SwiGLU не засчитываются.

Full solution suite: **428 passed, 1 skipped**. Отдельный native metadata parity: **1 passed**; opt-in проверка объясняет штатный пропуск в общем suite. Soup regression: **79 passed**. Пересекающиеся suites не суммируются. [Оригинальные checks](final-checks.json), [solution log](final-solution-tests.log), [native parity](final-native-metadata-parity.log), [official training](final-official-training.log).

Модель Qwen2.5-1.5B-Instruct, исходный NF4,256 train + 256 eval строк, данные/токены/batch order,64 шага,seed,LoRA/dropout,optimizer/schedule и выходы Soup сохранены. Пакеты не менялись во время измерений; готовые внешние библиотеки kernels не добавлялись. `harness/`, `task/`, `tests/` совпадают с предыдущим Мерей по Git trees и проверены по точным snapshot bytes. Скрытые данные и платформенная сдача остаются отдельной проверкой.

## Отбор вариантов

Screenings ниже — отдельные одиночные измерения; их времена не подмешиваются в итоговые восемь запусков. A использует attention-cache294 MiB; B добавляет dropout/cast; C добавляет MLP reservation при общем 768 MiB; D хранит только MLP до 900 MiB без dropout-fusion; E сохраняет C и убирает signature work заведомых misses. Итоговый выбор опирается на финальную парную серию и полный допуск E. Microbenchmarks не объявляются ускорением всей задачи.

| Оригинальный screening | Шаги 3+, с | Полная команда, с | Пик, MiB |
|---|---:|---:|---:|
| [attention_1-result.json](attention_1-result.json) | 21.3 | 41.77 | 4499 |
| [cache768_1-result.json](cache768_1-result.json) | 21.093 | 41.96 | 5039 |
| [fastmiss_1-result.json](fastmiss_1-result.json) | 21.019 | 41.46 | 5039 |
| [fused_1-result.json](fused_1-result.json) | 21.348 | 42.01 | 4499 |
| [mlp900_1-result.json](mlp900_1-result.json) | 21.265 | 41.91 | 5095 |
| [paired_candidate_1-result.json](paired_candidate_1-result.json) | 20.826 | 41.21 | 5039 |
| [paired_candidate_2-result.json](paired_candidate_2-result.json) | 21.332 | 42.01 | 5039 |
| [paired_previous_1-result.json](paired_previous_1-result.json) | 21.334 | 41.86 | 5175 |
| [paired_previous_2-result.json](paired_previous_2-result.json) | 22.803 | 43.36 | 5175 |
| [previous_1-result.json](previous_1-result.json) | 21.302 | 42.06 | 5175 |

Полный all-weight cache Анаис не выбран из-за опубликованного пика 7040 MiB. Ранние A/B/D screenings не установили значимого выигрыша шагов; они не выданы за выбранный runtime. Более ранние неудачные fused GEMM/Gluon эксперименты не включены.

## Исходник и воспроизведение

Runtime commit: `ce2c4e4ab547088abd17dde87997e16f798a4695`. Solution tree: `1791f95ed4509df573e61a0230163482caac440d`. Solution SHA256: `a7aa443b124b4a20e28591ef89039b6c4f3e92c3e164d12e5257af6d9e199016`. Оригинальный result SHA256: `49dc63651388b1db928a2cfb826c7a7a49b71417cc1e62349fc7cf114c85eaca`. Архив `merey-stage3-final-evidence.zip`, SHA256 `7d8dc8ea36e55fd35d7027e74d635a3c6a0e950c66118349568887e8c12ff1d8`; его manifest проверяет все 240 файлов. Выбранные исходные bytes JSON/log копируются без пересборки. [Manifest выбранных доказательств](published-evidence-manifest.json).

Публикуемый `Мерей/solution` берётся целиком из этого измеренного Git tree через Git API. Он не загружается из локального Windows checkout. Новые документы и [чистый reproduction notebook](../../templates/stage3_reproduction.ipynb) находятся вне solution, поэтому не меняют проверенный digest. Сам notebook здесь не выполнялся; его code outputs пусты. Папка Анаис и остальные папки участников не изменяются. Readback публикации и её commit фиксируются отдельной квитанцией после обновления main; самоссылка на ещё не созданный commit в этот измеренный artifact не добавляется.

- [final-result.json](final-result.json)
- [final-checks.json](final-checks.json)
- [final-source-proof.json](final-source-proof.json)
- [artifact-source-proof.json](artifact-source-proof.json)
- [final-current-merei-abba.json](final-current-merei-abba.json)

Два прежних документа из solution сохранены без изменения bytes: [заметки](previous_solution_documents/OPTIMIZATION_NOTES.md) и [notebook](previous_solution_documents/optimization_validation.ipynb).
