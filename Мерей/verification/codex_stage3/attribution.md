# Происхождение решения третьего этапа

Этот документ фиксирует происхождение кода; он не заменяет исходные лицензии, комментарии авторов, измерения и допуск ТЗ.

Непосредственная основа — [`Анаис/solution`](https://github.com/promaladoi/Decentrathon-6.0/tree/78a2ec998ca7b665d1aa70299aeb6f2f2a105a57/Анаис/solution) из commit `78a2ec998ca7b665d1aa70299aeb6f2f2a105a57`, Git tree `6b95d4887398eed8802ef008ed843ff02b58721c`. Взяты её опубликованные исходники V7. Папка Анаис используется для чтения; результат готовится в Мерей.

Анаис V7 опирается на выбранный командный v45 Нұржігіта: measured source commit `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`, repository snapshot `735187a88975867781d40104bb2bc91d9b582fce`. Эта цепочка приведена в исходных [V7_REPORT.md](https://github.com/promaladoi/Decentrathon-6.0/blob/78a2ec998ca7b665d1aa70299aeb6f2f2a105a57/Анаис/V7_REPORT.md) и [V7_PROVENANCE.json](https://github.com/promaladoi/Decentrathon-6.0/blob/78a2ec998ca7b665d1aa70299aeb6f2f2a105a57/Анаис/V7_PROVENANCE.json).

| Вклад | Происхождение |
|---|---|
| Идея и исходная реализация run-local NF4 decode cache, source/version invalidation | Анаис; адаптируется с сохранением атрибуции |
| Точное компактное хранение frozen tied embedding/lm_head, proof kernel, управление lifetime временных tensors и подготовка launchers | Нұржігіт, выбранный v45, через Анаис V7 |
| Исправленный private training consumer, packed unpadding, фактический AMP gradient scale, packed dropout, batch metadata, публичные API/guards, frozen half-head cache и metadata snapshot | Мерей, измеренный stage2 `766d0a39fe8394be8a1cd280401af1adde6d5206`, через командную интеграцию |
| Исходная командная интеграция unpadding/packed attention, MLP, NF4, RMSNorm, RoPE и LM-head statistics | Владимир; исходные комментарии указывают `987da36e1efa90652e1a7f8c48afef50685dbc40` и последующие командные варианты |
| Арифметика и исходная проверка собственного точного NF4 decode kernel | Мерей `b7c406dea7dbdc379149910efe044dccf155d083`, атрибуция в nf4.py сохранена |
| example_swiglu_fwd_kernel / example_swiglu_bwd_kernel | Пример организаторов; код и исходная атрибуция сохраняются, в собственную долю kernels не засчитывается |
| Native GEMM, efficient/Cutlass attention, native dropout и AMP | Исходные PyTorch/bitsandbytes; не объявляются новыми собственными ядрами |

Изменения этого этапа в Мерей: ограничение исходного NF4-кэша Анаис фиксированными byte reservations; усиление source/output/stream guards и weak-reference lifetime; независимая проверка. Вариант E объединяет FP16 store/input cast с существующим собственным dropout-кernel, сохраняя исходную FP32-арифметику и native RNG. Окончательный состав выбранного решения: **вариант E: bounded attention/MLP cache768 MiB, weak-reference/source/output/stream guards, ранний raw decode заведомых misses, lossless tied storage, lifetime logits и private dropout/cast fusion**. Его commit **ce2c4e4ab547088abd17dde87997e16f798a4695**, solution tree **1791f95ed4509df573e61a0230163482caac440d**, digest **a7aa443b124b4a20e28591ef89039b6c4f3e92c3e164d12e5257af6d9e199016** подтверждены оригинальными артефактами.

Общие командные улучшения не представляются заново написанными в этом этапе. Диагностические helpers, тесты и benchmarks отделяются в отчёте от собственного производственного арифметического вклада. Сопоставление правок выполнено также после нормализации переводов строк: CRLF/LF сами по себе не означают изменение алгоритма или авторства.

Исходные LICENSE/NOTICE и комментарии не удаляются. Документация и новые доказательства сохраняются вне проверенного solution subtree. Папки Анаис и остальных участников, исходные harness/task/tests не меняются. Измеренные исходники, оригинальные result/checks и их SHA256 должны сопровождать итоговую публикацию отдельно от этой заметки.
