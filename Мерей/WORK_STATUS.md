# Мерей — состояние работы

## Последний runtime: 29419f43

[Выбранный source](https://github.com/promaladoi/Decentrathon-6.0/commit/29419f43d9b9173d1ba5d6824a31589ad3b84df7):
**339 solution tests + 79 Soup tests PASS** на T4. Текущий Мерей и новая
версия в одной серии: **53.38 → 49.17 с** median whole command (−7.9%),
**5275 → 5175 MiB** maximum peak. Старые runs 52.30 / 54.46 с; новые
48.50 / 49.84 с. Timed steps 3–64: **25.707 → 25.737 с** — это не
доказательство ускорения steady-state training.

[Изменения и точные границы A/B](solution/OPTIMIZATION_NOTES.md) ·
[Reproduction notebook, pinned 29419f43](solution/optimization_validation.ipynb) ·
[Official admission/quality proof](verification/codex_optimization/final-checks.json).
Допуск и качество проверяются отдельно исходным harness; A/B их не
заменяет. Документация может иметь более поздний docs-only publication
commit при тех же production `.py`; окончательный checks hash относится
к опубликованным bytes. Платформенная сдача остаётся отдельным этапом.

## Прежний checkpoint — сохранённая история

Полный Kaggle T4 development checkpoint прошёл **10/10 automatic checks**.
**181 passed, 0 skipped** solution; **79 passed** Soup с патчами. Финальная платформенная сдача ещё не оформлена.

| Показатель | Базовый Soup | Решение | Отношение / проверка |
| --- | --- | --- | --- |
| Полное время soup train, с | 82.19 | 56.09 | 1.4653× |
| GPU peak, MiB (nvidia-smi) | 14345 | 5275 | база/решение 2.7194× |
| Public eval loss | 2.04962 | 2.04965 | JSON Δ +0.001% |
| Собственные kernels, steps 3+ | — | 15.08% | минимум двух запусков, ≥5% |
| Автоматические правила | — | 10/10 PASS | публичные данные |

| Одна сессия, два запуска каждого | Владимир pinned | Новое решение | Отношение reference/candidate |
| --- | --- | --- | --- |
| Полное время, с (median) | 59.655 | 56.095 | 1.0635× |
| GPU peak, MiB (maximum) | 5641 | 5275 | 1.0694× |
| Training loop, с (median) | 30.79 | 31.78 | диагностика |

A/B — performance diagnostic относительно pinned командного кода Владимира.
Official score сравнивает только native Soup и новое решение.
Собственная активная доля — минимум **15.08%**, steps 3+;
два organizer SwiGLU kernel не учитываются.

Saved Kaggle full development version: [№363884710](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363884710), private.
Snapshot ZIP SHA256: `1916a1eab83d5ad80877937c6dc1203b63c10e7a151a64d49ca9beb89401403c`.
Measured solution SHA256: `60216a16a1f993117f90e6028e93e4096fbacc4fb21afefbccd3f7f3a2e3a7f0`.
Result SHA256: `53b72beee7141c1b18cb2b7d6b6ea60f796ecbdd23dbb8b05c1b945b9c8b8a7b`; совпадает с checks и audit.
Рабочий code commit: [66bcd9b6585b027561a158448d87c2699f975931](https://github.com/promaladoi/Decentrathon-6.0/commit/66bcd9b6585b027561a158448d87c2699f975931).
Все 27 solution-файла этого commit
побайтно сверены с измеренным source snapshot по GitHub readback receipt.
`kaggle_notebook_dev.ipynb` — точная побайтовая копия входного notebook
этой saved версии, а не новый исполненный clone notebook.


Авторство командной основы Владимира сохранено. В candidate включены
private scope с публичным Tensor fallback, actual-scale head backward,
direct packed RoPE/native attention, real-row native-RNG dropout на больших
размерах и numerical contract fixes. Все protected files без изменений.

Воспроизведение: [pinned working notebook](templates/kaggle_working_notebook.ipynb),
GPU T4 x2 + Internet + Run All. Он закрепляет code commit и проверяет
solution SHA256 до тестов из папки `Мерей`. Этот clone notebook ещё не
исполнен; текущие доказательства — saved embedded development run.

[result](verification/final-result.json), [checks](verification/final-checks.json), [full Kaggle log](verification/final-run.log), [A/B](verification/final-comparison.json), [source audit](verification/source-audit.json). [README](README.md), [verification](LOCAL_VERIFICATION.md),
[submission](SUBMISSION.md).

Рабочее место — [личная папка Мерей](https://github.com/promaladoi/Decentrathon-6.0/tree/main/%D0%9C%D0%B5%D1%80%D0%B5%D0%B9).
Финальная платформенная сдача ещё не оформлена. По указанию пользователя
[платформенный репозиторий](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) и папки других участников не
изменяются. Для сдачи остаются final commit в репозитории платформы,
публичный saved notebook из исходного шаблона с этим commit и успешным
Run All, данные команды, демо до 3 минут и презентация. Скрытые строки и
другие seed не проверены; окончательный допуск и баллы определяет жюри.

