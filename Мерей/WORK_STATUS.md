# Мерей — жұмыс күйі

Жаңа DDC шешімі толық Kaggle T4 тексеруінен өтті: **10/10 ереже PASS**.
**241 passed, 1 skipped** solution; **79 passed** Soup с патчами. Финалдық тапсыру әлі рәсімделмеген.

| Показатель | База | Решение | Отношение / проверка |
| --- | --- | --- | --- |
| Полное время soup train, с | 88.74 | 78.18 | 1.1351× |
| GPU peak, MiB, nvidia-smi | 14345 | 7381 | база/решение 1.9435× |
| Public eval loss | 2.04964 | 2.04964 | JSON Δ +0.000% |
| Собственные кернелы, steps 3+ | — | 6.60% | минимум двух запусков, ≥5% |
| Автоматические правила | — | 10/10 PASS | публичные данные |

Жеке кернелдер үлесі кемінде **6.60%**, steps 3+; SwiGLU үлгісі
есепке кірмейді. Толық уақыт компиляцияны қамтиды. Шамалар осы бір
сессиядағы екі base және екі solution іске қосуынан алынған.

Кодтың commit-і: [b7c406dea7dbdc379149910efe044dccf155d083](https://github.com/promaladoi/Decentrathon-6.0/commit/b7c406dea7dbdc379149910efe044dccf155d083). [Kaggle №363837033](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363837033)
әзірге private және embedded DDC snapshot қолданады.
Snapshot: `ddc7740aa180e81cbdafa886d4fa44c1698cd2f79ae1cc45e63dbd45bedb033b`. Solution: `698b33301c9a191f3b8bfadc502a0c15d7a08870579c465cfed119a06a2d4d77`.
Все 32 solution-файла рабочего commit побайтно сверены с измеренным snapshot; measured solution SHA256 совпадает.

Private training metrics FP16 storage, batched rewind, strided accuracy және
narrow frozen NF4 MLP қосылған. Head recomputation және compressed-only
SwiGLU training-те қосылмайды. Публичный model outputs FP32 күйінде қалады.
Барлық 15 harness/tests/task файлдары өзгеріссіз, басқа қатысушылардың
қалталары мен платформалық репозиторийге жазылған жоқ.

Іске қосу: [working notebook](templates/kaggle_working_notebook.ipynb),
GPU T4 x2 + Internet + Run All. Ол `b7c406dea7dbdc379149910efe044dccf155d083` commit-ін алып,
`/kaggle/working/repo/Мерей` ішінен measured SHA256-ны тексеріп орындайды.
Бұл clone notebook өзі әлі орындалған жоқ; дәлел — толық embedded run.
Жергілікті репозиторий түбірінен `cd Мерей` жасаңыз.

Историческая v9 №363823158 имела solution `wall_s=72.91`
и peak `11181 MiB`; текущие значения — `78.18`
и `7381 MiB`. Это **разные Kaggle-сессии**, не парный
v9/new A/B. Разница не приписывается одному изменению; текущий speedup
вычислен неизменённым harness относительно базы той же новой сессии.


Дәлелдер: [result](verification/t4-result.json), [checks](verification/t4-checks.json), [Kaggle UI log](verification/t4-run.log), [source audit](verification/source-audit.json). [Толық түсіндірме](README.md),
[verification](LOCAL_VERIFICATION.md), [submission](SUBMISSION.md).
Финалға: платформалық final commit, public saved notebook, команда деректері,
3 минутқа дейінгі демо және презентация. Hidden data мен басқа seed
тексерілмеді; соңғы допуск пен ұпай бөлек анықталады.
