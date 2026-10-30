# Мерей — жұмыс күйі

Half-input CE қосылған шешім публичный деректердегі толық Kaggle T4 тексеруінен өтті: **10/10 ереже PASS**. Бұл — әзірлеу қалтасы; финалдық тапсыру әлі рәсімделмеген.

- Бастапқы файлдар командаға берілген [репозиторийден](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) алынған; source commit: `fc44fd2d2bf23f624e95cef655f03fecbd87ec46`. Платформа репозиторийіне жазылған жоқ.
- Кодтың жеке репозиторийдегі commit-і: [`60918ea61d91c9995799392dc350344a66e8af3e`](https://github.com/promaladoi/Decentrathon-6.0/commit/60918ea61d91c9995799392dc350344a66e8af3e).
- `harness/`, `tests/`, `task/` бастапқы Git байттарымен сақталған. Басқа қатысушылардың қалталары және бастапқы `templates/kaggle_notebook.ipynb` өзгертілмейді.
- RMSNorm, vocabulary CE/entropy, RoPE және NF4 decode үшін **8 жеке Triton-кернел** қолданылады. SwiGLU — ұйымдастырушылардың үлгісі, жеке кернел болып саналмайды. Жаңа half-input CE бұрынғы үш vocabulary кернелін пайдаланады; қосымша кернел санын жасамайды.
- Half-input CE fp16 логиттерді кернелде float32-ге көтеріп, градиентті соңында fp16-ге жазады; model/Accelerate қайтарған логиттер өзгермейді. Бұл loss үшін бөлек толық float32 көшірмесін сақтаудан босатады.
- Тексерулер: **194 passed, 1 skipped** solution тесттері; **79 passed** Soup тесттері; **10/10 harness checks PASS**.

## Нақты өлшемдер

`python harness/run_task.py --repeats 2`, реті base → solution → solution → base. Толық уақыт компиляцияны қамтиды. Summary уақыты — екі қайталаудың медианасы; жад — максимум peak.

| Көрсеткіш | Base | Solution |
|---|---:|---:|
| Толық уақыт, с | 81.38 | 72.91 |
| GPU peak, MiB, nvidia-smi | 14345 | 11181 |
| Public eval loss | 2.04964 | 2.04965 |
| Жеке кернелдер үлесі, steps 3+ | — | кемінде 8.5% |

Speedup — **1.1162×**, memory base/solution — **1.283×**. Сапа өзгерісі +0.5% шегінде. JSON-дағы `quality_delta_pct = 0.0` дөңгелектелген мән; ол loss дәл бірдей дегенді білдірмейді.

Жеке іске қосулар: base_1 **86.09 с**, solution_1 **72.96 с**, solution_2 **72.85 с**, base_2 **76.66 с**. Solution peak екі ретте де **11181 MiB**.

Алдыңғы расталған есептегі solution peak **12381 MiB** болған; екі есептің айырмасы **1200 MiB, шамамен 9.7%**. Алдыңғы base уақыты 86.68 с, қазіргісі 81.38 с болғандықтан, бөлек іске қосулардағы 81.82 → 72.91 с айырмасын half-input CE-дің жеке себептік speedup-ы деп көрсетпейміз. Негізгі speedup осы іске қосудың base/solution жұбынан есептелген.

## Дәлелдер

- [Сақталған Kaggle development нұсқасы №363823158](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363823158) әзірге **private** және embedded source snapshot қолданады.
- Snapshot SHA256: `c1b9cba94e65421fc0b204c097b725509edd32587c54db5a4b1a9d28961cc7d4`.
- Өлшенген solution SHA256: `9d65e00dce6240bdc5fa6de66d6df83c338941585e443ccda57b277efed59f52`.
- [Нәтиже](verification/half-t4-result.json), [10 ереже](verification/half-t4-checks.json), [Kaggle UI көрсеткен іске қосу логы](verification/half-t4-run.log), [source аудиті](verification/half-source-audit.json). Лог — Kaggle көрсеткен кесте жолдарының көшірмесі; метрикалар бастапқы JSON есептерімен расталған.
- Синтетикалық жеке өлшемдер: [жалпы microbenchmark](verification/half-t4-microbenchmark.json), [half-input CE microbenchmark](verification/half-ce-microbenchmark.json). Олар толық оқыту уақытынан және nvidia-smi peak-тен бөлек көрсеткіштер.
- GitHub-тағы 17 solution Python файлы өлшенген snapshot-пен жолсоңдарын CRLF/LF қалыпқа келтіргенде сәйкес келеді; 14-і байт бойынша бірдей. Үш файлдың GitHub-тағы көшірмесі LF, өлшенген snapshot-тағысы CRLF қолданады. Snapshot/solution хештері бүкіл GitHub-көшірменің байттық хеші ретінде көрсетілмейді. [Көшірме аудиті](verification/github-working-copy-audit.json).
- Алдыңғы сәтті №363806584 нұсқасы мен оның бастапқы есептері тарихи дәлел ретінде сақталған. RoPE/RMSNorm/NF4 tuning кандидаттары осы жаңа шешімге қабылданған жоқ.
- Толық түсіндірме: [README](README.md), [LOCAL_VERIFICATION](LOCAL_VERIFICATION.md), [SUBMISSION](SUBMISSION.md). Бастапқы команда README-і: `docs/TEAM_STARTER_README.md`; бастапқы Мерей README-і: `docs/MEREY_ORIGINAL_README.md`.

## Іске қосу

Kaggle-де `templates/kaggle_working_notebook.ipynb` импорттап, GPU T4 x2 және Internet қосып, Run All орындаңыз. Ол жеке репозиторийдің `60918ea61d91c9995799392dc350344a66e8af3e` commit-ін алып, `/kaggle/working/repo/Мерей` ішінен `--repeats 2` арқылы жұмыс істейді. Notebook өлшеуге дейін сәйкес емес optional torchao пакетін алып тастайды, бастапқы pinned тәуелділіктер мен T4 тексеруін сақтайды. Бұл бөлек clone-notebook дайындалған, бірақ өзі әлі орындалмаған; жоғарыдағы расталған нәтиже development snapshot-тан алынған. Өлшеу кезінде қатар GPU жұмысын жүргізмеңіз.

Жергілікті жұмыс үшін репозиторий түбірінен `cd Мерей` жасаңыз. Бастапқы платформалық notebook `templates/kaggle_notebook.ipynb` өзгеріссіз сақталған. Түбірдегі `kaggle_notebook_dev.ipynb` — қазіргі тексерілген №363823158 нұсқасының C1 embedded source snapshot-ы бар development notebook; сақталған Kaggle нұсқасы әлі private. Алдыңғы embedded notebook Git тарихында сақталған.

## Финалға қалғаны

ТЗ бойынша платформалық репозиторийдегі финалдық commit, сол commit-пен сәтті орындалған public Kaggle notebook, команда деректері, 3 минутқа дейінгі демо және презентация керек. Бұлар әлі дайын емес. Пайдаланушының нұсқауы бойынша әзірлеу тек осы `Мерей` қалтасында жүргізіледі. Жасырын деректер мен басқа seed тексерілмеді; финалдық допуск пен ұпайды жюри анықтайды.
