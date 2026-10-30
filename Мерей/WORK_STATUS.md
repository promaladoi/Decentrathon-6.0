# Мерей — жұмыс күйі

Шешім публичный деректердегі толық Kaggle T4 тексеруінен өтті. Бұл — әзірлеу қалтасы; финалдық тапсыру әлі рәсімделмеген.

- Бастапқы файлдар командаға берілген [репозиторийден](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) алынған; source commit: `fc44fd2d2bf23f624e95cef655f03fecbd87ec46`.
- `harness/`, `tests/`, `task/` бастапқы байттарымен сақталған. Басқа қатысушылардың қалталары өзгертілмейді. Платформа репозиторийіне ештеңе жазылған жоқ.
- Кодтың жеке репозиторийдегі commit-і: [`3c389e9f1259031123026c99057f78b3ad57b671`](https://github.com/promaladoi/Decentrathon-6.0/commit/3c389e9f1259031123026c99057f78b3ad57b671).
- `solution/` ішінде RMSNorm, vocabulary CE/entropy, RoPE және NF4 decode үшін 8 жеке Triton-кернел бар. SwiGLU — ұйымдастырушылардың үлгісі, жеке кернел болып саналмайды.
- Қазіргі 8-кернелдік нұсқа: **165 passed, 1 skipped** solution тесттері; **79 passed** Soup тесттері; **10/10 harness checks PASS**.
- `--repeats 2` бойынша base → solution → solution → base: толық уақыт **86.68 → 81.82 с**, peak GPU **14345 → 12381 MiB**, speedup **1.0594×**, memory ratio **1.1586×**. Уақыт — екі қайталаудың медианасы; жад — максимум peak.
- Public eval loss **2.04963 → 2.04969**, айырма **+0.003%**, рұқсат етілгені +0.5%. Жеке кернелдер үлесі екі solution іске қосудың минимумында **7.3%**, талап ≥5%.
- Бұрынғы 5-кернелдік нұсқа 9/10 ережеден өткен, үлесі 4.4% болған. Ол тарихи нәтиже; қазіргі қорытынды оның орнына берілген.
- [Сақталған Kaggle development нұсқасы №363806584](https://www.kaggle.com/code/promaladoi/soup-kaggle-development?scriptVersionId=363806584) әзірге **private** және embedded source snapshot қолданады. Source snapshot SHA256: `eaa4e00672e0f2133bcbb1452ae73652c8e9dbaa3c1dc89b280725d0e9f80308`.
- Дәлелдер: [result.json](verification/t4-result.json), [checks.json](verification/t4-checks.json), [microbenchmark](verification/t4-microbenchmark-final.json), [іске қосу логы](verification/kaggle-t4-eight-kernel-run.log). Microbenchmark — толық уақыттан бөлек синтетикалық өлшем.
- GitHub-көшірмедегі 14 solution Python файлы өлшенген snapshot-пен жолсоңдарын CRLF/LF қалыпқа келтіргенде сәйкес келеді; 11-і байт бойынша бірдей. Үш файлдың GitHub-тағы көшірмесі LF, өлшенген snapshot-тағысы CRLF қолданады. Snapshot хеші бүкіл GitHub-көшірменің байттық хеші ретінде көрсетілмейді. [Көшірме аудиті](verification/github-working-copy-audit.json).
- Толық түсіндірме: [README](README.md), [LOCAL_VERIFICATION](LOCAL_VERIFICATION.md), [SUBMISSION](SUBMISSION.md). Команданың бастапқы README-і: `docs/TEAM_STARTER_README.md`; бастапқы Мерей README-і: `docs/MEREY_ORIGINAL_README.md`.

## Іске қосу

Kaggle-де `templates/kaggle_working_notebook.ipynb` импорттап, GPU T4 x2 және Internet қосып, Run All орындаңыз. Ол `promaladoi/Decentrathon-6.0` репозиторийінің `3c389e9f1259031123026c99057f78b3ad57b671` commit-ін алып, `/kaggle/working/repo/Мерей` ішінен жұмыс істейді. Notebook өлшеуге дейін сәйкес емес optional torchao пакетін алып тастайды, бастапқы pinned тәуелділіктер мен T4 тексеруін сақтайды. Бұл бөлек clone-notebook дайындалған, бірақ өзі әлі орындалмаған; жоғарыдағы расталған нәтиже development snapshot-тан алынған. Өлшеу кезінде қатар GPU жұмысын жүргізмеңіз.

Жергілікті жұмыс үшін репозиторий түбірінен `cd Мерей` жасаңыз. Дәл қайталама өлшем үшін `python harness/run_task.py --repeats 2` қолданыңыз. Бастапқы платформа notebook-ы `templates/kaggle_notebook.ipynb` өзгеріссіз сақталған. `kaggle_notebook_dev.ipynb` — embedded source snapshot бар бөлек development нұсқасы.

## Финалға қалғаны

ТЗ бойынша платформалық репозиторийдегі финалдық commit, сол commit-пен сәтті орындалған public Kaggle notebook, команда деректері, 3 минутқа дейінгі демо және презентация керек. Бұлар әлі дайын емес. Пайдаланушының нұсқауы бойынша әзірлеу тек осы `Мерей` қалтасында жүргізіледі. Жасырын деректер мен басқа seed тексерілмеді; финалдық допуск пен ұпайды жюри анықтайды.
