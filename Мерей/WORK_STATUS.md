# Мерей — жұмыс күйі

Бұл — әзірлеу қалтасы, финалдық тапсыру емес.

- Бастапқы файлдар командаға берілген [репозиторийден](https://github.com/BAITC-Hacks/hack-a5a4072f-wither-ai) алынған; source commit: `fc44fd2d2bf23f624e95cef655f03fecbd87ec46`.
- `harness/`, `tests/`, `task/` бастапқы байттарымен көшірілді. Басқа қатысушылардың қалталары өзгертілмейді.
- `solution/` ішінде RMSNorm, vocabulary CE/entropy, RoPE және NF4 decode үшін 8 жеке Triton-кернел бар. SwiGLU — ұйымдастырушылардың үлгісі, жеке кернел болып саналмайды.
- Бұрынғы 5-кернелдік нұсқа T4-де 86 solution тестін және 79 Soup тестін өткізді. Толық өлшем: base 85.48 с / 14345 MiB / public loss 2.04969; solution 79.26 с / 12383 MiB / loss 2.04966.
- Бірақ сол нұсқа тек 9/10 ережеден өтті: жеке кернелдер үлесі 4.4%, қажеті кемінде 5%. Сондықтан ол финалға дайын емес.
- Қазіргі RoPE/NF4 қосылған 8-кернелдік нұсқа T4-де 165 solution тестін өткізді (1 skipped), сондай-ақ 79 Soup тестінен өтті. Толық қайталама бенчмарк пен барлық ережелерге допуск әлі расталмаған; тексеру жалғасуда. Алдыңғы нұсқаның уақыт/жад нәтижелері оған телінбейді.
- Тексеру notebook-ы: [Kaggle development](https://www.kaggle.com/code/promaladoi/soup-kaggle-development). `kaggle_notebook_dev.ipynb` осы кодтың source snapshot-ын қамтиды; бұл финалдық commit-пен жұмыс істейтін сдача notebook-ы емес.
- Өзгермеген команда README-і: `docs/TEAM_STARTER_README.md`; бастапқы Мерей README-і: `docs/MEREY_ORIGINAL_README.md`.

## Іске қосу

Kaggle-де `templates/kaggle_working_notebook.ipynb` импорттап, GPU T4 x2 және Internet қосып, Run All орындаңыз. Ол `promaladoi/Decentrathon-6.0` репозиторийінің `3c389e9f1259031123026c99057f78b3ad57b671` commit-ін алып, `/kaggle/working/repo/Мерей` ішінен жұмыс істейді. Notebook өлшеуге дейін сәйкес емес optional torchao пакетін алып тастайды, бастапқы pinned тәуелділіктер мен T4 тексеруін сақтайды, тесттер мен өзгермеген harness-ті іске қосады. Өлшеу кезінде қатар GPU жұмысын жүргізмеңіз.

Жергілікті нұсқамен жұмыс істеу үшін репозиторий түбірінен `cd Мерей` жасаңыз; `task/prepare_data.py`, `solution/` тесттері және `harness/*.py` сол қалтадан орындалады. Бастапқы платформа notebook-ы `templates/kaggle_notebook.ipynb` өзгеріссіз сақталған. `kaggle_notebook_dev.ipynb` — embedded source snapshot бар бөлек development нұсқасы.

ТЗ бойынша финалдық нәтиже платформа берген BAITC-Hacks репозиторийінде, public Kaggle notebook-та және финалдық commit-ке сілтемелермен тапсырылуы тиіс. Пайдаланушының нұсқауы бойынша әзірлеу қазір тек осы `Мерей` қалтасында жүргізіледі; платформа репозиторийіне ештеңе жазылған жоқ.
