# Тексеру есебі: starter

Дерек көзі: қатысушы чатқа жіберген `harness/check_rules.py` output. Assistant бұл GPU тексеруін өзі орындаған жоқ. Raw `checks.json` әлі берілген жоқ.

- Код commit SHA: `699daa9984b4a52e4f6a3334855435e150e2a8f4`; кейін notebook clone output-тан тексерілді.
- Notebook editor: https://www.kaggle.com/code/nellzmi/notebook161b21487d/edit
- Public results version: әлі расталмаған.

| № | Тексеру | Нәтиже | Output / түсіндірме |
|---|---|---|---|
| 1 | Base және solution аяқталды | PASS | 1 base және 1 solution, екеуі де аяқталды |
| 2 | Тыйым салынған кітапханалар жоқ | PASS | loaded: none; imported: none; packages changed: none |
| 3 | Өз Triton кернелдері іске қосылды | FAIL | Мысалдың example_swiglu_fwd_kernel және example_swiglu_bwd_kernel кернелдері есептелмейді |
| 4 | Өз кернелдері ≥5% training уақытын алады | FAIL | Өз кернелдерінің үлесі 0.0%, steps 3+ |
| 5 | Сол тапсырма және Soup outputs сақталған | PASS | identical within tolerance |
| 6 | Training loss базамен сәйкес | PASS | first step 0.00%, mean 0.02%; шектер 1% және 3% |
| 7 | Eval loss шегінде | PASS | base-пен салыстырғанда -0.001% |
| 8 | Уақыт пен жады шегінде | PASS | time x1.043; memory x0.863 of base |
| 9 | T4 және таза өлшеу | PASS | clean |
| 10 | Patched Soup regression тесттері | PASS | 79 passed, 14 warnings in 2.56s |

Қорытынды: SOME CHECKS FAILED; `all checks passed on this result: False`. Бұл starter үшін күтілген нәтиже. Барлық талап орындалды деп көрсетуге болмайды.

Келесі әрекет: `02_kernel_research/PROFILE.md` бойынша профиль алу; өз кернелін `solution/` ішіне жазып, қосқаннан кейін барлық 10 тексеруді қайта орындау. Дайын SwiGLU мысалын атауын ауыстырып немесе қайта жазып өз кернеліміз ретінде көрсетуге болмайды.
