# Бірінші Kaggle іске қосу есебі

Күйі: starter training және тексерулер орындалды. Дерек көзі: қатысушы чатқа жіберген Kaggle output; assistant бұл GPU іске қосуын өзі орындаған жоқ. Raw JSON файлдары репозиторийге әлі берілген жоқ.

- Жауапты: қатысушының GitHub аккаунты NURIKZIX, Kaggle аккаунты nellzmi.
- Іске қосу күні / уақыты: output-та көрсетілмеген.
- Notebook editor: https://www.kaggle.com/code/nellzmi/notebook161b21487d/edit
- Нәтижелері сақталған public notebook version сілтемесі: әлі расталмаған.
- Notebook clone жасаған нақты commit SHA: output берілмеген, анықталмаған.
- Жабдық: clean-measurement check №9 Tesla T4 және бір GPU пайдаланылғанын растады.
- Run саны: 1 base, 1 solution; бұл жюридің hidden-data финалдық тексеруі емес.

| Көрсеткіш | Base Soup | Starter solution |
|---|---|---|
| Толық soup train уақыты, секунд | 87.87 | 91.62 |
| Peak GPU memory, MiB | 14345 | 12385 |
| Eval loss | 2.04969 | 2.04967 |

- Speedup: 0.959x; scoring үшін 1.000x болып есептелді.
- Memory base/solution: 1.158x. Жады 1960 MiB, шамамен 13.7% азайды.
- Уақыт шамамен 4.3% ұзарды; әзірге жылдамдату жоқ.
- Eval loss vs base: -0.001% (harness output бойынша).
- Warm-up: 51.96 с; негізгі уақыт кестесіне warm-up қосылмаған.
- PASS: 1, 2, 5, 6, 7, 8, 9, 10.
- FAIL: 3, 4. Себебі starter мысалының кернелдері өз кернеліміз болып есептелмейді; өз кернел үлесі 0.0%.
- Regression: 79 passed, 14 warnings in 2.56s.
- `all checks passed on this result`: False. Әзірге финалға дайын шешім емес.

Қатысушы сессиясындағы файлдар:

- `/kaggle/working/repo/runs/result.json`
- `/kaggle/working/repo/runs/checks.json`

Келесі әрекет: бұрынғы `runs/` нәтижесін сақтай отырып, `runs_profile/` ішіне профиль алу; содан кейін операция таңдап, өз Triton кернелін жазу. Профиль командалары: `02_kernel_research/PROFILE.md`.
