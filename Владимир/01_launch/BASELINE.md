# Kaggle іске қосу есебі: Владимир

Күйі: Владимир папкасының solution-ы Kaggle T4-те өлшенді. Дерек көзі: қатысушы чатқа жіберген Kaggle output (`harness/run_task.py`, `harness/check_rules.py`, `harness/score.py`); assistant бұл GPU іске қосуын өзі орындаған жоқ. Raw JSON файлдары репозиторийге әлі берілген жоқ.

- Жауапты: Владимир, git авторы Panzershreck211.
- Notebook: `Владимир/kaggle_notebook_dev.ipynb` (репозиторийді clone жасап, Владимир папкасына өтеді).
- Нәтижелері сақталған public notebook version сілтемесі: әлі жоқ.
- Жабдық: clean-measurement check №9 Tesla T4 және бір GPU пайдаланылғанын растады.
- Бұл жюридің hidden-data финалдық тексеруі емес.

## 1. Алғашқы өз кернелдері: v1, commit `97c34a4`

NF4 dequant және RMSNorm forward/backward кернелдері. Run саны: 1 base, 1 solution.

| Көрсеткіш | Base Soup | Solution v1 |
|---|---|---|
| Толық soup train уақыты, секунд | 79.5 | 81.8 |
| Peak GPU memory, MiB | 14345 | 12383 |
| Eval loss | 2.04965 | 2.04964 |

- Speedup: 0.971x; scoring үшін 1.000x болып есептелді.
- Memory base/solution: 1.158x.
- Өз кернелдерінің үлесі: 5.2% (steps 3+); 5% шегіне өте жақын.
- PASS: 1–10. Regression: 79 passed, 14 warnings in 0.56s.
- `all checks passed on this result`: True.

## 2. v2: commit `d080f2b`, сақталған көшірмесі `saved/v2-d080f2b/`

Тек нақты токендерді есептеу (unpadding), chunked lm_head + loss, residual add пен RMSNorm бір кернелде, fp16 4-bit MLP, RoPE кернелі, компиляция warm-up. Run саны: 1 base, 1 solution.

| Көрсеткіш | Base Soup | Solution v2 |
|---|---|---|
| Толық soup train уақыты, секунд | 93.2 | 63.5 |
| Peak GPU memory, MiB | 14345 | 5641 |
| Eval loss | 2.04969 | 2.04968 |

- Speedup: 1.468x; scoring үшін 1.468x.
- Memory base/solution: 2.543x. Жады 8704 MiB, шамамен 60.7% азайды.
- Бұл сессияда base баяу болды: 93.2 с, басқа сессияларда 79–80 с. Сондықтан бұл speedup шын мәнінен жоғары шығуы мүмкін.
- Eval loss vs base: -0.000%.
- Өз кернелдерінің үлесі: 11.5% (steps 3+).
- PASS: 1–10; `all checks passed on this result`: True. Толық кесте: `03_validation/CHECKS.md`.

## 3. Ағымдағы solution: commit `410e36b`

Негізі: Мерейдің `66bcd9b` шешімі (ол Владимирдің `987da36` жұмысынан өсті: packed varlen attention, нақты жолдардағы dropout). Оған Владимирдің өзгерістері: dropout кернелі әр batch үшін қайта компиляцияланбайды, lm_head-тің fp16 салмағы бір run-да бір рет жасалады, train.log-қа кезең уақыттары жазылады. Run саны: 2 base, 2 solution.

| Run | Барлығы, с | Оқыту қадамдары, с | Қадамнан тыс, с | 1-қадам, с | Қадам медианасы, с |
|---|---|---|---|---|---|
| base_1 | 78.8 | 55.1 | 23.7 | 1.42 | 0.874 |
| base_2 | 80.1 | 58.1 | 21.9 | 1.11 | 0.930 |
| solution_1 | 52.1 | 29.0 | 23.1 | 3.58 | 0.386 |
| solution_2 | 52.5 | 29.5 | 23.0 | 3.52 | 0.392 |

- Орташа: base 79.45 с, solution 52.3 с, шамамен 1.52x.
- Бұл output-та peak memory, eval loss және 10 тексеру көрсетілмеген. Оларды осы commit үшін қайта алу керек.
- Келесі commit `13d3770` (warm-up ағынында kernel launcher дайындау) басқа сессияда 55.9 және 57.9 с көрсетті. Сессиялар арасындағы айырма 5–7 с болуы мүмкін, сондықтан бір сессиядағы A/B салыстыру әлі жасалған жоқ.

Келесі әрекет: бір Kaggle сессиясында `410e36b` және `13d3770` нұсқаларын қатар өлшеп, жылдамырағын `check_rules.py` пен `score.py` арқылы толық тексеру.
