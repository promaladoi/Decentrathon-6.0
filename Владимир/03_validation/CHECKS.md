# Тексеру есебі: Владимир

Дерек көзі: қатысушы чатқа жіберген `harness/check_rules.py` және `harness/score.py` output. Assistant бұл GPU тексеруін өзі орындаған жоқ. Raw `checks.json` әлі берілген жоқ.

## v2: commit `d080f2b`

- Код commit SHA: `d080f2bd87e53a31ccb2b3045bad914e360490de`; көшірмесі `saved/v2-d080f2b/solution/`.
- Notebook: `Владимир/kaggle_notebook_dev.ipynb`.
- Public results version: әлі жоқ.

| № | Тексеру | Нәтиже | Output / түсіндірме |
|---|---|---|---|
| 1 | Base және solution аяқталды | PASS | 1 base және 1 solution, екеуі де аяқталды |
| 2 | Тыйым салынған кітапханалар жоқ | PASS | loaded: none; imported: none; packages changed: none |
| 3 | Өз Triton кернелдері іске қосылды | PASS | add_rmsnorm_fwd/bwd, rmsnorm_fwd/bwd (`rmsnorm.py`), nf4_dequant (`nf4.py`), lm_ce (`lmhead.py`), heads_rows, rope_pad_fwd/bwd (`unpad.py`). Мысалдың example_swiglu_fwd/bwd кернелдері есептелмейді |
| 4 | Өз кернелдері ≥5% training уақытын алады | PASS | 11.5%, steps 3+ |
| 5 | Сол тапсырма және Soup outputs сақталған | PASS | identical within tolerance |
| 6 | Training loss базамен сәйкес | PASS | first step 0.02%, mean 0.01%; шектер 1% және 3% |
| 7 | Eval loss шегінде | PASS | base-пен салыстырғанда -0.000% |
| 8 | Уақыт пен жады шегінде | PASS | time x0.681; memory x0.393 of base |
| 9 | T4 және таза өлшеу | PASS | clean |
| 10 | Patched Soup regression тесттері | PASS | 79 passed, 14 warnings in 1.13s |

Қорытынды: ALL CHECKS PASSED; `all checks passed on this result: True`. Speedup 1.468x, memory base/solution 2.543x, eval loss vs base -0.000%.

## v1: commit `97c34a4`

10/10 PASS, `all checks passed on this result: True`. Өз кернелдерінің үлесі 5.2%, 5% шегіне өте жақын. Speedup 0.971x (scoring үшін 1.000x), memory base/solution 1.158x. Сандары: `01_launch/BASELINE.md`.

## Алдыңғы solution: commit `410e36b`

Уақыт өлшенді: 2 run орташасы base 79.45 с, solution 52.3 с, шамамен 1.52x. Бұл commit үшін `check_rules.py` output-ы берілген жоқ.

## Командалық v45: commit `41bc6ec`

Осы commit-тағы `solution/` Нұржігіттің өлшенген v45 коды (`7a3db8f`)-мен байт бойынша бірдей. Нұржігіт папкасындағы тексеру: 10/10 PASS, өз кернелдерінің үлесі 14.6%, eval loss -0.001%, time x0.566, memory x0.288 of base (`Нұржігіт/03_validation/reports/codex_v45/checks.json`).

## Ағымдағы solution: v45 + түзетілген GitHub өзгерістері

`solution/` енді v45-пен байт бойынша бірдей емес: MLP backward буфері қайта пайдаланылады, ал `mlp.py` мен `nf4.py`-дағы GitHub өзгерістерінің қателері түзетілді (толығырақ: `01_launch/BASELINE.md`, 5-бөлім). 10 тексеру бұл нұсқа үшін әлі жүргізілген жоқ.

Келесі әрекет: `kaggle_notebook_dev.ipynb` арқылы Run All орындап, осы бөлімді Владимир папкасындағы жаңа output-пен толтыру.
