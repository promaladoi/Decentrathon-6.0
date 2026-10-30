<!-- latest-verified -->
# Latest source-verified paired measurements

| Version | Code | Base s | Solution s | Peak MiB | Speedup | Eval delta | Checks |
|---|---|---:|---:|---:|---:|---:|---:|
| v16 | `4a5cfd2` | 85.07 | 60.41 | 5601 | 1.4082x | +0.001% | 10/10 |
| v17r1 | `7c5f6d8` | 84.28 | 54.19 | 5601 | 1.5553x | -0.002% | 10/10 |
| v18r1 | `d01b2d8` | 80.94 | 55.61 | 4701 | 1.4555x | +0.000% | 10/10 |
| v20r1 | `22ff279` | 81.23 | 55.81 | 4467 | 1.4555x | -0.000% | 10/10 |
| v21 | `a67816b` | 80.84 | 55.48 | 4467 | 1.4571x | -0.001% | 10/10 |
| v22 | `7a8b38f` | 80.70 | 54.64 | 4467 | 1.4769x | +0.001% | 10/10 |
| v23 | `d8ac022` | 80.68 | 52.76 | 6747 | 1.5292x | -0.003% | 10/10 |
| v24 | `053c9c4` | 78.99 | 50.95 | 6651 | 1.5503x | -0.001% | 10/10 |
| v25 | `fd296f4` | 80.87 | 48.97 | 6651 | 1.6514x | +0.000% | 10/10 |
| v26 | `79643de` | 80.84 | 46.38 | 6651 | 1.7430x | -0.000% | 10/10 |
| v27 | `7efba21` | 80.79 | 45.83 | 7153 | 1.7628x | +0.002% | 10/10 |
| v28 | `4aaaade` | 80.75 | 44.43 | 7153 | 1.8175x | -0.000% | 10/10 |

20–30-second target remains unmet. Raw reports and proof are in `reports/`.
<!-- /latest-verified -->

# Команда нұсқаларын салыстыру

Қатысушылар бірдей бекітілген task/model/data/LoRA/optimizer баптауларын және
ұйымдастырушылардың өзгермеген harness файлын пайдаланады. Әр нұсқа платформаның
репозиторийіндегі жеке branch пен нақты code commit SHA арқылы белгіленеді.

| Нұсқа | Code commit | 10 check | Speedup | Memory ratio | Eval loss delta | Ескерту |
|---|---|---|---|---|---|---|
| Starter | 699daa9984b4a52e4f6a3334855435e150e2a8f4 | 8 PASS; 3,4 FAIL | 0.959x | 1.158x | -0.001% | Қатысушы берген output; кейін notebook clone output-тан SHA тексерілді |
| Codex / Нұржігіт v1 | d4d4231e019b8ef6f282076ae7e42972ff37bede | 9 PASS; 4 FAIL | 0.972x | 1.158x | +0.000% | Өз кернелдерінің үлесі 3.3%, финалға жарамайды |
| Codex / Нұржігіт v2 | 6c7370c09ce303949d015ac85b28935e466cd70e | 10 PASS | 0.998x | 1.158x | +0.003% | 2 paired run; өз кернелдері 5.1%, шекке жақын |
| Codex / Нұржігіт v3 | 8f82109fdec5f239c0cd4df21ed27fded849cd18 | 10 PASS | 1.033x | 1.158x | +0.000% | 2 paired run; өз кернелдері 7.1%; speed gain <5%, scoring-де тең |
| Original public repo, direct verification | c7c2fcd15972ce8b898d8d3ff732ee192c817ceb | 8 PASS; 3,4 FAIL | 0.916x | 1.158x | +0.003% | 1 paired run; өзгертілмеген ұйымдастырушы мысалы |
| Codex / Нұржігіт v4 experiment | 12b9f58f32e70fc1db6ca47b7860eb9870a9ffd8 | 10 PASS | 1.036x | 1.158x | +0.000% | 2 paired run; 37 kernel tests; өз кернелдері 7.1%; v3-тен айқын ұтыс дәлелденбеді |
| Codex v5 | 58c7378 | 10 PASS | 1.0636x | 1.3867x | -0.000% | FP16 MLP intermediates; 2 paired runs |
| Codex v6 | f92ebf2 | 10 PASS | 1.1032x | 1.3867x | +0.000% | NF4 widening; 37 tests; 2 paired runs |
| Codex v7 | a044007 | 10 PASS | 0.9995x | 1.3869x | -0.002% | LoRA fusion rejected: slower; 2 paired runs |
| Codex v8 | 7435a92 | 10 PASS | 1.0144x | 1.5690x | +0.002% | Direct causal loss; slow LoRA still included; 2 paired runs |
| Codex v9 | 57182a2 | 10 PASS | 1.1075x | 1.5652x | -0.001% | Earlier candidate: 41 tests, own kernels 7.3%; 2 paired runs |
| Codex v10 | e19b464 | 10 PASS | 1.1156x | 1.9425x | -0.001% | Earlier validated candidate: 54 tests, own kernels 7.0%; 2 paired runs |
| Codex v11 | 9e104d4 | No complete report | — | — | — | Superseded by v12; rerun hit concurrent-process OOM, no accepted measurement |
| Codex v12 | bd2b49a | 10 PASS | 1.1608x | 1.9425x | -0.000% | Latest complete report; 64 kernel tests, own share 8.1%; 2 paired runs |
| Combined v13 | 13eaa8a | 10 PASS (participant console) | 1.398x | 2.490x | -0.001% | 106 tests; own share 11.3%; 2 paired runs; raw archive pending download |
| Combined v14 | fd5e482 | 4 unit tests FAIL | — | — | — | Exact dropout mask mismatch; no benchmark; raw console log saved |
| v14 repair | 3016d55 | Pending | — | — | — | Original dropout operator; 6 exact dropout tests before training |
| Combined v15 | 389f037 | 6 unit tests FAIL | — | — | — | No training; downloaded source matches Git; row-stride bug identified |
| Combined v16 | 4a5cfd2 | 10 PASS | 1.4082x | 2.5611x | +0.001% | Fully archived; 115 tests, own share 12.9%; 2 paired runs |
| Combined v17 | 7c5f6d8 | Pending | — | — | — | 104 CPU layout cases; 23 new GPU cases, paired-byte decoder |
| Combined v18 | d01b2d8 | Pending | — | — | — | Lossless frozen storage; 15 CPU tests / 5 CUDA skips, 158 full GPU cases |
| Combined v19 | 1e2fe4b | Not independently measured | — | — | — | Direct native mask intermediate; included in v20 |
| Combined v20 | 22ff279 | Pending | — | — | — | Packed real-token mask; 124 CPU layout cases, 174 full GPU cases |
| v17r1 | 7c5f6d8 | 10 PASS | 1.5553x | 2.5611x | -0.002% | Fully archived; 138 tests; 84.28 / 54.19 s |
| v18r1 | d01b2d8 | 10 PASS | 1.4555x | 3.0515x | +0.000% | Fully archived; 158 tests; 80.94 / 55.61 s |
| v20r1 | 22ff279 | 10 PASS | 1.4555x | 3.2113x | -0.000% | Fully archived; 174 tests; 81.23 / 55.81 s |
| v21 | a67816b | 10 PASS (console) | ~1.457x | ~3.211x | -0.001% | Raw archive collection pending |
| v22 | 7a8b38f | Running | — | — | — | CLI startup; 7 CPU parser contracts passed |
| v23 | d8ac022 | Pending | — | — | — | Frozen weight reuse; 14 CPU contracts / 4 CUDA skips |
| Анаис | әлі берілмеген | әлі берілмеген | жоқ | жоқ | жоқ | |
| Владимир (user-reported) | main/d080f2b component | Full checks not supplied | 1.468x | 2.543x | -0.0% | User summary: 93.2 / 63.5 s, 14345 / 5641 MiB; not independently validated here |
| Мерей | әлі берілмеген | әлі берілмеген | жоқ | жоқ | жоқ | |

Әр нұсқа үшін жинау керек:

1. Осы ресми репозиторийдегі нақты code commit сілтемесі.
2. Сол commit-ке COMMIT мәні бекітілген, нәтижелері сақталған Kaggle notebook нұсқасы.
3. Толық `result.json` және сол нәтижеге жасалған `checks.json`.
4. Қай операция өзгерді, өз кернелдері қай файлда, алынған идеялардың көздері.

Алдымен барлық 10 тексеруден өткен нұсқалар қаралады. Кейін әр нұсқаның өз
сессиясындағы base-пен қатынастары салыстырылады; әртүрлі сессиядағы секундтарды
тікелей салыстырып жеңімпаз таңдау дұрыс емес. Жылдамдық пен жады ұтысы
екі түрлі нұсқада жақсы болуы мүмкін; нақты жюри балдары басқа командалардың
ең жақсы нәтижелеріне де тәуелді.

Бірнеше нұсқаны біріктірсек, біріктірілген код жаңа commit болып белгіленіп,
барлық тест, paired benchmark және rule checks қайта орындалады. Соңғы
тапсыруға командадан бір тексерілген нұсқа беріледі.

## Есептерді автоматты салыстыру

Әр қатысушының өзгертілмеген `result.json` және `checks.json` файлдарын жеке
папкаға қойып, мынадай командамен салыстыруға болады:

```bash
python 03_validation/compare_variants.py Codex=reports/codex Anais=reports/anais Vladimir=reports/vladimir Merey=reports/merey
```

Скрипт нәтижелерді қатар көрсетеді, 10 тексерудің өткенін және checks есебіндегі
SHA256 нәтижелер файлының нақты байттарымен сәйкес екенін тексереді. Бұл кодтың
жюри тарапынан қабылданатынына немесе hidden-data нәтижесіне кепіл емес.
