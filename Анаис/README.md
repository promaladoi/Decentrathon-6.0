# Анаис — ML Training & Evaluation

Свежая независимая проверка на Kaggle Tesla T4: **81.26 → 59.34 с**, **1.369× ускорение**, **10/10 PASS**.
Память: **14345 → 5641 MiB (−60.68%)**. Public eval loss: **2.04965 → 2.04964**.
Цель ≤60 секунд: **достигнута**.

## Результаты и доказательства

- [Итоговый отчёт](EVALUATION_REPORT.md) — методика, сравнение и границы результатов.
- [Подробный отчёт кандидата](CANDIDATE_REPORT.md) — все проверки и анализ training loss.
- [Таблица CSV](PERFORMANCE.csv), [производительность](performance.png), [кривые loss](training_loss.png).
- [Raw result.json](candidate-result.json), [checks.json](candidate-checks.json), [SHA256](EVIDENCE_SHA256.json).
- [Презентация PowerPoint](PRESENTATION.pptx), [текст выступления](PRESENTATION.md).
- [Starter: отдельная проверка](STARTER_REPORT.md), [starter result](starter-result.json), [starter checks](starter-checks.json).

## Воспроизведение

В Kaggle импортировать [kaggle_optimized_candidate.ipynb](kaggle_optimized_candidate.ipynb), выбрать **GPU T4 x2**, **Internet On** и выполнить ячейки сверху вниз. Проект и код кандидата встроены в notebook; GitHub-токен не нужен. Обучение использует один GPU. [Архив исходников](optimized-candidate-source.zip) и [происхождение кода](CANDIDATE_PROVENANCE.json).

Измеренный кандидат проверен одним полным парным запуском для быстрого отбора. Notebook воспроизведения настроен на два повтора: base → solution → solution → base. Starter отдельно измерен двумя повторами. Сравнивать отношения следует внутри одной группы. Время включает весь soup train и компиляцию, а не только training loop.

[Рабочий Kaggle notebook](https://www.kaggle.com/code/aneka123321/notebooke3a59f573b/edit) содержит выполненные ячейки. Доступность публичной сохранённой версии указывается отдельно; ссылка на редактор сама по себе не является доказательством публичной публикации.

## Инструменты оценки

[benchmark_report.py](benchmark_report.py) формирует REPORT.md, comparison.csv, loss_analysis.csv и metrics.json из официальных артефактов. [Четыре теста](test_report.py) пройдены локально и на Kaggle; GPU-тесты кандидата: **37 passed**. Результат 79 регрессионных тестов Soup отражён в официальной проверке 10.

```bash
python Анаис/benchmark_report.py --result runs/result.json --out Анаис/reports/latest
python -m unittest discover -s Анаис -p test_report.py -v
```

[Протокол проверки](VALIDATION.md). Скрипт не заменяет harness/check_rules.py. После изменения решения нужны новые замеры и проверки.

## Авторство и ограничения

Оптимизированные ядра: **Владимир**, сохранённый вариант `Владимир/saved/v2-d080f2b/solution`, snapshot commit `2d77f774725a18c2b16f86d76fe6ec481fe61562`. Работа Анаис: независимое тестирование, воспроизведение, оценка качества, инструменты отчётности и презентация. Harness, task и tests не менялись.

Это публичные данные и один парный замер оптимизации. Устойчивость на другом seed и скрытых данных не подтверждена. Финальный допуск определяет жюри. Финальная отправка на платформу хакатона не выполнялась.


## Полный проект из исходного архива

В эту папку добавлены исходные `harness/`, `task/`, `tests/`, `templates/`, четыре рабочие папки этапов, LICENSE и NOTICE. Их содержимое сохранено. Оригинальный README находится в [STARTER_README.md](STARTER_README.md), исходный пример решения — в `starter_solution/`.

Рабочая `solution/` содержит проверенный вариант Владимира с результатом **59,34 с и 10/10 PASS**. Её SHA256 совпадает с `candidate-checks.json`. В исходной конфигурации и измерителе изменений нет.

После клонирования репозитория перейти в папку `Анаис` и выполнять:

```bash
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python benchmark_report.py --result runs/result.json --out reports/latest
```

Окружение и закреплённые зависимости устанавливаются notebook. Для запуска через Kaggle импортировать `kaggle_optimized_candidate.ipynb` или `templates/anais_optimized.ipynb`.
