# Анаис — обучение Qwen2.5-1.5B-Instruct на T4

Последний подтверждённый результат: **55.68 с полного запуска**, медиана двух холодных запусков; base **84.93 с**, **10/10 PASS**. Память **7765 MiB**, eval loss **2.04963**.

Цели **30 с и 10–20 с пока не достигнуты**. [Итоговый отчёт и все эксперименты](SPEED_REPORT.md) содержит реальные замеры и ограничения.



## Файлы и запуск

- `solution/` — точный код последнего повторного замера; [авторство и SHA256](FASTEST_PROVENANCE.json).
- [result.json](speed-result.json), [checks.json](speed-checks.json) — официальные результаты двух парных запусков.
- [Готовый Kaggle notebook](kaggle_fastest.ipynb) — проект внутри, T4 x2, Internet On, выполнять сверху вниз.
- [Исходники](fastest-source.zip), [все эксперименты и свидетельства](speed-experiments.zip).
- `harness/`, `task/`, `tests/`, `templates/`, четыре папки этапов, LICENSE, NOTICE — исходный проект из архива.
- [Правила кейса](STARTER_README.md), исходный пример `starter_solution/`.

После клонирования перейти в папку `Анаис`:

```bash
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
python benchmark_report.py --result runs/result.json --out reports/latest
```

Ядра основаны на работе Владимира; кэширование весов, эксперименты и оценка — работа Анаис. Условия обучения и измеритель сохранены. Финальный допуск на скрытых данных определяет жюри.

## Предыдущие результаты

[Отчёт 59.34 с](EVALUATION_REPORT.md), `candidate-*`, прежние графики, PRESENTATION и `kaggle_optimized_candidate.ipynb` сохранены как история и относятся к предыдущей версии. Их значения не описывают текущую `solution/`.
