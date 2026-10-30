# Анаис — Qwen2.5-1.5B-Instruct на T4

Последняя проверка: **52.40 с полного запуска**, **27.02 с обучения**, **187 тестов решения + 79 тестов Soup**, **10/10 правил PASS**. Base в этой же сессии: **90.19 с**. Модель, данные и настройки кейса сохранены.

**Полный запуск за 20–30 с пока не достигнут.** 52.40 с — один холодный парный замер, не медиана. [Подробный отчёт](SPEED_REPORT.md) объясняет метрики, качество, ограничения и авторство.

## Запуск и файлы

- `solution/` — точный проверенный V6; [происхождение и SHA256](FASTEST_PROVENANCE.json).
- [result.json](speed-result.json), [checks.json](speed-checks.json) — официальные результаты.
- [Готовый Kaggle notebook](kaggle_fastest.ipynb): импортировать, выбрать **GPU T4 x2**, **Internet On**, выполнять сверху вниз. Notebook делает два парных повтора; опубликованный первичный замер — один.
- [Архив исходников](fastest-source.zip), [все эксперименты](speed-experiments.zip).
- [Предыдущая V4](previous-v4-source.zip), [предыдущий отчёт](SPEED_REPORT_PREVIOUS.md) — 55.68 с по медиане двух запусков.

Исходные `harness/`, `task/`, `tests/`, `templates/`, четыре папки этапов, LICENSE и NOTICE сохранены. [Исходные правила](STARTER_README.md). После установки окружения в Linux/CUDA перейти в эту папку:

```bash
python task/prepare_data.py
python -m pytest solution -q
python harness/run_task.py --repeats 2
python harness/check_rules.py
```

Командные ядра и интеграция — Владимир и Мерей, с атрибуцией в исходниках. Кэширование, новые тесты кэша, независимая оценка и отчёт — работа Анаис. Финальный допуск на скрытых данных определяет жюри. Прежние `candidate-*`, EVALUATION_REPORT, графики, PRESENTATION и `kaggle_optimized_candidate.ipynb` относятся к старой версии 59.34 с.
