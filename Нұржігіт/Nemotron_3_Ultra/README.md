# Рабочая копия для NVIDIA Nemotron 3 Ultra

Модель-помощник анализирует и улучшает код **Qwen2.5-1.5B-Instruct на T4**.
Начальный код — проверенный командный v45: **45.03 s / 4135 MiB / 10/10**.
Цель нового кандидата: <=27 s полной команды, <=7151 MiB в каждом запуске,
неувеличение reported eval loss и исходные 10/10. Цель пока не достигнута.

## Передать Nemotron

1. Выбери NVIDIA Nemotron 3 Ultra в своём интерфейсе.
2. Прикрепи ZIP этой рабочей папки или дай coding-агенту доступ к этой папке.
3. Вставь целиком [PROMPT_RU.txt](PROMPT_RU.txt).
4. Полученный вариант и отчёт сохраняй здесь; команда затем сравнит кандидатов.

В обычном чате модель возвращает код/patch. Файлы и обучение меняются только
при наличии соответствующих инструментов или после применения patch в проекте.

## Состав

- `solution/`: все 48 Python-файлов точного измеренного v45.
- `harness/`, `task/`, `tests/`: неизменённые исходные файлы организаторов.
- `evidence/v45/`: оригинальные result/checks, per-run metrics/logs и proof.
- `CLONE_MANIFEST.json`: источник, контрольные суммы и границы проверенного baseline.
- `04_submission/run_candidate.py`: тесты, две base/solution пары, 10 checks,
  дополнительные пользовательские ограничения и архив.
- `03_validation/user_constraints.py`: предел 7151 MiB, loss и цель 27 s.

Самостоятельного GPU-результата у новой папки ещё нет. Числа v45 выше относятся
к сохранённому исходному измерению. Авторство командных компонентов сохранено
в [TEAM_INTEGRATION.md](TEAM_INTEGRATION.md), LICENSE и NOTICE.

## Проверить в Kaggle

GPU T4 x2, Internet On; измерение использует одну T4. В установленной Kaggle
среде с PyTorch 2.11 сначала установи зависимости из `requirements-kaggle.txt`.
Открой корень этой рабочей папки и выполни:

```bash
python -m pip install -q -r requirements-kaggle.txt
python task/prepare_data.py
CUDA_VISIBLE_DEVICES=0 python 04_submission/run_candidate.py --version v52_nemotron
```

Из Kaggle code cell можно вызывать shell-команды с `!`; `%cd` должен указывать
именно на `Нұржігіт/Nemotron_3_Ultra` в клонированном репозитории. Runner
сохранит `/kaggle/working/codex_v52_nemotron.zip`. Перед измерением обе GPU
должны быть свободны. Для каждой следующей попытки выбирай новый version.
Полное время измеряет исходный harness с чистыми compiler caches.

Baseline source: `7a3db8fd5e24bbcb58affb475007a5cf93c56f07`.
[Исходный proof](evidence/v45/collection_proof.json) ·
[Репозиторий команды](https://github.com/promaladoi/Decentrathon-6.0).
