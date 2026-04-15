# Готовность статьи (IITI'26): актуальный статус на 2026-04-16

## 1) Что уже реализовано в коде

В репозитории реализованы и покрыты тестами три архитектурные линии:

- `Stacked ANFIS` (`src/ruanfis/stacked.py`)
- `Hierarchical ANFIS` (`src/ruanfis/hierarchical_anfis.py`)
- `Deep Fuzzy Feature Learning` (refined deep pipeline)

Также реализованы:

- единый benchmark-контур (`examples/run_real_datasets_benchmark.py`);
- multi-seed агрегация качества, сложности и устойчивости;
- экспорт отчётов в `.txt/.json/.md`;
- проверка ключевых сценариев автотестами.

Проверка:

```bash
/home/lebedeffson/Code/venv/bin/python -m pytest -q
```

## 2) Какие результаты уже финализированы

Основной пакет статьи (5 датасетов, 3 seed):

- `docs/artifacts/ruanfis_paper5x3_report.json`
- `docs/artifacts/ruanfis_paper5x3_report.txt`
- `docs/artifacts/ruanfis_paper5x3_summary.md`

Абляция профилей DFFL:

- `docs/artifacts/ruanfis_ablation_baseline_5x3_report.json`
- `docs/artifacts/ruanfis_ablation_quality_balanced_5x3_report.json`
- `docs/artifacts/dffl_profile_ablation_5x3.md`

Дополнительная верификация на крупном наборе:

- `docs/artifacts/ruanfis_california_3x_gpu_v3_report.json`
- `docs/artifacts/ruanfis_california_3x_gpu_v3_report.txt`
- `docs/artifacts/ruanfis_california_3x_gpu_v3_summary.md`

## 3) Актуальный снимок выводов

Для основного пакета `paper5x3` интегрально по quality-рангу лидирует `Hierarchical ANFIS`, а `DFFL` демонстрирует более сильную устойчивость активных правил среди глубоких вариантов.

Для крупного набора `california_housing` (`20640`, `seed=19,23,29`) внутри нейро-нечёткой группы лидирует `Stacked ANFIS`:

- `Stacked ANFIS`: `RMSE = 0.1159`
- `Hierarchical ANFIS`: `RMSE = 0.1195`
- `DFFL`: `RMSE = 0.1350`
- `Shallow Fuzzy`: `RMSE = 0.1745`

Классический ориентир на этом наборе: `random_forest_regressor`, `RMSE = 0.1062`.

## 4) Что осталось до финальной подачи

1. Свести финальные русскую и английскую версии к одному формальному объёму.
2. Проверить библиографию под шаблон IITI и окончательно заменить `*_tbd` ссылки.
3. Зафиксировать camera-ready формулировки для Abstract/Conclusion.
4. Подготовить финальный комплект: `.md + .docx + артефакты + рисунки`.

## 5) Рабочая формула позиционирования

`Stacked ANFIS` и `Hierarchical ANFIS` выступают как сильные baseline-линии по качеству, а `DFFL` остаётся методически центральной линией как архитектура интерпретируемого глубокого представления с акцентом на устойчивость структуры объяснений.
