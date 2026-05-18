# Единая статья: практический план добивки

## Решение

Делаем одну объединенную работу, но с честной иерархией вкладов:

1. **Budget-Prune** — основной надежный метод структурного бюджетного сокращения.
2. **Gate-L1** — обучаемая sparse-абляция, особенно полезная на малом медицинском датасете.
3. **RuleFit** — внешний интерпретируемый baseline.
4. **Stable Budget-Prune** — stability-aware расширение; H-based B=400 проверка пройдена с малой потерей F1.

Так мы не выбрасываем текущие эксперименты и не строим Q1-claim на одном proxy.

## Что уже можно брать в статью

### Основная таблица методов

Файл:

```bash
docs/unified_main_methods_table.csv
```

Ключевые строки:

- Covtype B=400: Budget-Prune `0.8044`, Gate-L1 `0.7939`, Random-B `0.6756`.
- Breast Cancer B=400: Gate-L1 `0.9700`, Budget-Prune `0.9642`, RuleFit `0.9578`.
- SUSY сейчас содержит только RuleFit rows; использовать как внешний baseline/runtime context, не как победу KAFN.

### Контрольные проверки

Файл:

```bash
docs/unified_control_checks_table.csv
```

Что закрывает:

- LR top-K слабее Budget-Prune на Covtype: delta F1 `+0.0181`, `+0.0246`, `+0.0256`.
- Stable Budget-Prune proxy улучшает Jaccard:
  - B=100: `0.2887 -> 0.5076`, retention `0.6997`.
  - B=200: `0.3939 -> 0.6284`, retention `0.7715`.
  - B=400: `0.6384 -> 0.8296`, retention `0.9284`.
- H-based B=400 validation:
  - Budget-Prune H-LR F1: `0.7771`.
  - Stable Budget-Prune H-LR F1: `0.7739`.
  - F1 drop: `0.0032`.
  - fidelity delta: `+0.0116`.
  - agreement to full KAFN: `0.8562`.
  - Jaccard to Budget-Prune subset: `0.7464`.
- Runtime context есть, но его нельзя подавать как строгое speed-сравнение обучения.

## Что надо добить практически

### Минимум для объединенной статьи

1. Оставить Stable Budget-Prune как subsection: "stability-aware extension".
2. Написать честно: Stable Budget-Prune на B=400 сохраняет качество почти без потери, но fidelity немного хуже.
3. Не заявлять, что Stable Budget-Prune лучше по всем метрикам.

Это уже можно собрать в одну сильную версию без нового долгого обучения.

### Если хотим сделать Stable Budget-Prune главным вкладом

Минимальный H-based контроль уже сделан для Covtype B=400. Чтобы сделать вклад ещё сильнее, можно расширить:

```text
dataset: covtype_binary_20000
budget: 100, 200, 400
seeds: 19, 23, 29
methods:
  - Budget-Prune
  - Stable Budget-Prune
metrics:
  - F1
  - ROC-AUC
  - PR-AUC
  - fidelity to full KAFN probability
  - prediction agreement with full KAFN
  - selected-subset Jaccard
```

Критерий продвижения Stable Budget-Prune в главный вклад:

```text
stable_jaccard > budget_prune_jaccard
and F1 drop <= 0.01
and fidelity gap is small
```

Для B=400 критерий по F1 выполняется (`drop=0.0032`), но fidelity чуть хуже. Поэтому формулировка:

> Budget-Prune with a stability-aware extension for compact and reproducible KAFN rule dictionaries.

Не формулировать как "Stable всегда лучше Budget-Prune".

## Как пересобрать unified tables

```bash
.venv_run/bin/python compact_stable_kafn/scripts/build_unified_q1_tables.py --docs-dir docs
```

Выходы:

```text
docs/unified_main_methods_table.csv
docs/unified_control_checks_table.csv
docs/unified_q1_result_package.md
```

## Что не делать

- Не смешивать SUSY RuleFit с отсутствующими SUSY KAFN quality rows как будто это полная сравнительная таблица.
- Не писать "KAFN быстрее RuleFit" без уточнения, что это разные pipeline.
- Не подавать Stable Budget-Prune как полностью доказанный метод до H-based проверки.
- Не раздувать сейчас новые методы: EBM, OMP, Group Lasso, Stability Selection.

## Финальная структура статьи

1. Introduction: budgeted interpretability for Routed KAFN.
2. Method: active rule dictionary and Budget-Prune.
3. Stability extension: Stable Budget-Prune.
4. Baselines: Random-B, Gate-L1, LR top-K, RuleFit.
5. Datasets: Covtype20k, Breast Cancer, SUSY RuleFit context.
6. Results:
   - main quality table;
   - LR top-K control;
   - subset stability;
   - RuleFit comparison;
   - local explanation.
7. Limitations:
   - Stable Budget-Prune currently proxy-validated unless H-based check is completed;
   - SUSY comparison needs matched KAFN quality rows for full claim.
