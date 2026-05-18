# Единая статья: практический план добивки

## Решение

Делаем одну объединенную работу, но с честной иерархией вкладов:

1. **Budget-Prune** — основной надежный метод структурного бюджетного сокращения.
2. **Gate-L1** — обучаемая sparse-абляция, особенно полезная на малом медицинском датасете.
3. **RuleFit** — внешний интерпретируемый baseline.
4. **Stable Budget-Prune** — stability-aware расширение; переносим в главный вклад только после H-based проверки F1/fidelity.

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
- Runtime context есть, но его нельзя подавать как строгое speed-сравнение обучения.

## Что надо добить практически

### Минимум для объединенной статьи

1. Оставить Stable Budget-Prune как subsection: "stability-aware extension".
2. Написать честно: full F1/fidelity для Stable Budget-Prune — следующий контроль, proxy уже показывает перспективу на B=400.
3. Не заявлять, что Stable Budget-Prune полностью валидирован.

Это уже можно собрать в одну сильную версию без нового долгого обучения.

### Если хотим сделать Stable Budget-Prune главным вкладом

Нужно сделать один финальный H-based контроль:

```text
dataset: covtype_binary_20000
budget: 400
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

Если это выполняется — статья становится одной Q1-oriented работой:

> Budget-Prune + Stable Budget-Prune for compact and reproducible KAFN rule dictionaries.

Если нет — Stable Budget-Prune остается как честный appendix/future work, а основная статья не ломается.

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
