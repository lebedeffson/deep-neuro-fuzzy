# Все последние таблицы в одном месте (2026-04-19)

Собрано из финальных артефактов репозитория.

## RUANFIS: основной пакет 5x3 (3 seeds)

Источник: `docs/artifacts/ruanfis_paper5x3_report.json`

| model | avg_rank | wins | avg_rules | avg_active_rule_jaccard | diabetes:rmse | linnerud_weight:rmse | breast_cancer:f1 | wine_binary:f1 | digits_binary:f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_shallow | 3.000 | 1 | 36.00 | 0.2673 | 0.1944 | 0.3693 | 0.9747 | 0.9432 | 0.9293 |
| ruanfis_stacked_anfis | 2.200 | 2 | 50.00 | 0.1415 | 0.2100 | 0.3166 | 0.9795 | 0.9444 | 1.0000 |
| ruanfis_hierarchical_anfis | 2.000 | 2 | 74.80 | 0.1449 | 0.2000 | 0.2949 | 0.9746 | 0.9577 | 0.9953 |
| ruanfis_refined_deep | 2.800 | 0 | 96.73 | 0.2626 | 0.1973 | 0.3170 | 0.9472 | 0.9552 | 0.9501 |

## RUANFIS: large пакет 2x3 (3 seeds)

Источник: `docs/artifacts/ruanfis_large_real_2d_3s_v1_report.json`

| model | avg_rank | wins | avg_rules | avg_active_rule_jaccard | california_housing:rmse | covtype_binary_20000:f1 |
| --- | --- | --- | --- | --- | --- | --- |
| ruanfis_shallow | 4.000 | 0 | 36.00 | 0.5432 | 0.1606 | 0.7101 |
| ruanfis_stacked_anfis | 1.000 | 2 | 50.00 | 0.2131 | 0.1122 | 0.8129 |
| ruanfis_hierarchical_anfis | 2.000 | 0 | 90.00 | 0.2561 | 0.1170 | 0.8117 |
| ruanfis_refined_deep | 3.000 | 0 | 113.00 | 0.4214 | 0.1337 | 0.7518 |

## DFFL абляция: profile=baseline (5x3, 3 seeds)

Источник: `docs/artifacts/ruanfis_ablation_baseline_5x3_report.json`

| model | avg_rank | wins | avg_rules | avg_active_rule_jaccard | diabetes:rmse | linnerud_weight:rmse | breast_cancer:f1 | wine_binary:f1 | digits_binary:f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_shallow | 2.600 | 1 | 36.00 | 0.2662 | 0.1947 | 0.3693 | 0.9794 | 0.9710 | 0.9373 |
| ruanfis_stacked_anfis | 1.800 | 1 | 50.00 | 0.1533 | 0.1930 | 0.3215 | 0.9728 | 0.9722 | 0.9907 |
| ruanfis_hierarchical_anfis | 2.400 | 2 | 75.00 | 0.1455 | 0.1908 | 0.3236 | 0.9726 | 0.9164 | 1.0000 |
| ruanfis_refined_deep | 3.200 | 1 | 71.93 | 0.1850 | 0.2029 | 0.2767 | 0.9612 | 0.9552 | 0.9232 |

## DFFL абляция: profile=quality_balanced (5x3, 3 seeds)

Источник: `docs/artifacts/ruanfis_ablation_quality_balanced_5x3_report.json`

| model | avg_rank | wins | avg_rules | avg_active_rule_jaccard | diabetes:rmse | linnerud_weight:rmse | breast_cancer:f1 | wine_binary:f1 | digits_binary:f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_shallow | 2.800 | 0 | 36.00 | 0.2602 | 0.1954 | 0.3693 | 0.9725 | 0.9710 | 0.9482 |
| ruanfis_stacked_anfis | 2.000 | 1 | 50.00 | 0.1581 | 0.1973 | 0.3104 | 0.9749 | 0.9867 | 0.9907 |
| ruanfis_hierarchical_anfis | 1.600 | 4 | 75.00 | 0.1566 | 0.1949 | 0.3032 | 0.9794 | 0.9195 | 1.0000 |
| ruanfis_refined_deep | 3.600 | 0 | 90.60 | 0.1257 | 0.2008 | 0.4439 | 0.9587 | 0.9310 | 0.9722 |

## Лучший RUANFIS vs лучший sklearn baseline (сводная)

Собрано из объединения 5x3 и large 2x3.

| dataset | metric | best_ruanfis | value | best_baseline | value | gap (baseline advantage>0) |
| --- | --- | --- | ---: | --- | ---: | ---: |
| diabetes | rmse | ruanfis_shallow | 0.1944 | linear_regression | 0.1779 | 0.0166 |
| linnerud_weight | rmse | ruanfis_hierarchical_anfis | 0.2949 | random_forest_regressor | 0.3374 | -0.0425 |
| breast_cancer | f1 | ruanfis_stacked_anfis | 0.9795 | mlp_classifier | 0.9886 | 0.0091 |
| wine_binary | f1 | ruanfis_hierarchical_anfis | 0.9577 | mlp_classifier | 1.0000 | 0.0423 |
| digits_binary | f1 | ruanfis_stacked_anfis | 1.0000 | mlp_classifier | 1.0000 | 0.0000 |
| california_housing | rmse | ruanfis_stacked_anfis | 0.1122 | hist_gradient_boosting_regressor | 0.0972 | 0.0150 |
| covtype_binary_20000 | f1 | ruanfis_stacked_anfis | 0.8129 | extra_trees_classifier | 0.8546 | 0.0417 |

## Final Full пакет (single-seed, full models)

Источник: `artifacts/benchmarks/final_full_2026-04-19/FINAL_COMPARISON.md`

## california_housing
- Metric: `rmse`
- Best overall: `hist_gradient_boosting_regressor` (sklearn) = `0.0986`
- Best RUANFIS: `ruanfis_stacked_anfis` = `0.1103`

| model | family | value |
| --- | --- | ---: |
| hist_gradient_boosting_regressor | sklearn | 0.0986 |
| extra_trees_regressor | sklearn | 0.1046 |
| random_forest_regressor | sklearn | 0.1064 |
| ruanfis_stacked_anfis | ruanfis | 0.1103 |
| ruanfis_hierarchical_anfis | ruanfis | 0.1153 |
| ruanfis_refined_deep | ruanfis | 0.1273 |
| mlp_regressor | sklearn | 0.1310 |
| linear_regression | sklearn | 0.1484 |
| ruanfis_shallow | ruanfis | 0.1557 |

## covtype_binary_20000
- Metric: `f1`
- Best overall: `extra_trees_classifier` (sklearn) = `0.8561`
- Best RUANFIS: `ruanfis_hierarchical_anfis` = `0.8269`

| model | family | value |
| --- | --- | ---: |
| extra_trees_classifier | sklearn | 0.8561 |
| random_forest_classifier | sklearn | 0.8498 |
| ruanfis_hierarchical_anfis | ruanfis | 0.8269 |
| hist_gradient_boosting_classifier | sklearn | 0.8206 |
| ruanfis_stacked_anfis | ruanfis | 0.8198 |
| mlp_classifier | sklearn | 0.7945 |
| ruanfis_refined_deep | ruanfis | 0.7671 |
| logistic_regression | sklearn | 0.7540 |
| ruanfis_shallow | ruanfis | 0.7408 |

## Оперативные малые прогоны (covtype_binary_8000, для тюнинга)

Важно: это **не основной пакет статьи**, а локальные одно/двух-сидовые прогоны на уменьшенном стенде `covtype_binary_8000`.  
Их нужно использовать как инженерный трек тюнинга, а не как финальные “paper” таблицы.

| run_label | file | dataset | model | f1 | total_rules | active_rules | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| refined_best_overall_small_runs | `covtype8000_dffl_balancedgroups_seed23.json` | covtype_binary_8000 | ruanfis_refined_deep | 0.7983 | 178.0 | 18.0 | single-seed/small-stand, best local tuning, not paper primary |
| refined_best_in_series | `tmp_after_patch4_covtype8000_refined_tunedthr.json` | covtype_binary_8000 | ruanfis_refined_deep | 0.7929 | 194.0 | 32.0 | single-seed/small-stand, not paper primary |
| one_phase_best_in_series | `tmp_small_covtype8000_after_softf1.json` | covtype_binary_8000 | ruanfis_refined_deep | 0.7844 | 224.0 | 84.0 | single-seed/small-stand, not paper primary |
| stacked_reference_same_small_stand | `tmp_covtype8000_stacked_hier_smallbudget.json` | covtype_binary_8000 | ruanfis_stacked_anfis | 0.8098 | 50.0 | 20.0 | single-seed/small-stand, not paper primary |
| refined_last_check | `tmp_small_covtype8000_refined_restored.json` | covtype_binary_8000 | ruanfis_refined_deep | 0.7749 | 194.0 | 65.0 | single-seed/small-stand, not paper primary |

## Свежий latest DFFL прогон (2026-04-19, fast sanity)

Источник:
- `artifacts/benchmarks/dffl_latest_covtype8000_2026-04-19/refined_fast/report.json`
- `artifacts/benchmarks/dffl_latest_covtype8000_2026-04-19/one_phase_fast/report.json`

Важно: это быстрый технический прогон на очень малом бюджете эпох (`pretrain=2`, `decision_pretrain=2`, `max_epochs=4`), только для проверки текущей версии кода.

| run_label | profile | one_phase | dataset | model | f1 | total_rules | active_rules |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| latest_refined_fast | baseline | false | covtype_binary_8000 | ruanfis_refined_deep | 0.5223 | 134.0 | 60.0 |
| latest_one_phase_fast | baseline | true | covtype_binary_8000 | ruanfis_refined_deep | 0.7220 | 134.0 | 55.0 |

---

# Полное описание архитектур и методологии (для текста статьи)

Ниже собран единый технический блок, который можно вставлять в рукопись как основу разделов «Метод», «Экспериментальная постановка» и «Обсуждение». Текст написан как инженерно-научное описание без привязки к конкретному рисунку, чтобы вы могли отрисовать схемы вручную в нужном стиле.

## 1) Общая постановка для всех моделей

Цель сравнения состоит не в выборе «единственного победителя», а в явной фиксации компромисса между качеством прогноза, структурной ценой модели и устойчивостью интерпретируемой внутренней структуры. Входной вектор обозначается как `x in R^d`, выход — `y_hat`. Во всех вариантах ядром остается нечёткий вывод на базе правил с параметризуемыми функциями принадлежности и нормировкой активаций.

Для правила `r` используется базовая форма активации:

`w_r = sigma(g_r) * prod_(i,j in A_r) mu_ij(z_i)`,

где `mu_ij` — степень принадлежности, `A_r` — антецеденты правила, `sigma(g_r)` — обучаемый гейт правила. Нормировка:

`w_bar_r = w_r / (sum_q w_q + eps)`.

Именно на этом общем механизме строятся все четыре архитектуры, но различаются глубина, организация блоков и роль скрытых представлений.

## 2) Архитектура 1: Shallow ANFIS

Shallow-вариант является контрольной точкой: один слой нечёткого вывода, одна база правил, финальный вывод через нормированные веса правил. Его задача — дать базовый уровень интерпретируемости и качества без глубины. Это минимально сложная конфигурация, где проще всего читать правила, но сложнее моделировать многоуровневые нелинейные взаимодействия.

Практический смысл для статьи: это нижняя опорная линия по архитектурной сложности и отправная точка для ответа на вопрос, что именно дает углубление.

## 3) Архитектура 2: Stacked ANFIS

Stacked-вариант строится как последовательная композиция нечётких блоков. Выход предыдущего блока становится входом следующего. Формально это глубина как каскад:

`x -> h^(1) -> h^(2) -> ... -> y_hat`.

Сильная сторона: часто лучшее качество на табличных задачах за счет поэтапного усложнения представления. Слабая сторона: промежуточные состояния интерпретируются хуже, чем в явно концептно-ориентированном варианте, потому что каждый слой действует как очередной предсказательный преобразователь, а не как строго определённый уровень семантических концептов.

Практический смысл для статьи: это «quality-driven deep baseline» внутри семейства RUANFIS.

## 4) Архитектура 3: Hierarchical ANFIS

Hierarchical-вариант организует входы по группам, сначала обрабатывает локальные подпространства, затем агрегирует результаты на верхнем уровне. Это глубина как иерархия локальных экспертных блоков:

`x -> {local fuzzy blocks} -> {aggregation block} -> y_hat`.

Сильные стороны: контролируемый рост числа правил относительно полного глобального перебора и стабильное качество на большинстве наборов. Слабая сторона: глобальная интерпретация полного пути решения требует аккуратного разбора межблочной агрегации.

Практический смысл для статьи: это лучший компромисс по quality внутри deep-ветки на части стендов и основной конкурент stacked.

## 5) Архитектура 4: DFFL (Refined Deep)

DFFL задаёт глубину как последовательное построение нечётких концептов, а не как простой каскад предикторов. В скрытых уровнях применяются концептные consequents нулевого порядка:

`c = sum_r w_bar_r v_r`,

где `v_r` — вектор концепта правила. Финальное решение вычисляется на последнем уровне через более выразительный решающий слой.

Ключевая идея: внутренние слои должны оставаться интерпретируемыми концептами, а не непрозрачными латентными активациями.

### Режимы обучения DFFL

Есть два режима, и их нужно явно различать в статье.

`one-phase`: после инициализации сразу единое дообучение. Это проще, быстрее и часто даёт более предсказуемое качество на малых инженерных прогонах.

`refined (two-phase)`: этапность + refinement-цикл перестройки/дообучения. Потенциально лучше для структурной устойчивости, но чувствительнее к настройкам и может деградировать по quality при недостаточном бюджете или неудачном профиле.

## 6) Что показывают текущие результаты по факту

На пакетах `5x3` и `2x3` DFFL не является лидером по quality. В среднем лидируют stacked/hierarchical, а DFFL чаще дает преимущество по устойчивости внутренней rule-структуры (`active-rule Jaccard`) ценой более высокой структурной стоимости (больше total rules).

Это означает, что корректный тезис статьи: DFFL — не «абсолютный победитель», а архитектура для сценариев, где важна воспроизводимость внутренней интерпретируемой структуры и трассировка решения по слоям.

## 7) Как правильно подать интерпретируемость (чтобы не было претензии «декларативно»)

Нужно включать не только агрегатные метрики, но и один полноценный case-level разбор на реальном объекте:

вход объекта `x*` -> топ-правила в каждом скрытом блоке -> агрегированный концептный вектор -> топ-правила решающего слоя -> вклад в итоговую вероятность/класс.

В этом блоке обязательно показываются:

числовые степени принадлежности признаков для ключевых антецедентов;
нормированные веса правил;
какие именно правила «дошли» до решения;
краткая текстовая интерпретация для инженера/эксперта.

Такой пример закрывает главный риск рецензии: «концепты объявлены, но не продемонстрированы».

## 8) Техническое задание на ваши ручные рисунки (без автогенерации)

Ниже что рисовать, чтобы фигуры были «статейные», а не технические скетчи.

Рисунок A (архитектуры): одна полоса с четырьмя панелями `shallow/stacked/hierarchical/DFFL`, одинаковая графическая грамматика, подписи «что является узлом» и «что является правилом».

Рисунок B (качество): по датасетам столбцы для основных моделей; отдельно panel для `5x3`, отдельно для `2x3`.

Рисунок C (компромисс): scatter `quality vs total_rules`, цветом `active-rule Jaccard`; это сразу показывает trade-off и убирает спор «кто лучше вообще».

Рисунок D (case-level прозрачность DFFL): схема прохождения одного объекта по слоям с топ-правилами и числовыми весами.

Рисунок E (абляция DFFL): сравнение `baseline/quality_balanced/quality_auto` по качеству и сложности.

## 9) Что просили усилить и как это закрывается в тексте

Запрос «дать обзор методологических линий, которые пытались делать символические/правиловые решения» закрывается в Related Work через отдельный абзац о нейро-символьной и нейро-нечёткой линии, где DFFL позиционируется как глубинный концептный вариант внутри rule-based семейства.

Запрос «методология без воды» закрывается структурой: постановка задачи -> модельный класс -> четыре архитектуры -> протокол обучения -> метрики -> статистический анализ -> ограничения.

Запрос «сравнение с обычным ANFIS и другими моделями» уже закрыт таблицами выше; в тексте нужно явно писать, что сравнение идёт и внутри семейства RUANFIS, и с внешними sklearn baseline.

Запрос «математика не с места в карьер» закрывается коротким вводным абзацем перед формулами: зачем именно такая форма правил, почему нужны нормировка и гейт, почему в DFFL скрытые consequents нулевого порядка.

## 10) Минимальный блок для финализации рукописи

Чтобы это стало готовым submission-пакетом высокого уровня, обязательны четыре вещи:

добавить один полноценный case-level explainability раздел;
добавить CI/значимость на ключевые сравнения;
держать честный тезис про trade-off, а не «DFFL лучший во всем»;
синхронизировать все численные значения между основным текстом и таблицами из этого файла.
