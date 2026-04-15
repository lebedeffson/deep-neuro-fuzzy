# Финальная Таблица: Quality / Complexity / Stability

- Источник данных: `docs/artifacts/ruanfis_paper5x3_report.json`
- Датасеты: diabetes, linnerud_weight, breast_cancer, wine_binary, digits_binary
- Seeds: 19, 23, 29
- Профиль DFFL: `quality_auto`

## 1) Интегральная таблица по fuzzy-моделям

| Модель | Avg Rank (quality) | Wins | Avg Total Rules | Avg Active Rules | Avg Active Rule Jaccard | Avg Decision Rule Jaccard |
| --- | --- | --- | --- | --- | --- | --- |
| Shallow Fuzzy | 3.000 | 1 | 36.00 | 19.73 | 0.2673 | 0.6022 |
| Stacked ANFIS | 2.200 | 2 | 50.00 | 21.53 | 0.1415 | 0.2517 |
| Hierarchical ANFIS | 2.000 | 2 | 74.80 | 31.73 | 0.1449 | 0.1751 |
| Deep Fuzzy Feature Learning (DFFL) | 2.800 | 0 | 96.73 | 46.33 | 0.2626 | 0.1836 |

## 2) Качество по каждому датасету (test mean ± std)

| Модель | diabetes (RMSE) | linnerud_weight (RMSE) | breast_cancer (F1) | wine_binary (F1) | digits_binary (F1) |
| --- | --- | --- | --- | --- | --- |
| Shallow Fuzzy | 0.1944 ± 0.0015 | 0.3693 ± 0.2114 | 0.9747 ± 0.0119 | 0.9432 ± 0.0188 | 0.9293 ± 0.0308 |
| Stacked ANFIS | 0.2100 ± 0.0071 | 0.3166 ± 0.1365 | 0.9795 ± 0.0056 | 0.9444 ± 0.0197 | 1.0000 ± 0.0000 |
| Hierarchical ANFIS | 0.2000 ± 0.0227 | 0.2949 ± 0.1729 | 0.9746 ± 0.0175 | 0.9577 ± 0.0340 | 0.9953 ± 0.0066 |
| Deep Fuzzy Feature Learning (DFFL) | 0.1973 ± 0.0130 | 0.3170 ± 0.1390 | 0.9472 ± 0.0126 | 0.9552 ± 0.0371 | 0.9501 ± 0.0307 |

## 3) Победитель по датасетам (по quality-метрике)

| Датасет | Метрика | Победитель | Значение |
| --- | --- | --- | --- |
| diabetes | RMSE | Shallow Fuzzy | 0.1944 |
| linnerud_weight | RMSE | Hierarchical ANFIS | 0.2949 |
| breast_cancer | F1 | Stacked ANFIS | 0.9795 |
| wine_binary | F1 | Hierarchical ANFIS | 0.9577 |
| digits_binary | F1 | Stacked ANFIS | 1.0000 |
