# Article Support Tables (2026-04-27)

## Capacity-matched (Covtype full)

| Model | F1 | ROC-AUC | PR-AUC | Total rules | Active rules | Active-rule Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stacked | 0.9049 +/- 0.0043 | 0.9701 +/- 0.0023 | 0.9673 +/- 0.0026 | 106.0000 +/- 0.0000 | 46.3333 +/- 2.8674 | 0.0945 |
| hierarchical | 0.8540 +/- 0.0051 | 0.9302 +/- 0.0063 | 0.9213 +/- 0.0077 | 171.0000 +/- 0.0000 | 73.0000 +/- 3.5590 | 0.2408 |
| dffl | 0.8021 +/- 0.0075 | 0.8781 +/- 0.0090 | 0.8586 +/- 0.0113 | 107.0000 +/- 0.0000 | 18.3333 +/- 2.4944 | 0.0175 |

## Tau Sensitivity (paper_main, fuzzy-only)

| tau | Model | Mean active-rule Jaccard across datasets |
| --- | --- | ---: |
| 0.3 | shallow | 0.2959 |
| 0.3 | stacked | 0.4511 |
| 0.3 | hierarchical | 0.4381 |
| 0.3 | dffl | 0.2261 |
| 0.5 | shallow | 0.2580 |
| 0.5 | stacked | 0.1218 |
| 0.5 | hierarchical | 0.1640 |
| 0.5 | dffl | 0.2198 |
| 0.7 | shallow | 0.2765 |
| 0.7 | stacked | 1.0000 |
| 0.7 | hierarchical | 1.0000 |
| 0.7 | dffl | 0.2090 |

## Runtime Table (Covtype full)

| Model | Train time (s, mean over seeds) | Inference+eval (s, single-model seed23) |
| --- | ---: | ---: |
| stacked | 69.57 | 2.19 |
| hierarchical | 122.58 | 2.89 |
| dffl | 423.89 | 2.48 |

## Reproducibility Appendix (Hyperparameters)

| Run | Dataset | Seeds | Max epochs | Batch | Patience | Device | Rule threshold |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| covtypefull_capacity_matched_fuzzy3_3s_2026-04-27 | covtype_binary_full | 19,23,29 | 12 | 2048 | 4 | cuda | 0.5 |
| paper_main_fuzzy_tau03_3s_2026-04-27 | diabetes,linnerud_weight,breast_cancer,wine_binary,digits_binary | 19,23,29 | 8 | 256 | 3 | cuda | 0.3 |
| paper_main_fuzzy_tau05_3s_2026-04-27 | diabetes,linnerud_weight,breast_cancer,wine_binary,digits_binary | 19,23,29 | 8 | 256 | 3 | cuda | 0.5 |
| paper_main_fuzzy_tau07_3s_2026-04-27 | diabetes,linnerud_weight,breast_cancer,wine_binary,digits_binary | 19,23,29 | 8 | 256 | 3 | cuda | 0.7 |
