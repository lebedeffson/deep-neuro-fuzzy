# ANFIS Transfer Check (Main-text Compact Table)

| Model | Dataset | Method | B | tau(meta) | F1 | Rule Jaccard | Meta-cluster Jaccard | Mean Pairwise Corr | Effective Rules |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ANFIS (standalone) | Breast Cancer | Budget-Prune | 100 | 0.90 | 0.9793 | 0.3487 | 0.7731 | 0.7221 | 100.0 |
| ANFIS (standalone) | Breast Cancer | LORO-BCE | 100 | 0.90 | 0.9793 | 0.3471 | 0.7593 | 0.7590 | 100.0 |
| ANFIS (standalone) | Breast Cancer | C-Prune (tau=0.90) | 100 | 0.90 | 0.9817 | 0.1129 | 0.6991 | 0.4830 | 39.0 |

Notes:
- Transfer check purpose: portability of post-training selection, not full architecture benchmark.
- These numbers are from the same 3-seed protocol as KAFN core tables.
