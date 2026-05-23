# Cluster-Prune Runtime (Covtype)

- seeds: [19, 23, 29]
- tau: [0.75]
- budget_ref: 100

| Metric | Mean seconds |
|---|---:|
| LORO importance | 0.8265 |
| Corr matrix | 0.0184 |
| Cluster select (complete linkage) | 0.0090 |
| Total selection (LORO+corr+cluster) | 0.8539 |
| Refit+eval (3 methods, B=100) | 0.5637 |

Note: runtime is workflow context, not strict apples-to-apples training-speed comparison.