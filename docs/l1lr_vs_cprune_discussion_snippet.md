## L1-LR vs Cluster-Prune (discussion snippet)

L1-LR on the full activation matrix `H` is a strong sparse predictive baseline. At `B=100`, its F1 is slightly above Cluster-Prune (`0.7791` vs `0.7779`).

This does not invalidate Cluster-Prune because the objectives differ:
- **L1-LR** optimizes predictive sparsity of a re-trained surrogate head.
- **Cluster-Prune** performs post-training vocabulary reduction with explicit redundancy control by selecting representatives of correlated functional rule groups.

Therefore Cluster-Prune should not be claimed as universally best-F1 method. Its contribution is redundancy-aware interpretability and functional-group stability analysis (`meta_cluster_jaccard`) under constrained rule vocabularies.

Practical guidance:
- maximize F1: prefer L1-LR / LORO-F1-like selectors;
- reduce redundancy and keep group-level interpretability: prefer Cluster-Prune.
