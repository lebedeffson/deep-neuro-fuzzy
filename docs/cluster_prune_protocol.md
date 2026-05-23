# Cluster-Prune Protocol (Canonical)

## Scope
Cluster-Prune is a redundancy-aware post-training rule selection method.
Do not claim strict Q1-level guarantees before multi-seed and multi-dataset validation.

## Core Hypothesis
Low rule-level Jaccard can be caused by interchangeable, highly correlated rules.
Therefore, selection should operate on groups of similar rules, not only on individual rules.

## Critical Method Constraints
1. LORO-BCE sign must be:
   - `I_r = BCE(z_without_r) - BCE(z_full)`
   - `I_r > 0` means rule `r` is useful.
2. If claiming minimum within-cluster correlation, use `complete` linkage (not `average`).
3. For scipy hierarchical clustering use condensed distances:
   - `Z = linkage(squareform(dist, checks=False), method="complete")`
4. Numerical hygiene:
   - replace `nan` correlations with `0`
   - clamp correlations to `[-1, 1]`
   - diagonal distance must be `0`
5. Do not compute cluster Jaccard from local cluster IDs built independently per seed.
   - For cross-seed cluster stability use global meta-clusters.

## Stage 1: Local C-Prune (per seed)
Data/config:
- dataset: `covtype_binary_20000`
- seeds: `19,23,29`
- budgets: `25,50,100`
- thresholds: `0.75,0.85,0.90`
- split for selection: validation only

Algorithm:
1. Build `H_val` (`n_val x m`).
2. Compute LORO-BCE importance `I_r`.
3. Compute `abs_corr` across rule activations.
4. `dist = 1 - abs_corr`.
5. Cluster with complete linkage and threshold `t = 1 - corr_threshold`.
6. Per cluster, choose representative with maximal `I_r`.
7. Cluster score = representative importance.
8. Select top-`B` clusters.
9. Selected rules = chosen representatives.
10. If `n_clusters < B`, keep `effective_selected_rules < B` (no fill-up in base variant).

Metrics:
- F1, ROC-AUC, PR-AUC, fidelity, agreement
- rule_jaccard
- mean/max abs pairwise corr among selected rules
- n_clusters, effective_selected_rules

## Stage 2: Global Meta-Cluster Stability
Purpose: valid cross-seed functional stability.

Algorithm:
1. Build fixed unlabeled reference set `D_ref` from train/val features.
2. For each seed/rule, compute activation signature on `D_ref`.
3. Stack all signatures (all seeds) into one matrix.
4. Cluster globally by `dist = 1 - abs_corr` (complete linkage).
5. Assign each rule a global `meta_cluster_id`.
6. Map each selected rule set to selected meta-cluster set.
7. Compute `meta_cluster_jaccard` across seed pairs.

Output columns:
- `dataset,budget,method,corr_threshold,seed_a,seed_b,rule_jaccard,meta_cluster_jaccard`

## Interpretation Rules
- Strong evidence scenario:
  - no large F1 drop,
  - substantially lower redundancy,
  - `meta_cluster_jaccard` > `rule_jaccard`.
- Do not claim strict theoretical guarantees for F1 from correlation-only arguments.
- Theory text should be positioned as controlled approximation intuition, not proof of optimality.

## Reporting Guidance
Main text:
- present Cluster-Prune as redundancy-aware diagnostic/selection extension.
- emphasize: rule instability may reflect representative swapping within stable functional groups.

Supplementary:
- threshold sensitivity (`0.75/0.85/0.90`)
- detailed per-seed tables
