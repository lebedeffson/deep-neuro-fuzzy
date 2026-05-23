# Q1 Patch Closure (4 critical points)

## 1) ANFIS in main text
Closed with transfer-check assets and ready text:
- `docs/tables_anfis_transfer_main.csv`
- `docs/tables_anfis_transfer_main.md`
- `docs/anfis_transfer_main_text_snippet.md`

## 2) Theory strength
Upgraded from heuristic note to safe theorem-style motivation:
- `docs/cluster_prune_theory_note.md`

Contains four safe propositions:
- logit perturbation bound,
- Hoeffding fidelity bound,
- margin-based disagreement bound,
- correlation-to-profile-distance identity.

## 3) Local faithfulness deletion
Fixed implementation and sanity diagnostics:
- `compact_stable_kafn/scripts/rebuild_local_faithfulness.py`
- `docs/tables_local_faithfulness_detail.csv`
- `docs/tables_local_faithfulness_summary.csv`
- `docs/tables_local_faithfulness_sanity_checks.csv`

Current mode uses stable ridge surrogate head (`deletion-head=ridge`) and explicitly reports surrogate fidelity/agreement.

## 4) "Why not L1-LR" reviewer question
Added explicit positioning snippet:
- `docs/l1lr_vs_cprune_discussion_snippet.md`

Core message: Cluster-Prune is not universal best-F1, it is redundancy-aware post-training rule-vocabulary reduction with functional-group analysis.
