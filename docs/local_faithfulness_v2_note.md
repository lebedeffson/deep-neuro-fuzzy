## Local Faithfulness v2 (deletion-based) status

We fixed the previous degeneracy in deletion metrics by replacing ill-conditioned least-squares head reconstruction with a stable ridge surrogate head fitted to `full_train_logits`.

Implemented in:
- `compact_stable_kafn/scripts/rebuild_local_faithfulness.py`
- mode: `--deletion-head ridge --ridge-alpha 1.0`

Generated outputs:
- `docs/tables_local_faithfulness_detail.csv`
- `docs/tables_local_faithfulness_summary.csv`
- `docs/tables_local_faithfulness_sanity_checks.csv`

Sanity outcomes now pass:
- selected-rule hashes differ across budgets/methods;
- `deletion_delta_bce` changes with budget for C-Prune (was constant before).

Important wording for manuscript:
- Current deletion metrics are **surrogate-head deletion** (stable and reproducible), not exact original-head deletion, because original `theta/bias` were not stored in v18 H-artifacts.
- For exact deletion, next artifact export must include `original_theta` and `original_bias` in rule-key order, plus reconstruction sanity check.
