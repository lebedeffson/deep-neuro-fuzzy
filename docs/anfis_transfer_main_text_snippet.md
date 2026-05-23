## Transferability Check on ANFIS (main-text snippet)

Cluster-Prune was additionally applied to a standalone ANFIS model on Breast Cancer as a transfer check (not a full architecture benchmark). The purpose is to test whether the method depends on Routed KAFN internals or only on generic post-training objects: rule activations, rule-level importance, and validation split.

Using the same 3-seed protocol and `B=100`:
- Budget-Prune: `F1=0.9793`, `rule_jaccard=0.3487`, `meta_jaccard=0.7731`, `mean_corr=0.7221`
- LORO-BCE: `F1=0.9793`, `rule_jaccard=0.3471`, `meta_jaccard=0.7593`, `mean_corr=0.7590`
- C-Prune (`tau=0.90`): `F1=0.9817`, `rule_jaccard=0.1129`, `meta_jaccard=0.6991`, `mean_corr=0.4830`, `effective_rules=39`

This confirms portability of Cluster-Prune as a post-training reduction routine for another neuro-fuzzy architecture. Detailed ANFIS tables are provided in Supplementary (`docs/tables_anfis_transfer_main.*` and full ANFIS result files).
