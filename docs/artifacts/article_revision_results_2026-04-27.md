# Article Revision Results (2026-04-27)

## Completed this cycle

- Capacity-matched comparison on `covtype_binary_full` (3 seeds, fuzzy-only):
  - run: `runs/covtypefull_capacity_matched_fuzzy3_3s_2026-04-27`
  - stacked: `F1 0.9049 +/- 0.0043`, rules `106 +/- 0`
  - hierarchical: `F1 0.8540 +/- 0.0051`, rules `171 +/- 0`
  - DFFL: `F1 0.8021 +/- 0.0075`, rules `107 +/- 0`

- Active-rule threshold sensitivity (`tau = 0.3 / 0.5 / 0.7`) on `paper_main` (3 seeds):
  - runs:
    - `runs/paper_main_fuzzy_tau03_3s_2026-04-27`
    - `runs/paper_main_fuzzy_tau05_3s_2026-04-27`
    - `runs/paper_main_fuzzy_tau07_3s_2026-04-27`
  - consolidated table: `docs/artifacts/article_support_tables_2026-04-27.md`

- Runtime profiling on `covtype_binary_full` (seed 23, one model per run):
  - runs:
    - `runs/runtime_covtypefull_stacked_capacity_s23_2026-04-27`
    - `runs/runtime_covtypefull_hierarchical_capacity_s23_2026-04-27`
    - `runs/runtime_covtypefull_dffl_capacity_s23_2026-04-27`
  - training time summary:
    - stacked: `69.57s`
    - hierarchical: `122.58s`
    - DFFL: `423.89s`

- Reproducibility appendix table (hyperparams per run) generated in:
  - `docs/artifacts/article_support_tables_2026-04-27.md`

## Manuscript updates

- Added capacity-matched diagnostic paragraph in results.
- Added tau-sensitivity paragraph in results.
- Added runtime trade-off paragraph in discussion.
- Updated in both:
  - `main(2).tex`
  - `docs/article_current/main.tex`

## Optional large dataset

- SUSY run was started (`runs/susy1m_capacity_matched_fuzzy3_s23_2026-04-27`),
  but the dataset download stage was too slow and was stopped to finish mandatory revision items first.
