# Article Submission Checklist (2026-04-27)

## Submission readiness (current)

- Ready to submit with current evidence package.
- Optional extra strengthening only: add one more large discriminative dataset (SUSY/HIGGS).

## Already fixed

- Covtype DFFL numbers updated to the completed 3-seed run (`0.8361 +/- 0.0031`).
- The historical manuscript source files were later removed from git when the
  repository was converted to code/artifact-only storage.
- Covtype large-scale table no longer marks DFFL as single-seed.
- Epoch-level training logs added to benchmark pipeline:
  - `--log-epochs`
  - `--log-epochs-every`
  - trainer prints `[epoch] ...` lines.

## Current stable run bundles to cite

- Stacked full Covtype (3 seeds, tuned):
  - `runs/covtypefull_stacked_w30_r30_s32_d015_t600_hs25_3s_2026-04-26/covtype_binary_full_report.json`
- DFFL full Covtype (3 seeds, fast profile):
  - `runs/covtypefull_dffl_fast_3s_2026-04-27/covtype_binary_full_report.json`
- KDD full baseline/scalability snapshot:
  - `runs/kddfull_baselines_s23_2026-04-26/kddcup99_binary_full_report.json`

## Remaining before submission

1. Optional quality strengthening: finish one more discriminative large dataset (SUSY or HIGGS subset/full).  
   Current status: SUSY run was started, but download was too slow; deferred to a separate run window.

## Newly completed in this cycle

1. Capacity fairness:
   - run completed: `runs/covtypefull_capacity_matched_fuzzy3_3s_2026-04-27`
   - key result: stacked `F1=0.9049+/-0.0043`, hierarchical `0.8540+/-0.0051`, DFFL `0.8021+/-0.0075`.
2. Stability sensitivity:
   - runs completed: `paper_main_fuzzy_tau03/05/07_3s_2026-04-27`
   - table assembled in `docs/artifacts/article_support_tables_2026-04-27.md`.
3. Repro appendix:
   - compact hyperparameter table assembled in `docs/artifacts/article_support_tables_2026-04-27.md`.
4. Runtime table:
   - runtime profile runs completed for stacked/hierarchical/DFFL on Covtype full (seed 23),
   - table assembled in `docs/artifacts/article_support_tables_2026-04-27.md`.

## Positioning to keep

- Keep conservative claim: trade-off study (quality/complexity/stability), not SOTA superiority claim.
