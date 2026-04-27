# Article Submission Checklist (2026-04-27)

## Already fixed

- Covtype DFFL numbers updated to the completed 3-seed run (`0.8361 +/- 0.0031`) in:
  - `main(2).tex`
  - `docs/article_current/main.tex`
  - `docs/artifacts/README.md`
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

1. Capacity fairness: run a matched-capacity comparison (stacked vs hierarchical vs DFFL) on Covtype full.
2. Stability sensitivity: add `tau={0.3,0.5,0.7}` active-rule Jaccard sensitivity table.
3. Repro appendix: add compact per-dataset config table (optimizer, LR, epochs, batch, rules, group size, regularization weights).
4. Runtime table: train/infer/runtime per model for large datasets.
5. Optional quality strengthening: add one more discriminative large dataset (SUSY or HIGGS subset/full).

## Positioning to keep

- Keep conservative claim: trade-off study (quality/complexity/stability), not SOTA superiority claim.
