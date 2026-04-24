# Distillation + Threshold Tuning Round (2026-04-24)

## Code changes
- Added binary label-distillation options for non-DFFL fuzzy models in `examples/run_real_datasets_benchmark.py`:
  - `--fuzzy-distill-weight`
  - `--fuzzy-distill-models`
  - `--fuzzy-distill-teacher-trees`
- Fixed `--tune-fuzzy-threshold` behavior: it now tunes threshold for all selected fuzzy models (previously effectively DFFL-only).
- Added optional calibration-aware threshold tuning:
  - `--tune-fuzzy-threshold-calibrated`

## Covtype experiments (stacked)
- Distill (w=0.35), seed23: `F1=0.7933` -> reject.
- Distill (w=0.10), seed23: `F1=0.7808` -> reject.
- Threshold tuning fix, seed23: threshold `0.43`, `F1=0.8216` (single-seed gain).
- Threshold tuning fix, 3 seeds (full budget): `F1=0.8080 +/- 0.0156` -> below unified stacked `0.8129 +/- 0.0108`.
- Threshold tuning fix + calibrated, 3 seeds (full budget): `F1=0.8095 +/- 0.0134` -> still below unified stacked.

## Conclusion
- Distillation in current form degrades quality.
- Per-model threshold tuning is now implemented correctly, but for covtype it overfits thresholds on validation and does not improve 3-seed mean.
- Keep unified baseline numbers for paper tables.
