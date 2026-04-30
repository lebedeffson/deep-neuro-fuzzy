# Finalization State (2026-04-27)

## What We Already Have (Reliable)

- Historical article bundle existed at the time of this note; manuscript
  sources were later removed from git.
- Reproducibility manifest pipeline (runtime/CUDA/git/command).
- Strong Covtype tuned stacked result with 3 seeds.
- Fuzzy-vs-baseline snapshots on Covtype and KDD (seed 23).

## Covtype Full Snapshot

| Model | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.7551 +/- 0.0000 | 0.8274 +/- 0.0000 | 0.8040 +/- 0.0000 |
| shallow fuzzy | 0.7700 +/- 0.0000 | 0.8311 +/- 0.0000 | 0.7951 +/- 0.0000 |
| DFFL (seed 23) | 0.7952 +/- 0.0000 | 0.8695 +/- 0.0000 | 0.8474 +/- 0.0000 |
| HGB | 0.8321 +/- 0.0000 | 0.9175 +/- 0.0000 | 0.9089 +/- 0.0000 |
| MLP | 0.8591 +/- 0.0000 | 0.9411 +/- 0.0000 | 0.9346 +/- 0.0000 |
| hierarchical fuzzy (seed 23) | 0.9033 +/- 0.0000 | 0.9682 +/- 0.0000 | 0.9649 +/- 0.0000 |
| stacked fuzzy (3 seeds tuned) | 0.9466 +/- 0.0015 | 0.9895 +/- 0.0005 | 0.9886 +/- 0.0005 |
| ExtraTrees | 0.9547 +/- 0.0000 | 0.9918 +/- 0.0000 | 0.9912 +/- 0.0000 |
| Random Forest | 0.9559 +/- 0.0000 | 0.9924 +/- 0.0000 | 0.9917 +/- 0.0000 |

## KDD Full Snapshot (Scalability)

| Model | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: |
| shallow fuzzy | 0.9993 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| stacked fuzzy | 0.9999 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| hierarchical fuzzy | 0.9998 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| Logistic Regression | 0.9991 +/- 0.0000 | 0.9997 +/- 0.0000 | 0.9999 +/- 0.0000 |
| MLP | 0.9999 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| RF | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| ExtraTrees | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| HGB | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

## Gaps To Close Before Submission

1. Run a practical multi-seed large-suite pack (at least 3 seeds, ideally 5-10).
2. Add threshold sensitivity report for active-rule criterion (`tau=0.3/0.5/0.7`).
3. Add compact appendix table with concrete training/config values per dataset.
4. Keep claims aligned: trade-off study, not SOTA superiority claim.

## Run Control Note

- The very long `q1_large` 10-seed run from 2026-04-26 was stopped because it was still in DFFL seed-1 stage after >1 hour and did not produce incremental dataset outputs.
- Next step is staged faster runs (`--seed-count 3`, `--dffl-fast-gpu`) to get complete comparable outputs first.
