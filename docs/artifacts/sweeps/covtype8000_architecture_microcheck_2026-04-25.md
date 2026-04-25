# covtype_binary_8000 architecture micro-check (seed=23)

Setup: DFFL, one-phase, max_epochs=25, patience=8, batch_size=1024, tuned threshold + calibration.

| Variant | Key settings | Test F1 |
|---|---|---:|
| Control | residual off, top_k=24, total_budget=160, swap=0 | 0.7751 |
| No top-k pruning | residual off, top_k=None, total_budget=160, swap=0 | 0.7751 |
| Experimental new arch (rolled back) | residual on + softer high-dim cap + top_k=32 | 0.7567 |

Conclusion:
- Residual path + softer cap in this form worsened small covtype.
- top_k=24 is not the limiting factor here (same score as top_k=None).
- Keep control-like profile for covtype DFFL until next architecture change.
