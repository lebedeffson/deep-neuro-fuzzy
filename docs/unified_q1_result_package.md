# Unified Q1-Oriented Result Package

## What We Can Use Now

- Main method story: Budget-Prune is the reliable structural pruning method; Gate-L1 is a learned sparsification ablation that can win on small data.
- External baseline story: RuleFit is included as an interpretable competitor, not as a toy baseline.
- Control story: LR on the same top-K rules is consistently weaker than Budget-Prune on Covtype.
- Covtype B=400: Budget-Prune F1=0.8044, Gate-L1 F1=0.7939, Random-B F1=0.6756.
- Breast Cancer B=400: Gate-L1 F1=0.9700, Budget-Prune F1=0.9642, RuleFit F1=0.9578.
- Stable Budget-Prune proxy at B=400: current_jaccard=0.6384; stable_jaccard=0.8296; retention=0.9284.
- Stable Budget-Prune H-based validation at B=400: bp_f1=0.7771; stable_f1=0.7739; fidelity_delta=0.0116; agreement_stable=0.8562; jaccard_to_bp=0.7464.
- Runtime anchor (SUSY B=200): budget_prune_runtime=25.74s (+/-0.94); selection-after-full-dictionary context (not strict train-speed race).
- Full KAFN reference (SUSY no-prune): full_no_prune_runtime=24.72s (+/-1.63); n=3; includes training+evaluation with full active dictionary.

## How To Combine Into One Paper

Use Budget-Prune as the main submitted method, Gate-L1 as ablation, RuleFit as external baseline, and Stable Budget-Prune as a validated stability-aware extension at B=400.

Important: Stable Budget-Prune improves stability with a small F1 drop in the H-based check; do not claim it improves every metric.

## Remaining Practical Gap

Covtype still has no RuleFit row, and SUSY still lacks matched KAFN quality rows in the unified main comparison. Keep those claims separate unless we run/recover them.

## Metric Notes

Importance retention in Stable proxy tables means retained heldout importance mass: sum(importance of selected subset under heldout seed) / sum(importance of heldout top-B).
Runtime numbers are context metrics for compactization workflow; they are not a strict apples-to-apples full training speed comparison against RuleFit.
SUSY B=200 runtime anchor reports the compact budgeted KAFN pipeline under that budget setting.
SUSY no-prune full reference reports full active-dictionary KAFN runtime over available seeds and is shown only as context, not as a strict speed race against RuleFit.
The work does not claim full end-to-end KAFN training speed superiority over RuleFit; the focus is selection quality and subset stability.
