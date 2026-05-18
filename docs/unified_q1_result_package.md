# Unified Q1-Oriented Result Package

## What We Can Use Now

- Main method story: Budget-Prune is the reliable structural pruning method; Gate-L1 is a learned sparsification ablation that can win on small data.
- External baseline story: RuleFit is included as an interpretable competitor, not as a toy baseline.
- Control story: LR on the same top-K rules is consistently weaker than Budget-Prune on Covtype.
- Covtype B=400: Budget-Prune F1=0.8044, Gate-L1 F1=0.7939, Random-B F1=0.6756.
- Breast Cancer B=400: Gate-L1 F1=0.9700, Budget-Prune F1=0.9642, RuleFit F1=0.9578.
- Stable Budget-Prune proxy at B=400: current_jaccard=0.6384; stable_jaccard=0.8296; retention=0.9284.

## How To Combine Into One Paper

Use Budget-Prune as the main submitted method, Gate-L1 as ablation, RuleFit as external baseline, and Stable Budget-Prune as a stability-aware extension subsection.

Important: Stable Budget-Prune should stay clearly labelled as a stability-aware extension until full H-based F1/fidelity checks are complete.

## Missing Final Practical Check

Run one H-based validation pass for Stable Budget-Prune on Covtype B=400. If it preserves F1/fidelity, promote it into the main contribution. If not, keep it as future work.
