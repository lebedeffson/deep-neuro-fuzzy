## Rule-level Stability Limitation (production-facing note)

Rule-level Jaccard remains low at aggressive budgets (`B=25..100`) even when meta-cluster Jaccard is high. This means different runs can select different rule IDs from the same functional groups.

Practical implication:
- if a production workflow requires **exact rule identity stability**, use larger budgets (e.g., `B>=400`) or avoid strict pruning;
- for functional reproducibility of explanations, prefer reporting both rule-level and meta-cluster stability.

This is a limitation of hard budgeted reduction in redundant vocabularies, not evidence that explanations are semantically inconsistent.
