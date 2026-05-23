# C-Prune Theory Note (Heuristic, For Paper Draft)

This note formalizes why cluster-level stability can be much higher than rule-level stability.

## Setup

Let the compact model logit be

\[
z(x)=b+\sum_{r\in S}\theta_r h_r(x), \quad h_r(x)\in[0,1].
\]

Rules are grouped into clusters by activation-profile similarity (absolute correlation threshold \(\tau\)).
Inside a cluster \(C\), one representative \(r^\*\) is selected.

## Lemma 1 (Replacement Error Bound, sample-level in \(L_2\))

Let \(u_r, u_s \in \mathbb{R}^n\) be centered+normalized activation vectors of rules \(r,s\) on a reference sample, and
\[
|\mathrm{corr}(u_r,u_s)|\ge \tau.
\]
Then
\[
\|u_r-\mathrm{sign}(\rho_{rs})u_s\|_2 \le \sqrt{2(1-\tau)}.
\]

Hence, for bounded rule weight \(|\theta|\le W\), replacing \(r\) by \(s\) changes the logit vector on the reference sample by at most
\[
\|\Delta z\|_2 \le W\sqrt{2(1-\tau)}.
\]

Interpretation: larger \(\tau\) implies smaller replacement error.

## Corollary 1 (Cluster Representative Approximation)

If every rule in cluster \(C\) satisfies \(|\mathrm{corr}(r,r^\*)|\ge\tau\), then replacing all rules in \(C\) by representative \(r^\*\) yields bounded aggregate perturbation of logits, with bound proportional to \(\sqrt{1-\tau}\).

So C-Prune controls approximation error via \(\tau\): lower \(\tau\) -> coarser clusters (more compression, potentially larger approximation error), higher \(\tau\) -> finer clusters (less compression, smaller error).

## Proposition 1 (Why Meta-Cluster Jaccard > Rule Jaccard)

Assume each functional cluster contains multiple near-substitutable rules. Across seeds, optimization may pick different members of the same cluster due to small training perturbations. Then:

1. Rule identity overlap is low (low rule-level Jaccard),
2. Cluster identity overlap is high (high meta-cluster Jaccard).

Thus, cluster-level stability is expected to dominate rule-level stability in redundant vocabularies.

## Practical Implication

For explanation reproducibility, reporting only rule-level Jaccard is misleading in the presence of redundant correlated rules. A paired report:

- rule-level Jaccard,
- meta-cluster Jaccard

is necessary to separate "ID instability" from "semantic instability".

## Scope / Limitations

- Bounds are sample-dependent and heuristic (not global worst-case guarantees for F1/AUC).
- Meta-cluster stability depends on reference signature design (we used distributional activation signatures when row-aligned cross-seed references are unavailable).
- This note supports methodological validity, not strict optimality guarantees.

