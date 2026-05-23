# Cluster-Prune: Theory Motivation (Safe Claims)

## Proposition 1 (Logit perturbation bound for subset pruning)
For full head
`z_full(x)=theta_0 + sum_{r=1}^m theta_r h_r(x)`
and compact subset `S`
`z_S(x)=theta_0 + sum_{r in S} theta_r h_r(x)`,
we have

`|z_full(x)-z_S(x)| <= sum_{r notin S} |theta_r| |h_r(x)|`.

Averaging on validation gives

`(1/n) sum_i |z_full(x_i)-z_S(x_i)| <= sum_{r notin S} |theta_r| (1/n) sum_i |h_r(x_i)|`.

This directly motivates proxy importance `|theta_r| * mean|h_r|`.

## Proposition 2 (Probability-level fidelity via Hoeffding)
Define probability gap `D_S(x)=|sigma(z_full(x))-sigma(z_S(x))| in [0,1]`.
For i.i.d. evaluation sample `T={x_i}_{i=1}^n`, with probability at least `1-delta`:

`E[D_S(X)] <= (1/n) sum_i D_S(x_i) + sqrt(log(1/delta)/(2n))`.

So empirical fidelity controls expected fidelity up to concentration term.

## Proposition 3 (Disagreement decomposition with margin)
For binary decisions by sign of logits:

`P( yhat_S(X) != yhat_full(X) ) <= P( |z_full(X)| <= gamma ) + P( |z_full(X)-z_S(X)| > gamma )`

for any `gamma>0`.
Using Markov on the second term:

`P( |z_full-z_S| > gamma ) <= E|z_full-z_S| / gamma`.

Hence agreement depends on both full-model margin and mean logit perturbation.

## Proposition 4 (Correlation implies profile closeness on validation)
For centered+normalized activation profiles `h~_r, h~_s`:

`|| h~_r - sign(rho_rs) h~_s ||_2^2 = 2(1-|rho_rs|)`.

If `|rho_rs| >= tau`, then

`|| h~_r - sign(rho_rs) h~_s ||_2 <= sqrt(2(1-tau))`.

This is a formal justification for clustering rules by activation correlation; it is not a direct F1 guarantee.

## Proposition 5 (Representative replacement contribution bound)
Let rule `r` be replaced by representative `s` in a cluster.
For any sample `x`:

`|theta_r h_r(x) - theta_s h_s(x)| <= |theta_r| |h_r(x)-h_s(x)| + |theta_r-theta_s| |h_s(x)|`.

If `|theta_r| <= W`, `|theta_s| <= W`, and `|h_s(x)| <= Hmax`, then

`|theta_r h_r(x) - theta_s h_s(x)| <= W |h_r(x)-h_s(x)| + |theta_r-theta_s| Hmax`.

Averaging over validation:

`E_val |theta_r h_r - theta_s h_s| <= W E_val |h_r-h_s| + |theta_r-theta_s| Hmax`.

Combined with Proposition 4 (high-correlation profiles imply small normalized profile distance), this gives a controlled
approximation argument for representative selection inside clusters.

## Scope statement
These are deviation-control and stability-motivation results.  
They do **not** claim global optimality of Cluster-Prune for F1.
