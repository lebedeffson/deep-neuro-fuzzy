# Deep Fuzzy Feature Learning: Current Mathematical Formulation

## Current model claim

The current `ruanfis` architecture should be treated as a **deep fuzzy feature learning**
model rather than a fully general stacked deep ANFIS.

The model is organized as a sequence of transparent fuzzy stages

`x -> c^(1) -> c^(2) -> ... -> c^(L) -> y_hat`

where each hidden stage produces interpretable fuzzy concepts and the final layer performs
first-order Sugeno inference.

## Hidden block

For a hidden block with inputs `z = (z_1, ..., z_p)`:

1. Fuzzification:
   `mu_ij(z_i) in [0, 1]`

2. Rule weights:
   `w_r = sigma(g_r) * product_(i,j in A_r) mu_ij(z_i)`

3. Normalization:
   `w_bar_r = w_r / (sum_q w_q + eps)`

4. Concept output:
   `c = sum_r w_bar_r v_r`

Here `v_r` is the consequent concept vector of rule `r`.

## Decision layer

The final layer is a first-order Sugeno layer:

`y_hat = sum_r w_bar_r (a_r0 + a_r^T z)`

This preserves expressivity at the decision level while keeping the hidden representation
interpretable.

## Structural control of rule growth

Rule growth is controlled through:

- local blocks over feature groups;
- bounded rule arity;
- bounded number of generated rules;
- prototype-based rule generation from data.

This avoids the flat combinatorial explosion of a single global rule base.

## Stage-wise initialization and pretraining

The current development path is:

1. generate local rule bases from data;
2. bootstrap hidden blocks from rule activation profiles;
3. pretrain each stage with a temporary linear prediction head;
4. regenerate the next stage on the stabilized concept space;
5. initialize and pretrain the final Sugeno decision layer.

This moves the implementation closer to a truly deep transparent neuro-fuzzy architecture.

The implementation also supports **rule re-estimation from a trained reference model**:
after end-to-end fitting, the hidden and decision rule bases can be rebuilt on the
stabilized concept space produced by the trained model, and then stage-wise pretrained again.
This should be treated as a stronger data-driven reinitialization step, not as a guaranteed
improvement over every already fine-tuned reference model.

## Explainability

The implementation now provides two complementary views:

- exact local block-wise rule contributions;
- exact final decision decomposition into bias and concept contributions;
- path-based approximate hidden concept attribution to final outputs.

The last item is intentionally approximate and should be described as path-based attribution,
not as an exact global decomposition.

## Next mathematical targets

The next research-grade extensions are:

- iterative rule re-estimation after stage pretraining;
- stronger concept regularization and semantic constraints;
- richer overlap and coverage control for membership families;
- more faithful global attribution of lower-layer rules to the final prediction;
- systematic benchmarks against shallow fuzzy and non-fuzzy baselines.
