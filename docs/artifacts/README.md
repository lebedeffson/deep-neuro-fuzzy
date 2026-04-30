# Paper-Facing Artifacts

This directory is for compact, curated result summaries used by the manuscript.
Raw experiment bundles live under `runs/` and are intentionally ignored by git.

## Current Manuscript

- Canonical LaTeX: `main(2).tex`
- Current focus: Routed Kolmogorov--Arnold Fuzzy Networks (Routed KAFN)
- Current KAFN figures:
  - `docs/figures/fig5_kafn_architecture.png`
  - `docs/figures/fig6_kafn_explanation_flow.png`
  - generator: `docs/figures/generate_kafn_figures.py`

## Current KAFN Run Sources

The main KAFN manuscript numbers were taken from local raw bundles:

- `runs/kafn_paper_all_3s_2026-04-30`
- `runs/cov20k_deep_kanfis_teacher_grouped_q12_fan20_r936_3s_2026-04-30`
- `runs/cov20k_kaanifs_tg_q12_f20_r936_binaryterms_3s_2026-04-30`
- `runs/cov20k_kaanifs_proj_p8w4_fixed_3s_2026-04-30`
- `runs/cov20k_fuzzy4_fixcheck_3s_2026-04-29`
- `runs/cov20k_kaanifs_binary_q12_vs_sklearn_3s_2026-04-30`

These directories are not committed because they are raw outputs. Keep the exact
run names in manuscript text or notes when a table depends on them.

## Historical Support Tables

- `article_revision_results_2026-04-27.md`
- `article_support_tables_2026-04-27.md`
- `finalization_state_2026-04-27.md`

These files document the previous DFFL/stacked/hierarchical revision state and
remain useful as traceability notes. They are not the canonical KAFN manuscript.

## Hygiene

Do not place raw logs, downloaded datasets, or whole benchmark directories here.
Only compact summaries that are directly referenced by the paper should be kept.
