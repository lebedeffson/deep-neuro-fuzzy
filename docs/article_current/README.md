# Article Current Bundle

Single place for the current article version.

## Files
- `main.tex` — current manuscript text (synced from project root `main(2).tex`)
- `references.bib` — bibliography used by `main.tex`
- `figures/` — current paper figures (`fig1`..`fig4`)
- `article_metrics.json` — canonical metrics used to regenerate figures
- `references_raw_input.bib` — raw bibliography file provided for merge/check

## Regenerate figures
```bash
python docs/figures/generate_iiti26_figures.py \
  --metrics-json docs/article_current/article_metrics.json \
  --out-dir docs/article_current/figures
```

## Notes
- `main.tex` in this folder is intended for article-only workflow.
- If root `main(2).tex` changes, re-copy to keep this bundle in sync.
