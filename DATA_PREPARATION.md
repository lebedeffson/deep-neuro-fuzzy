# Data Preparation

This repository supports two run modes:

1. `make smoke` (no large local datasets required),
2. `make all` (article-scale experiments; requires local prepared datasets).

## Required local datasets for full article runs

Prepare and place datasets under local storage expected by your run configs.
Large benchmarks used in this project include:

- `covtype_binary_20000` / full `covtype_binary`
- `SUSY-200k`
- optional KDDCup99 variants

The repository intentionally does not commit large raw datasets.

## Quick verification

After environment setup:

```bash
make smoke
```

This validates installation and core tests without heavy data.

