# Reproducibility

## 1) Environment

Option A (conda):

```bash
conda env create -f environment.yml
conda activate deep-neuro-fuzzy
```

Option B (pip):

```bash
python -m pip install -U pip
python -m pip install -e .[dev]
```

## 2) Data

Prepare datasets as described in [DATA_PREPARATION.md](DATA_PREPARATION.md).

## 3) Smoke check

```bash
make smoke
```

## 4) Full run

```bash
make all
```

`make all` expects locally prepared datasets and run configs.

## 5) Key outputs

Compact paper tables are generated under:

- `compact_stable_kafn/paper_tables/covtype20k/`
- `compact_stable_kafn/paper_tables/susy200k_q1/`
- `compact_stable_kafn/paper_tables/breast_q1/`

## 6) Version pin for article

Use a fixed git tag in manuscripts (example):

- `v1.0-submission`
- `https://github.com/lebedeffson/deep-neuro-fuzzy/tree/v1.0-submission`

Do not reference moving `main` for final reported numbers.
