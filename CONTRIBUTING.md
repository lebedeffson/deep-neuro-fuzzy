# Contributing

## Development setup

```bash
python -m pip install -e .[dev]
pytest -q
```

## Project expectations

- keep the mathematical semantics explicit;
- prefer small, composable modules over monolithic code;
- add or update tests for every non-trivial change;
- preserve explainability as a first-class API concern.

## Pull request checklist

- tests pass locally;
- public APIs are documented or discoverable from examples;
- new logic is covered by targeted tests;
- no generated caches or local artifacts are committed.
