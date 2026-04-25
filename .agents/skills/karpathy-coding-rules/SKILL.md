---
name: karpathy-coding-rules
description: Conservative coding behavior for non-trivial tasks. Use when modifying code, adding features, refactoring, or debugging.
---

# Karpathy-style coding rules

## Core behavior

1. Understand existing code first.
- Read neighboring files and reuse current abstractions.
- Avoid inventing new layers unless there is a clear benefit.

2. Prefer minimal changes.
- Make the smallest correct readable diff.
- Avoid broad refactors unless they materially improve correctness or maintainability.

3. Avoid unnecessary complexity.
- No new dependencies without strong need.
- Prefer straightforward code over clever code.

4. Match the project style.
- Follow existing naming, structure, formatting, and architecture.

5. Validate before declaring success.
- Run appropriate tests/smokes for touched scope.
- Re-read changed code for accidental breakage.
- State assumptions and remaining risks.

6. Tell the truth about status.
- Never claim a run/test/metric that was not actually produced.
- When blocked, state the real blocker and next practical step.

7. Keep design structured, not inflated.
- Prefer DRY by removing duplicated logic in touched scope.
- Use OOP where domain entities/behavior benefit from explicit boundaries.
- Apply SOLID pragmatically, not dogmatically.
- Favor explicit contracts over hidden side effects.

## Bugs

- Reproduce/localize first.
- Fix root cause with minimal surface area.
- Do not silently mix unrelated fixes.
- If fix does not hold under validation, iterate until root cause is addressed or clearly documented.

## Features

- Integrate at narrowest stable point.
- Extend existing flows before creating parallel ones.

## ML/research code

- Prefer reproducibility over shortcuts.
- Keep experiment settings explicit and rerunnable.
- Keep metrics/claims aligned with saved artifacts.
- Keep project hygiene: remove temporary debug scripts/files after conclusions are captured in proper artifacts.
