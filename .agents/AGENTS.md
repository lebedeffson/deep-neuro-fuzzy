## Skill coordination rules

For non-trivial coding tasks, combine the general coding skill with the most specific domain skill.

If multiple skills apply:
- follow the most specific skill for local decisions;
- keep the general coding rules for code quality and validation;
- prefer small targeted changes over broad refactors.

For benchmark/result changes, keep one source of truth (`.json`) and sync derived tables from it.

## Execution commitments

- Truthfulness first: never invent run results, metrics, or validations.
- If something was not run, say it explicitly.
- Fix-to-done loop: localize issue -> apply minimal fix -> validate -> document residual risk.
- Keep design clean and practical: DRY, clear OOP boundaries, and SOLID where it improves maintainability.
- Do not over-engineer: prioritize small reversible changes.
- Keep repository clean: avoid temporary file sprawl and remove task-generated junk when work is complete.

## Engineering checklist

Before finishing non-trivial work, verify:
- behavior is confirmed by tests/smoke for touched scope;
- docs/tables are synced with the latest source JSON;
- temporary debug artifacts are removed or moved into explicit artifact folders;
- claims are conservative and evidence-backed.

## Skill loading rules

For any non-trivial coding task, load and follow:
- `.agents/skills/karpathy-coding-rules/SKILL.md`

When repository/runtime state is unclear, or before large edits, also load:
- `.agents/skills/repo-bringup/SKILL.md`

When task touches benchmark scripts, metric tables, model-vs-baseline comparisons, sweeps, or paper result claims, also load:
- `.agents/skills/benchmark-tradeoff-eval/SKILL.md`

When task touches CUDA/device issues, long runs, performance bottlenecks, or `--gpu-only` flows, also load:
- `.agents/skills/linux-gpu-runtime/SKILL.md`
