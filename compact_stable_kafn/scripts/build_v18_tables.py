from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _load_budget_prune_seed_metrics(covtype_root: Path, budgets: tuple[int, ...], seeds: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        for budget in budgets:
            report = covtype_root / f'seed_{seed}' / f'budget_{budget}' / 'covtype_binary_20000_report.json'
            if not report.exists():
                continue
            payload = _read_json(report)
            per_seed = payload.get('per_seed_results', [])
            if not per_seed:
                continue
            models = per_seed[0].get('results', [])
            target = None
            for m in models:
                if m.get('model_name') == 'ruanfis_kanfis':
                    target = m
                    break
            if target is None:
                continue
            tm = target.get('test_metrics', {})
            rows.append(
                {
                    'dataset': 'covtype_binary_20000',
                    'seed': int(seed),
                    'budget': int(budget),
                    'method': 'budget_prune',
                    'f1': float(tm.get('f1', float('nan'))),
                    'roc_auc': float(tm.get('roc_auc', float('nan'))),
                    'pr_auc': float(tm.get('pr_auc', float('nan'))),
                    'n_features': int(budget),
                }
            )
    return rows


def _load_lr_topk_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out.append(
                {
                    'dataset': row['dataset'],
                    'seed': int(row['seed']),
                    'budget': int(row['budget']),
                    'method': row['method'],
                    'f1': float(row['f1']),
                    'roc_auc': float(row['roc_auc']),
                    'pr_auc': float(row['pr_auc']),
                    'n_features': int(row.get('n_features', row['budget'])),
                }
            )
    return out


def _copy_csv_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')


def _iter_runtime_meta(run_root: Path):
    for path in run_root.rglob('compact_run_meta.json'):
        if '/baselines/' in str(path):
            continue
        payload = _read_json(path)
        dataset = str(payload.get('dataset', ''))
        budget_label = str(payload.get('budget_label', ''))
        elapsed = float(payload.get('elapsed_sec', 0.0))
        method = 'budget_prune'
        p = str(path)
        if '/method_l1_budget/' in p:
            method = 'l1_budget'
        elif '/method_random_b/' in p:
            method = 'random_b'
        elif '/method_budget_prune/' in p:
            method = 'budget_prune'
        if elapsed <= 0.0:
            continue
        yield dataset, method, budget_label, elapsed


def _build_runtime_table(roots: list[Path]) -> list[dict]:
    bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for dataset, method, budget_label, elapsed in _iter_runtime_meta(root):
            key = (dataset, method, budget_label)
            bucket[key].append(elapsed)
    rows: list[dict] = []
    for (dataset, method, budget_label), vals in sorted(bucket.items()):
        rows.append(
            {
                'dataset': dataset,
                'method': method,
                'budget': budget_label,
                'n_runs': len(vals),
                'elapsed_sec_mean': float(mean(vals)),
                'elapsed_sec_std': float(pstdev(vals)) if len(vals) > 1 else 0.0,
                'elapsed_sec_min': float(min(vals)),
                'elapsed_sec_max': float(max(vals)),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controls-dir', type=Path, required=True)
    ap.add_argument('--docs-dir', type=Path, default=Path('docs'))
    ap.add_argument('--covtype-runs', type=Path, default=Path('compact_stable_kafn/runs/compact_stable_kafn_covtype20k'))
    ap.add_argument('--susy-runs', type=Path, default=Path('compact_stable_kafn/runs/compact_stable_kafn_susy200k_q1'))
    ap.add_argument('--breast-runs', type=Path, default=Path('compact_stable_kafn/runs/compact_stable_kafn_breast_q1'))
    ap.add_argument('--budgets', type=str, default='100,200,400')
    ap.add_argument('--seeds', type=str, default='19,23,29')
    args = ap.parse_args()

    budgets = tuple(int(x.strip()) for x in args.budgets.split(',') if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(',') if x.strip())

    lr_rows = _load_lr_topk_rows(args.controls_dir / 'tables_lr_topk_baseline.csv')
    bp_rows = _load_budget_prune_seed_metrics(args.covtype_runs, budgets, seeds)
    combined = sorted(bp_rows + lr_rows, key=lambda r: (r['dataset'], r['budget'], r['seed'], r['method']))
    _write_csv(
        args.docs_dir / 'tables_lr_topk_baseline.csv',
        ['dataset', 'seed', 'budget', 'method', 'f1', 'roc_auc', 'pr_auc', 'n_features'],
        combined,
    )

    _copy_csv_if_exists(
        args.controls_dir / 'tables_rule_activation_correlation.csv',
        args.docs_dir / 'tables_rule_activation_correlation.csv',
    )
    _copy_csv_if_exists(
        args.controls_dir / 'tables_importance_profile.csv',
        args.docs_dir / 'tables_importance_profile.csv',
    )

    runtime_rows = _build_runtime_table([args.covtype_runs, args.susy_runs, args.breast_runs])
    _write_csv(
        args.docs_dir / 'tables_compact_kafn_runtime_minimal.csv',
        ['dataset', 'method', 'budget', 'n_runs', 'elapsed_sec_mean', 'elapsed_sec_std', 'elapsed_sec_min', 'elapsed_sec_max'],
        runtime_rows,
    )

    print(f'Wrote: {args.docs_dir / "tables_lr_topk_baseline.csv"}')
    print(f'Wrote: {args.docs_dir / "tables_compact_kafn_runtime_minimal.csv"}')
    print(f'Copied(if present): {args.docs_dir / "tables_rule_activation_correlation.csv"}')
    print(f'Copied(if present): {args.docs_dir / "tables_importance_profile.csv"}')


if __name__ == '__main__':
    main()
