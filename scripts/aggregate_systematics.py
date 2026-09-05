"""Aggregate the Sec 6 items 4+5 systematic runs.

Consumes per-variant ``sensitivity_summary.csv`` files produced by
``slurm/sensitivity_systematics.hpc.sbatch`` and emits three artefacts
under ``results/systematics_aggregate/``:

1. ``per_variant_summary.csv`` — all variants side by side with their
   C_c, C_ν and per-event N_reco/ev for each dataset.
2. ``model_family_spread.csv`` — for item 4 (model-family systematic)
   the max-min spread in the per-event N_true/ev ratio
   $R_d = \\Ntrue(d)/\\Ntrue(\\text{defaultSpot})$ across the
   {net_default, hetero_hgt, hetero_sage} variants at seed 42.
3. ``seed_variance.csv`` — for item 5 (seed variance) the sample
   standard deviation of $R_d$ across the {seed 42, 123, 456}
   net_default variants.

Run:
    python scripts/aggregate_systematics.py \\
        --root results/systematics \\
        --baseline results/sensitivity_full_hpc_finish_4300982
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


DEFAULT_VARIANTS = [
    'hetero_hgt_seed42',
    'hetero_sage_seed42',
    'spectral_dynedge_seed42',
    'net_default_seed123',
    'net_default_seed456',
]


def _load_summary(root: Path, tag: str) -> pd.DataFrame | None:
    """Load the per-dataset sensitivity summary for one variant.
    Returns None if the file is missing (variant training failed)."""
    p = root / tag / 'sensitivity_e_pred' / 'sensitivity_summary.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df['variant'] = tag
    return df


def _ratio_to_default(df: pd.DataFrame, col: str = 'N_true') -> dict[str, float]:
    """Return {'zeroSpot': R_zero, 'bigSpot': R_big} where R_d = col(d) / col(default)."""
    idx = df.set_index('dataset')
    d = float(idx.loc['defaultSpot', col])
    if d == 0:
        return {'zeroSpot': float('nan'), 'bigSpot': float('nan')}
    return {ds: float(idx.loc[ds, col]) / d for ds in ('zeroSpot', 'bigSpot')}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='results/systematics',
                   help='Directory containing per-variant subdirs '
                        '(each with sensitivity_e_pred/sensitivity_summary.csv).')
    p.add_argument('--baseline',
                   default='results/sensitivity_full_hpc_finish_4300982',
                   help='Baseline (net_default seed 42) sensitivity run; '
                        'its e_pred[_fixed]/sensitivity_summary.csv becomes '
                        'the reference variant.')
    p.add_argument('--variants', nargs='*', default=DEFAULT_VARIANTS,
                   help='Systematics variant tags to include.')
    p.add_argument('--out-dir', default='results/systematics_aggregate')
    args = p.parse_args()

    root = Path(args.root)
    baseline = Path(args.baseline)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Baseline row ────────────────────────────────────────────────────────
    for cand in ('sensitivity_e_pred_fixed', 'sensitivity_e_pred'):
        base_csv = baseline / cand / 'sensitivity_summary.csv'
        if base_csv.exists():
            break
    else:
        print(f'ERROR: no baseline sensitivity_summary.csv under {baseline}')
        return 1
    base = pd.read_csv(base_csv)
    base['variant'] = 'net_default_seed42_baseline'
    print(f'baseline: {base_csv}')

    # ── Variant rows ────────────────────────────────────────────────────────
    frames = [base]
    present, missing = [], []
    for tag in args.variants:
        v = _load_summary(root, tag)
        if v is None:
            missing.append(tag)
            print(f'  SKIP {tag} — no sensitivity_summary.csv')
            continue
        present.append(tag)
        frames.append(v)
        print(f'  loaded {tag}')

    combined = pd.concat(frames, ignore_index=True)
    cols = ['variant', 'dataset', 'U_sym_MeV', 'threshold', 'efficiency_basis',
            'N_events', 'N_reco', 'N_reco/ev', 'N_true', 'N_true_err',
            'N_MC_truth', 'N_MC/ev', 'N_MC_clusters',
            'clusters_per_neutron', 'closure', 'closure_per_cluster']
    combined = combined[[c for c in cols if c in combined.columns]]
    per_var_path = out_dir / 'per_variant_summary.csv'
    combined.to_csv(per_var_path, index=False)
    print(f'\nwrote {per_var_path}')

    # ── Compute per-variant N_true/ev ratios vs defaultSpot ─────────────────
    ratio_rows = []
    for variant, g in combined.groupby('variant'):
        r = _ratio_to_default(g, col='N_true')
        cc = _ratio_to_default(g, col='closure_per_cluster')  # dataset-level C_c
        ratio_rows.append({
            'variant': variant,
            'R_zero_over_default_Ntrue': round(r['zeroSpot'], 4),
            'R_big_over_default_Ntrue':  round(r['bigSpot'],  4),
            'Cc_zero_over_default':      round(cc['zeroSpot'], 4),
            'Cc_big_over_default':       round(cc['bigSpot'],  4),
        })
    ratios = pd.DataFrame(ratio_rows)
    print('\nper-variant ratios (N_true(d)/N_true(defaultSpot)):')
    print(ratios.to_string(index=False))

    # ── Item 4: model-family spread ─────────────────────────────────────────
    family_variants = [v for v in ratios.variant
                       if any(k in v for k in ('net_default_seed42_baseline',
                                               'hetero_', 'spectral_'))]
    fam = ratios[ratios.variant.isin(family_variants)]
    fam_stats = []
    for col in ('R_zero_over_default_Ntrue', 'R_big_over_default_Ntrue'):
        v = fam[col].dropna()
        if len(v) < 2:
            continue
        fam_stats.append({
            'ratio':     col,
            'n_variants': int(len(v)),
            'mean':      round(float(v.mean()),   4),
            'max_min':   round(float(v.max()-v.min()), 4),
            'std':       round(float(v.std(ddof=1)),   4),
            'variants':  ','.join(fam.variant.tolist()),
        })
    fam_df = pd.DataFrame(fam_stats)
    fam_path = out_dir / 'model_family_spread.csv'
    fam_df.to_csv(fam_path, index=False)
    print(f'\nmodel-family spread (Sec 6 item 4):')
    print(fam_df.to_string(index=False))
    print(f'wrote {fam_path}')

    # ── Item 5: seed variance (net_default 42/123/456) ──────────────────────
    seed_variants = ['net_default_seed42_baseline', 'net_default_seed123',
                     'net_default_seed456']
    seed = ratios[ratios.variant.isin(seed_variants)]
    seed_stats = []
    for col in ('R_zero_over_default_Ntrue', 'R_big_over_default_Ntrue'):
        v = seed[col].dropna()
        if len(v) < 2:
            continue
        seed_stats.append({
            'ratio':     col,
            'n_seeds':   int(len(v)),
            'mean':      round(float(v.mean()),   4),
            'max_min':   round(float(v.max()-v.min()), 4),
            'std':       round(float(v.std(ddof=1)),   4),
            'variants':  ','.join(seed.variant.tolist()),
        })
    seed_df = pd.DataFrame(seed_stats)
    seed_path = out_dir / 'seed_variance.csv'
    seed_df.to_csv(seed_path, index=False)
    print(f'\nseed variance (Sec 6 item 5):')
    print(seed_df.to_string(index=False))
    print(f'wrote {seed_path}')

    # ── Provenance manifest ─────────────────────────────────────────────────
    manifest = {
        'baseline_summary': str(base_csv),
        'variant_root': str(root),
        'variants_present': present,
        'variants_missing': missing,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nwrote {out_dir / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
