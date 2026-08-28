"""Regression test for the sensitivity CLI pipeline.

Guards against the class of bugs that surfaced during the 2026-08-27
full-statistics cHARISMa run (job 4284506):

  * ``_discover_parquet`` silently preferred a leftover ``_smoke``
    parquet over the full ``_v2`` build, yielding physically
    impossible ``k`` values (189 clusters per neutron) and closure
    metrics of ``~240``.

  * The paper's ``C_ν = k · C_c`` bridge identity was quoted from a
    per-neutron denominator that no longer matched the per-cluster
    numerator once the parquet was wrong.

The test runs the full sensitivity CLI against the local smoke
prediction pickles at ``results/sensitivity_hpc_pooled_smoke/`` and
the local smoke parquet caches at ``notebooks/cache/``. It exercises
every codepath the paper depends on except the model inference
itself.

Run with either

    python tests/test_sensitivity_pipeline.py
    python -m pytest tests/

Both entry points work: the ``if __name__ == '__main__'`` block
mirrors the ``test_*`` functions so pytest is optional.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))

import pandas as pd

DATASETS = ('defaultSpot', 'zeroSpot', 'bigSpot')
PRED_ROOT = REPO / 'results' / 'sensitivity_hpc_pooled_smoke'
CACHE_ROOT = REPO / 'notebooks' / 'cache'
REQUIRED_COLUMNS = (
    'dataset', 'U_sym_MeV', 'threshold', 'efficiency_basis',
    'N_events', 'N_reco', 'N_true', 'N_MC_truth', 'N_MC_clusters',
    'clusters_per_neutron', 'closure', 'closure_per_cluster',
)


def _skip_if_missing_prereqs() -> None:
    for ds in DATASETS:
        pkl = PRED_ROOT / f'{ds}_hpc' / 'pred_clusters_hpc_pooled.pkl'
        if not pkl.exists():
            print(f'SKIP: missing prediction pickle {pkl}. Rebuild with '
                  f'scripts/run_pooled_scaler_sensitivity.py.')
            raise SystemExit(0)
    for ds, suffix in (('defaultSpot', ''), ('zeroSpot', '_smoke'),
                       ('bigSpot', '_smoke')):
        cache = CACHE_ROOT / f'ndet_dataset_smash_{ds}{suffix}'
        if not (cache / 'processed').exists():
            print(f'SKIP: missing smoke cache {cache}')
            raise SystemExit(0)


def _run_sensitivity(out_dir: Path, extra_args: list[str] | None = None) -> pd.DataFrame:
    cmd = [
        sys.executable, '-m', 'HGNDRecoGNN.scripts.sensitivity',
        '--out-dir', str(out_dir),
        '--dataset-cache-root', str(CACHE_ROOT),
        '--efficiency-basis', 'e_pred',
        '--threshold', '0.5', '--no-plot',
    ]
    for ds in DATASETS:
        cmd += ['--dataset', f'{ds}={PRED_ROOT}/{ds}_hpc/pred_clusters_hpc_pooled.pkl']
    if extra_args:
        cmd += extra_args
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO.parent) + ':' + env.get('PYTHONPATH', '')
    result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                            env=env)
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        # Echo any MC-truth warning to caller so a regression prints reasoning.
        if 'WARNING' in line or 'MC parquet' in line:
            print('  cli:', line)
    return pd.read_csv(out_dir / 'sensitivity_summary.csv')


# ── Direct unit test on the parquet-discovery helper ──────────────────────

def test_discover_parquet_prefers_v2_over_smoke() -> None:
    """A `_v2` cache must win over `_smoke` when both exist for the same
    dataset name.  Simulates the exact configuration that bit 4284506."""
    from HGNDRecoGNN.scripts.sensitivity import _discover_parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        smoke = root / 'ndet_dataset_smash_foo_smoke' / 'processed'
        v2    = root / 'ndet_dataset_smash_foo_v2'    / 'processed'
        smoke.mkdir(parents=True); v2.mkdir(parents=True)
        (smoke / '_hits_cache_deadbeef.parquet').write_bytes(b'')
        (v2    / '_hits_cache_c0ffee.parquet').write_bytes(b'')
        got = _discover_parquet(str(root), 'foo')
        assert 'v2' in got and 'smoke' not in got, (
            f'expected _v2 preference but got {got!r} — regression toward '
            'the 2026-08-27 bug that pulled smoke parquets on full-stats.')
    print('  OK: _v2 preferred over _smoke')


def test_discover_parquet_falls_back_to_smoke_with_warning() -> None:
    """A `_smoke`-only cache must still be found (fallback) — used by
    laptop workflows that don't have a `_v2` cache."""
    from HGNDRecoGNN.scripts.sensitivity import _discover_parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        smoke = root / 'ndet_dataset_smash_bar_smoke' / 'processed'
        smoke.mkdir(parents=True)
        (smoke / '_hits_cache_deadbeef.parquet').write_bytes(b'')
        got = _discover_parquet(str(root), 'bar')
        assert got and 'smoke' in got, f'expected smoke fallback, got {got!r}'
    print('  OK: smoke fallback works')


# ── End-to-end CLI test ───────────────────────────────────────────────────

def _load_summary() -> pd.DataFrame:
    _skip_if_missing_prereqs()
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _run_sensitivity(Path(tmpdir))
    return summary


def test_summary_has_required_columns() -> None:
    df = _load_summary()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f'summary CSV missing columns {missing}: got {list(df.columns)}'
    print(f'  OK: all {len(REQUIRED_COLUMNS)} required columns present')


def test_k_and_bridge_identity() -> None:
    """clusters_per_neutron in [0.4, 1.0]; C_ν = k·C_c to within 1 %."""
    df = _load_summary()
    for _, row in df.iterrows():
        ds = row['dataset']
        k = row['clusters_per_neutron']
        c_c = row['closure_per_cluster']
        c_nu = row['closure']
        assert 0.3 <= k <= 1.0, (
            f'{ds}: k={k} outside plausible [0.3, 1.0] — parquet may be '
            'the wrong cache (2026-08-27 regression signature).')
        assert 0.5 <= c_c <= 2.0, f'{ds}: C_c={c_c} outside sane [0.5, 2.0]'
        product = k * c_c
        assert abs(product - c_nu) / max(abs(c_nu), 1e-9) < 0.02, (
            f'{ds}: bridge identity fails: k*C_c={product:.4f} vs '
            f'C_nu={c_nu:.4f} (diff {abs(product-c_nu):.4f})')
        print(f'  OK: {ds:12s} k={k:.3f} C_c={c_c:.3f} C_nu={c_nu:.3f} '
              f'(k·C_c={product:.3f})')


def test_mc_truth_coverage_reasonable() -> None:
    """N_MC/ev must be > 0.1 across the three smoke samples; anything
    lower means the parquet is empty or filtered incorrectly."""
    df = _load_summary()
    for _, row in df.iterrows():
        rate = row['N_MC_truth'] / max(row['N_events'], 1)
        assert rate > 0.1, (
            f"{row['dataset']}: N_MC/ev={rate:.3f} < 0.1 — parquet "
            "MC-truth undercount; check _discover_parquet.")
        print(f"  OK: {row['dataset']:12s} N_MC/ev={rate:.3f}")


if __name__ == '__main__':
    tests = [
        test_discover_parquet_prefers_v2_over_smoke,
        test_discover_parquet_falls_back_to_smoke_with_warning,
        test_summary_has_required_columns,
        test_k_and_bridge_identity,
        test_mc_truth_coverage_reasonable,
    ]
    for t in tests:
        print(f'\n=== {t.__name__} ===')
        try:
            t()
        except SystemExit:
            raise
        except AssertionError as exc:
            print(f'  FAIL: {exc}')
            raise SystemExit(1)
    print('\nall tests passed.')
