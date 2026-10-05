"""Re-expressing cached features in another scaler must be exact.

Each S_pot cache was standardised with a scaler fitted to its own sample, so
the same physical neutron reaches the network differently depending on which
sample it came from. Across the three v3 caches the eToF scale differs by
0.41 % and its mean by 0.019 GeV, which biases the reconstructed energy by
about 0.8 % in the direction that cancels the S_pot signal. Standardisation is
affine, so the bias can be undone exactly rather than approximately — that is
what this checks.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def _coef(src: StandardScaler, dst: StandardScaler):
    """The mapping scripts/evaluate.py::_restandardise applies."""
    a = src.scale_ / dst.scale_
    b = (src.mean_ - dst.mean_) / dst.scale_
    return torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)


def test_round_trip_is_exact():
    rng = np.random.default_rng(0)
    raw = rng.normal(2.5, 2.6, size=(5000, 9))          # eToF-like columns

    # two samples whose feature distributions differ slightly, as the S_pot
    # samples' do
    src = StandardScaler().fit(raw)
    dst = StandardScaler().fit(raw * 1.004 + 0.019)

    x_src = torch.as_tensor(src.transform(raw), dtype=torch.float64)
    a, b = _coef(src, dst)
    got = x_src * a + b
    want = torch.as_tensor(dst.transform(raw), dtype=torch.float64)

    err = float((got - want).abs().max())
    assert err < 1e-9, f'restandardisation off by {err}'


def test_identity_when_scalers_match():
    rng = np.random.default_rng(1)
    raw = rng.normal(0, 1, size=(1000, 9))
    sc = StandardScaler().fit(raw)
    a, b = _coef(sc, sc)
    assert torch.allclose(a, torch.ones_like(a))
    assert torch.allclose(b, torch.zeros_like(b), atol=1e-12)


def test_surfacehit_column_is_untouched():
    """The tenth column is a raw flag, not a standardised feature."""
    rng = np.random.default_rng(2)
    raw = rng.normal(2.5, 2.6, size=(500, 9))
    src = StandardScaler().fit(raw)
    dst = StandardScaler().fit(raw * 1.004)

    x = torch.as_tensor(
        np.hstack([src.transform(raw), rng.integers(0, 2, (500, 1))]),
        dtype=torch.float64)
    flag_before = x[:, 9].clone()

    a, b = _coef(src, dst)
    x[:, :9] = x[:, :9] * a + b

    assert torch.equal(x[:, 9], flag_before)
    assert set(torch.unique(x[:, 9]).tolist()) <= {0.0, 1.0}


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
    print('all passed')
