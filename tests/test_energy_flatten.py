"""Spectrum-flattened energy loss must upweight the sparse high-energy bins.

The reference notebook trains the energy head twice, the second pass with a
`WeightedMSELoss` whose weights flatten the true-energy spectrum
(EgdeClass_Cluster_Jan25-xecs.ipynb cells 63 and 74-79).  Without it the loss
is dominated by the populated low-energy bins and the head pulls high-energy
clusters down; the converged checkpoint drifts from +/-1 % below 2.6 GeV to
-9.4 % at 4-5 GeV.  The hardness ratio's high band has no upper edge, so the
bias lands inside the observable.

`training/losses.py` is imported by file rather than through the package: the
package `__init__` pulls in `data.graph_dataset`, which needs sklearn, and this
test should run wherever torch does.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

_PATH = os.path.join(os.path.dirname(__file__), '..', 'training', 'losses.py')
_spec = importlib.util.spec_from_file_location('hgnd_losses', _PATH)
losses = importlib.util.module_from_spec(_spec)
sys.modules['hgnd_losses'] = losses      # dataclass resolves annotations here
_spec.loader.exec_module(losses)


def _falling_spectrum(n_low: int = 10_000, n_high: int = 300) -> torch.Tensor:
    """Many soft clusters, few hard ones — the shape the detector sees."""
    torch.manual_seed(0)
    return torch.cat([torch.rand(n_low) * 2.0, torch.rand(n_high) * 4.0 + 2.0])


def test_sparse_bins_weigh_more():
    edges, w = losses.build_energy_flatten_weights(_falling_spectrum(), bins=10)
    assert edges.numel() == 11 and w.numel() == 10

    cfg = losses.LossWeights(energy_flatten=True, energy_bin_edges=edges,
                             energy_bin_weights=w)
    lo = losses._energy_flatten_weights(torch.tensor([0.5]), cfg)
    hi = losses._energy_flatten_weights(torch.tensor([5.5]), cfg)
    assert hi > lo, f'high-energy weight {hi} must exceed low-energy {lo}'


def test_out_of_range_energies_land_in_end_bins():
    """np.digitize on interior edges only; nothing may index out of range."""
    edges, w = losses.build_energy_flatten_weights(_falling_spectrum(), bins=10)
    cfg = losses.LossWeights(energy_flatten=True, energy_bin_edges=edges,
                             energy_bin_weights=w)
    probe = torch.tensor([-5.0, 0.0, 3.0, 1e4])
    got = losses._energy_flatten_weights(probe, cfg)
    assert got.shape == probe.shape
    assert torch.isfinite(got).all()
    assert got[0] == w[0] and got[-1] == w[-1]


def test_disabled_by_default():
    """Existing results predate this; the default must stay plain MSE."""
    assert losses.LossWeights().energy_flatten is False
    assert losses._energy_flatten_weights(torch.tensor([1.0]),
                                          losses.LossWeights()) is None


def test_enabled_without_weights_raises():
    """Silently falling back to plain MSE would misreport what was trained."""
    try:
        losses._energy_flatten_weights(
            torch.tensor([1.0]), losses.LossWeights(energy_flatten=True))
    except ValueError:
        return
    raise AssertionError('expected ValueError when weights are missing')


def test_flattening_changes_the_energy_gradient():
    """The whole point: the same residuals must produce a different pull."""
    edges, w = losses.build_energy_flatten_weights(_falling_spectrum(), bins=10)
    e_true = torch.tensor([0.5, 5.5])
    plain_cfg = losses.LossWeights()
    flat_cfg = losses.LossWeights(energy_flatten=True, energy_bin_edges=edges,
                                  energy_bin_weights=w)

    grads = {}
    for tag, cfg in (('plain', plain_cfg), ('flat', flat_cfg)):
        # equal residuals, so any difference in the gradients is the weighting
        # and not the size of the miss
        pred = torch.tensor([0.4, 5.4], requires_grad=True)
        ww = losses._energy_flatten_weights(e_true, cfg)
        loss = (torch.nn.functional.mse_loss(pred, e_true) if ww is None
                else (ww * (pred - e_true) ** 2).mean())
        loss.backward()
        grads[tag] = pred.grad.clone()

    # plain MSE pulls both clusters equally hard; flattening pulls the sparse
    # high-energy one far harder
    plain_ratio = abs(float(grads['plain'][1] / grads['plain'][0]))
    flat_ratio = abs(float(grads['flat'][1] / grads['flat'][0]))
    assert abs(plain_ratio - 1.0) < 0.2, plain_ratio
    assert flat_ratio > 10 * plain_ratio, (plain_ratio, flat_ratio)


def test_sparse_bins_cannot_dominate_the_loss():
    """A tail bin holding a handful of clusters must not own the gradient.

    Exact flattening gives every bin an equal share of the loss however few
    clusters it holds. On the DCM sample the top two of ten bins held 29 of
    40,225 signal clusters and would have carried 20 % of the energy loss at a
    weight ratio of 1461, so the energy head would have been fitted to a couple
    of dozen examples.
    """
    torch.manual_seed(0)
    # a bulk plus a very sparse tail, the shape that triggers it
    e = torch.cat([torch.rand(20_000) * 3.0, torch.rand(25) * 4.0 + 4.0])

    _, w_raw = losses.build_energy_flatten_weights(e, bins=10, min_count=0)
    _, w_cap = losses.build_energy_flatten_weights(e, bins=10, min_count=200)

    raw_ratio = float(w_raw.max() / w_raw.min())
    cap_ratio = float(w_cap.max() / w_cap.min())
    assert raw_ratio > 100, raw_ratio
    assert cap_ratio < raw_ratio / 5, (raw_ratio, cap_ratio)

    counts = torch.histc(e, bins=10, min=float(e.min()), max=float(e.max()))
    share = (counts * w_cap) / (counts * w_cap).sum()
    assert float(share[-2:].sum()) < 0.10, float(share[-2:].sum())


def test_cap_preserves_relative_weighting_of_populated_bins():
    """The cap must not distort the bulk, where flattening is the whole point.

    Absolute weights do change, because capping is followed by a renormalisation
    that keeps the mean weight at 1 so the energy term holds its place in the
    composite loss. What must survive is the *ratio* between bins the cap did
    not touch.
    """
    e = _falling_spectrum()
    _, w_raw = losses.build_energy_flatten_weights(e, bins=10, min_count=0)
    _, w_cap = losses.build_energy_flatten_weights(e, bins=10, min_count=50)
    counts = torch.histc(e[e > 0], bins=10, min=float(e[e > 0].min()),
                         max=float(e[e > 0].max()))
    populated = [i for i in range(10) if counts[i] >= 50]
    assert len(populated) >= 2, populated
    ref = populated[0]
    for i in populated[1:]:
        r_raw = float(w_raw[i] / w_raw[ref])
        r_cap = float(w_cap[i] / w_cap[ref])
        assert abs(r_raw - r_cap) < 1e-4, (i, r_raw, r_cap)


def test_mean_weight_is_unity_so_the_energy_term_keeps_its_scale():
    """Enabling flattening must redistribute, not rescale.

    The composite loss sums the energy term with four BCE terms. If the mean
    weight drifted from 1 the energy term would change weight against them, and
    turning flattening on would silently retune the whole objective.
    """
    for e in (_falling_spectrum(),
              torch.cat([torch.rand(20_000) * 3.0, torch.rand(25) * 4.0 + 4.0])):
        edges, w = losses.build_energy_flatten_weights(e, bins=10,
                                                       min_count=200)
        pos = e[e > 0]
        wc = w[torch.bucketize(pos, edges[1:-1])]
        assert abs(float(wc.mean()) - 1.0) < 0.02, float(wc.mean())


def test_empty_bins_do_not_get_astronomical_weights():
    """An empty bin takes target/eps uncapped, which is a landmine.

    It contributes nothing to the training mean, so the defect is invisible
    until a cluster lands there at evaluation and carries a weight of order
    1e9. The cap bounds it to what a `min_count` bin would earn.
    """
    # a spectrum with a deliberate hole in the middle of the range
    e = torch.cat([torch.rand(10_000) * 1.0, torch.rand(10_000) * 1.0 + 4.0])
    _, w_raw = losses.build_energy_flatten_weights(e, bins=10, min_count=0)
    _, w_cap = losses.build_energy_flatten_weights(e, bins=10, min_count=200)
    assert float(w_raw.max()) > 1e6, float(w_raw.max())
    assert float(w_cap.max()) < 1e3, float(w_cap.max())


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
    print('all passed')
