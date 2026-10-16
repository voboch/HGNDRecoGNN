"""Seeding must happen before the model exists, and the split must be pinnable.

scripts/train.py constructed the model at line 264 while torch.manual_seed was
only called later, inside split_dataset and fit(). The initial weights -- and
the lazy SAGEConv shapes materialised by build_default_net's dummy forward --
were therefore drawn from whatever RNG state the process happened to be in, so
two runs with the same --seed did not start from the same weights.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pytest
import torch

REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from HGNDRecoGNN.training.train import (                     # noqa: E402
    TrainConfig, seed_everything, split_dataset)

ROOT = Path(__file__).resolve().parents[1]


def _fresh_weights(seed=None):
    if seed is not None:
        seed_everything(seed)
    torch.nn.Linear(8, 8)                 # consume some RNG, as a real build does
    return torch.nn.Linear(8, 8).weight.detach().clone()


def test_same_seed_gives_identical_initial_weights():
    assert torch.allclose(_fresh_weights(42), _fresh_weights(42))


def test_different_seeds_give_different_initial_weights():
    assert not torch.allclose(_fresh_weights(42), _fresh_weights(43))


def test_seed_everything_covers_numpy_too():
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    assert np.allclose(a, np.random.rand(5))


def test_cli_seeds_before_it_builds_the_model():
    """Structural, because the real model needs pyg-lib and a dataset.

    This asserts the ordering that was wrong: the seed call must precede the
    registry call that constructs the model.
    """
    src = (ROOT / "scripts" / "train.py").read_text()
    seed_at = src.find("seed_everything(args.seed)")
    build_at = src.find("model_registry.get(args.model")
    assert seed_at != -1, "scripts/train.py no longer seeds explicitly"
    assert build_at != -1, "could not find the model construction call"
    assert seed_at < build_at, \
        "the model is constructed before the seed is set, so initial weights " \
        "are not reproducible from --seed"


# ── the split seed ───────────────────────────────────────────────────────

class _FakeDataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _split_indices(**cfg_kw):
    cfg = TrainConfig(train_frac=0.5, **cfg_kw)
    tr, te = split_dataset(_FakeDataset(200), cfg)
    return sorted(tr.indices), sorted(te.indices)


def test_split_defaults_to_the_training_seed():
    """Historical behaviour: with split_seed unset, cfg.seed drives the split."""
    assert _split_indices(seed=42) == _split_indices(seed=42, split_seed=42)


def test_changing_only_the_training_seed_moves_the_split_by_default():
    assert _split_indices(seed=42) != _split_indices(seed=43)


def test_pinned_split_seed_holds_the_split_across_training_seeds():
    """So repeating a run with a new training seed keeps the same held-out set."""
    a = _split_indices(seed=42, split_seed=7)
    b = _split_indices(seed=43, split_seed=7)
    assert a == b, "split_seed did not isolate the split from the training seed"


def test_split_halves_are_disjoint_and_complete():
    tr, te = _split_indices(seed=42)
    assert not (set(tr) & set(te))
    assert len(tr) + len(te) == 200
