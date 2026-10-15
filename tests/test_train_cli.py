"""The CLI must turn flags into the configuration the job intended.

These runs cost a 12-hour GPU allocation each, and a flag that parses but
does not reach the model fails silently: the checkpoint records an
architecture it never had. `slurm/train_dcm.hpc.sbatch` carries a comment
about two earlier DCM jobs that "died on it in 2 s" for a comparable reason.

Nothing here needs a dataset, a GPU or pyg-lib, so it runs on the laptop
before anything is submitted.
"""

from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

_spec = importlib.util.spec_from_file_location(
    "_train_cli", ROOT / "scripts" / "train.py")
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)

from HGNDRecoGNN.training.train import TrainConfig  # noqa: E402


class _Args:
    """Stand-in for the argparse namespace, with argparse's own defaults."""

    def __init__(self, **kw):
        self.cluster_pooling = None
        self.cluster_size_feature = False
        self.hit_residual = False
        self.scheduler = None
        self.warmup_epochs = None
        self.lr_min = None
        self.grad_clip_norm = None
        self.__dict__.update(kw)


# ── architecture flags ────────────────────────────────────────────────────

def test_no_flags_means_no_arch_overrides():
    """An unflagged run must get net_default's own defaults, untouched."""
    assert _cli._net_default_arch_flags(_Args()) == {}


def test_each_arch_flag_reaches_arch_kwargs():
    got = _cli._net_default_arch_flags(_Args(
        cluster_pooling='mean_max_sum',
        cluster_size_feature=True,
        hit_residual=True))
    assert got == {'cluster_pooling': 'mean_max_sum',
                   'cluster_size_feature': True,
                   'hit_residual': True}


def test_store_true_flags_are_omitted_when_false_not_set_to_false():
    """Passing cluster_size_feature=False must not record it.

    arch_kwargs is written into the checkpoint, so an explicit False would
    claim the run configured something it only left alone.
    """
    got = _cli._net_default_arch_flags(_Args(cluster_size_feature=False))
    assert 'cluster_size_feature' not in got


def test_arch_flags_are_accepted_by_the_model_constructor():
    """The names must match Net.__init__, not merely be well-formed dicts."""
    import inspect
    spec = importlib.util.spec_from_file_location(
        "_dn", ROOT / "models" / "default_net.py")
    dn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dn)
    params = set(inspect.signature(dn.Net.__init__).parameters)
    flags = _cli._net_default_arch_flags(_Args(
        cluster_pooling='mean_max_sum',
        cluster_size_feature=True,
        hit_residual=True))
    unknown = set(flags) - params
    assert not unknown, f"Net.__init__ would reject: {unknown}"


# ── schedule flags ───────────────────────────────────────────────────────

def test_no_schedule_flags_means_no_overrides():
    assert _cli._schedule_overrides(_Args()) == {}


def test_schedule_flags_reach_the_config_and_are_valid_fields():
    got = _cli._schedule_overrides(_Args(
        scheduler='cosine', warmup_epochs=3, lr_min=1e-6,
        grad_clip_norm=1.0))
    assert got == {'scheduler': 'cosine', 'warmup_epochs': 3,
                   'lr_min': 1e-6, 'grad_clip_norm': 1.0}
    # must be real TrainConfig fields, or fit() raises TypeError on the node
    cfg = TrainConfig(**got)
    assert cfg.scheduler == 'cosine' and cfg.warmup_epochs == 3


def test_zero_warmup_is_distinguished_from_unset():
    """--warmup-epochs 0 is a deliberate choice and must survive as 0."""
    assert _cli._schedule_overrides(_Args(warmup_epochs=0)) == \
        {'warmup_epochs': 0}


def test_grad_clip_zero_is_not_silently_dropped():
    assert _cli._schedule_overrides(_Args(grad_clip_norm=0.0)) == \
        {'grad_clip_norm': 0.0}


# ── the exact invocations the sbatch scripts use ─────────────────────────

@pytest.mark.parametrize("argv,expect_cfg,expect_arch", [
    # step 3: budget and schedule only
    (['--scheduler', 'cosine', '--warmup-epochs', '3'],
     {'scheduler': 'cosine', 'warmup_epochs': 3}, {}),
    # step 4a: cluster size on top
    (['--scheduler', 'cosine', '--warmup-epochs', '3',
      '--cluster-size-feature'],
     {'scheduler': 'cosine', 'warmup_epochs': 3},
     {'cluster_size_feature': True}),
    # step 4b: and the wider pooling
    (['--scheduler', 'cosine', '--warmup-epochs', '3',
      '--cluster-size-feature', '--cluster-pooling', 'mean_max_sum'],
     {'scheduler': 'cosine', 'warmup_epochs': 3},
     {'cluster_size_feature': True, 'cluster_pooling': 'mean_max_sum'}),
    # step 4c: plus residuals
    (['--scheduler', 'cosine', '--warmup-epochs', '3',
      '--cluster-size-feature', '--cluster-pooling', 'mean_max_sum',
      '--hit-residual'],
     {'scheduler': 'cosine', 'warmup_epochs': 3},
     {'cluster_size_feature': True, 'cluster_pooling': 'mean_max_sum',
      'hit_residual': True}),
])
def test_planned_invocations_parse_to_the_intended_config(
        argv, expect_cfg, expect_arch, monkeypatch):
    """Parse the real argv the jobs pass, through the real parser."""
    full = ['train', '--root', '/nonexistent', '--model', 'net_default'] + argv
    monkeypatch.setattr(sys, 'argv', full)

    # main() builds a dataset, so only the parser is exercised here. It is
    # rebuilt by re-running the argument section via a stub main.
    import argparse
    captured = {}

    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        ns = real_parse(self, *a, **k)
        captured['ns'] = ns
        raise SystemExit(0)      # stop before the dataset is touched

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', spy)
    with pytest.raises(SystemExit):
        _cli.main()

    ns = captured['ns']
    assert _cli._schedule_overrides(ns) == expect_cfg
    assert _cli._net_default_arch_flags(ns) == expect_arch
    TrainConfig(**_cli._schedule_overrides(ns))   # must not raise


def test_save_best_on_defaults_to_val_in_the_cli(monkeypatch):
    import argparse
    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        captured['ns'] = real_parse(self, *a, **k)
        raise SystemExit(0)

    monkeypatch.setattr(sys, 'argv',
                        ['train', '--root', '/nonexistent',
                         '--model', 'net_default'])
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', spy)
    with pytest.raises(SystemExit):
        _cli.main()
    assert captured['ns'].save_best_on == 'val'
