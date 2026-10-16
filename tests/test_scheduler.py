"""The LR schedule must not decay the run into the ground before it ends.

With the reference milestones (10, 15, 25) and the default epochs=20, the
third milestone never fires and the learning rate is already 1 % of its
initial value by epoch 15 -- a quarter of the budget trains at 1e-5. The
converged SMASH checkpoint's best epoch was 18 of 20, i.e. it was still
improving when the rate had already collapsed. These tests pin the existing
schedule down and check the alternatives behave as claimed.
"""

from pathlib import Path
import sys

import pytest
import torch

REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from HGNDRecoGNN.training.train import TrainConfig, _build_scheduler  # noqa: E402


def _trajectory(cfg, metric=None):
    """The LR actually in force during each epoch 0..epochs-1."""
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    sched, needs_metric = _build_scheduler(cfg, opt)
    lrs = []
    for epoch in range(cfg.epochs):
        lrs.append(opt.param_groups[0]['lr'])
        if needs_metric:
            sched.step(metric[epoch] if metric else 1.0)
        else:
            sched.step()
    return lrs


def test_multistep_is_unchanged_and_collapses_early():
    cfg = TrainConfig(epochs=20, lr=1e-3, verbose=False)
    lrs = _trajectory(cfg)
    assert lrs[:10] == pytest.approx([1e-3] * 10)
    assert lrs[10:15] == pytest.approx([1e-4] * 5)
    assert lrs[15:20] == pytest.approx([1e-5] * 5)
    # the documented pathology, asserted so it cannot be lost silently
    assert lrs[-1] == pytest.approx(1e-5), \
        'the run still ends at 1 % of the initial LR'


def test_the_unreachable_milestone_is_reported(capsys):
    cfg = TrainConfig(epochs=20, verbose=True)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    _build_scheduler(cfg, opt)
    out = capsys.readouterr().out
    assert 'never fire' in out and '25' in out, out


def test_no_warning_when_the_budget_covers_the_schedule(capsys):
    cfg = TrainConfig(epochs=30, verbose=True)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    _build_scheduler(cfg, opt)
    assert 'never fire' not in capsys.readouterr().out


def test_cosine_keeps_a_usable_rate_where_multistep_has_collapsed():
    """The point of the alternative: real steps are still being taken late."""
    ms = _trajectory(TrainConfig(epochs=20, lr=1e-3, verbose=False))
    cos = _trajectory(TrainConfig(epochs=20, lr=1e-3, scheduler='cosine',
                                  verbose=False))
    assert cos[0] == pytest.approx(1e-3)
    assert all(b <= a + 1e-12 for a, b in zip(cos, cos[1:])), 'monotone decay'
    assert cos[15] > 10 * ms[15], (
        f'cosine at epoch 15 is {cos[15]:.2e}, multistep {ms[15]:.2e}')
    assert cos[-1] < 1e-4, 'cosine must still anneal by the end'


def test_cosine_respects_its_floor():
    cfg = TrainConfig(epochs=10, lr=1e-3, scheduler='cosine', lr_min=1e-5,
                      verbose=False)
    assert min(_trajectory(cfg)) >= 1e-5 - 1e-12


def test_warmup_ramps_up_then_hands_over():
    cfg = TrainConfig(epochs=12, lr=1e-3, scheduler='cosine',
                      warmup_epochs=3, warmup_factor=10.0, verbose=False)
    lrs = _trajectory(cfg)
    assert lrs[0] == pytest.approx(1e-4), 'starts at lr / warmup_factor'
    assert lrs[0] < lrs[1] < lrs[2], 'ramps up during warmup'
    assert lrs[3] == pytest.approx(1e-3, rel=1e-6), 'reaches lr after warmup'
    assert lrs[-1] < lrs[3], 'then anneals'


def test_plateau_holds_while_validation_improves_and_cuts_when_it_stalls():
    cfg = TrainConfig(epochs=12, lr=1e-3, scheduler='plateau',
                      plateau_patience=2, scheduler_gamma=0.1, verbose=False)
    improving = [1.0 - 0.01 * i for i in range(6)]
    stalled = [improving[-1]] * 6
    lrs = _trajectory(cfg, metric=improving + stalled)
    assert lrs[5] == pytest.approx(1e-3), 'no decay while still improving'
    assert lrs[-1] < 1e-3, 'decays once validation stops improving'


def test_warmup_with_plateau_is_refused_not_silently_dropped():
    cfg = TrainConfig(epochs=10, scheduler='plateau', warmup_epochs=2,
                      verbose=False)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    with pytest.raises(ValueError, match='warmup_epochs is not supported'):
        _build_scheduler(cfg, opt)


def test_unknown_scheduler_is_refused():
    cfg = TrainConfig(scheduler='onecycle', verbose=False)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    with pytest.raises(ValueError, match='unknown scheduler'):
        _build_scheduler(cfg, opt)


def test_default_checkpoint_selection_is_validation():
    assert TrainConfig().save_best_on == 'val', (
        "selecting on training loss keeps the most overfit epoch; 'train' "
        "must be an explicit opt-in for reproducing older runs")


# ── the warning must not assume how many milestones there are ────────────
# An earlier version indexed sorted(milestones)[1], which raised IndexError
# on a single-milestone schedule. The CLI accepts --scheduler-milestones 25,
# so that configuration is reachable and the diagnostic crashed the run it
# was supposed to warn about.

@pytest.mark.parametrize("milestones,epochs,expect_warning", [
    ((25,), 20, True),            # one, unreachable -- used to raise
    ((5,), 20, False),            # one, reachable
    ((), 20, False),              # none at all
    ((10, 15, 25), 20, True),     # the default: two fire, one does not
    ((25, 30), 20, True),         # several, none reachable
    ((10, 15), 20, False),        # all reachable
])
def test_milestone_warning_survives_any_milestone_count(
        milestones, epochs, expect_warning, capsys):
    cfg = TrainConfig(epochs=epochs, scheduler_milestones=milestones,
                      verbose=True)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    _build_scheduler(cfg, opt)      # must not raise for any of these
    out = capsys.readouterr().out
    assert ('never fire' in out) is expect_warning, out


def test_the_warning_quotes_the_rate_the_run_actually_ends_at(capsys):
    """With (10, 15, 25) and 20 epochs two milestones fire: lr * gamma^2."""
    cfg = TrainConfig(epochs=20, lr=1e-3, scheduler_gamma=0.1,
                      scheduler_milestones=(10, 15, 25), verbose=True)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    _build_scheduler(cfg, opt)
    out = capsys.readouterr().out
    assert '1.0e-05' in out and 'epoch 15' in out, out


def test_the_warning_says_so_when_no_milestone_fires(capsys):
    cfg = TrainConfig(epochs=20, lr=1e-3, scheduler_milestones=(25, 30),
                      verbose=True)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=cfg.lr)
    _build_scheduler(cfg, opt)
    out = capsys.readouterr().out
    assert 'never drops' in out, out
