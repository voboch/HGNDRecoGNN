"""fit() must resume: initialise its bookkeeping before reading the resume blob.

A chain of three cluster jobs died an hour in, each after a 50-minute preload,
on `UnboundLocalError: min_loss` — the resume block read counters that were
initialised below it.  The failure needs a GPU-free test, so this drives fit()
on a trivial model with a stub loader.
"""
from __future__ import annotations
import os, sys, tempfile
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from HGNDRecoGNN.training.train import fit, TrainConfig


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 1)

    def forward(self, batch):
        return self.lin(batch.x)


class _Batch:
    """Minimal stand-in for a PyG batch: the trainer calls .to(device)."""

    def __init__(self):
        self.x = torch.randn(8, 4)
        self.y = torch.randn(8, 1)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


def _loader(n=3):
    return [_Batch() for _ in range(n)]


def _fwd(model, batch, plan=None):
    return model(batch)


def _loss(out, batch, weights=None):
    # the trainer reads loss_dict['total']
    return {'total': torch.nn.functional.mse_loss(out, batch.y)}


def _run(ckpt_dir, epochs, start_epoch):
    torch.manual_seed(0)
    cfg = TrainConfig(epochs=epochs, start_epoch=start_epoch, batch_size=8,
                      device='cpu', checkpoint_dir=ckpt_dir,
                      checkpoint_name='model.pt', arch_name='tiny', verbose=False)
    return fit(_Tiny(), _loader(), _loader(), cfg,
               forward_fn=_fwd, loss_fn=_loss)


def test_resume_round_trip():
    d = tempfile.mkdtemp()
    _run(d, epochs=2, start_epoch=0)
    assert os.path.exists(os.path.join(d, 'optim.pt')), 'optim.pt not written'
    assert os.path.exists(os.path.join(d, 'last.pt')), 'last.pt not written'
    blob = torch.load(os.path.join(d, 'optim.pt'), map_location='cpu',
                      weights_only=False)
    assert blob['next_epoch'] == 2, blob['next_epoch']
    # the continuation is the case that broke the chain
    out = _run(d, epochs=4, start_epoch=blob['next_epoch'])
    assert len(out['train_loss']) == 2, 'resumed run should cover epochs 2 and 3'
    blob2 = torch.load(os.path.join(d, 'optim.pt'), map_location='cpu',
                       weights_only=False)
    assert blob2['next_epoch'] == 4, blob2['next_epoch']
    print('resume round-trip OK: 0-1 then 2-3, optimizer state carried')


def test_resume_without_optim_state():
    """A first continuation from a run that predates optim.pt must still work."""
    d = tempfile.mkdtemp()
    _run(d, epochs=1, start_epoch=0)
    os.remove(os.path.join(d, 'optim.pt'))
    _run(d, epochs=2, start_epoch=1)
    print('resume without optim.pt OK: falls back to a fresh optimizer')


if __name__ == '__main__':
    test_resume_round_trip()
    test_resume_without_optim_state()
    print('all resume tests passed')
