"""TrainConfig.epoch_checkpoints: every epoch kept, resume refuses gaps, default unchanged."""
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
import torch

from HGNDRecoGNN import device as device_mod
from HGNDRecoGNN.training import train as train_mod
from HGNDRecoGNN.training.checkpoint import load_checkpoint, save_checkpoint

CPU = device_mod.DeviceMap(main=torch.device('cpu'))


def run(tmp_path, monkeypatch, epochs, start_epoch=0, callback=None, model=None, **kw):
    model = model or torch.nn.Linear(2, 1)
    losses = iter([.9, .7, .8, .6, .65, .5, .55, .4] * 3)

    def step(model, *a, **k):  # weights move, so each epoch checkpoint differs
        with torch.no_grad():
            for p in model.parameters():
                p.add_(.01)
        return .5

    monkeypatch.setattr(train_mod, 'train_epoch', step)
    monkeypatch.setattr(train_mod, 'eval_epoch', lambda *a, **k: next(losses))
    cfg = train_mod.TrainConfig(epochs=epochs, start_epoch=start_epoch, device='cpu', checkpoint_dir=str(tmp_path),
                                verbose=False, **kw)
    train_mod.fit(model, [], [], cfg, plan=CPU, epoch_callback=callback)
    return model


def test_default_keeps_the_historical_file_set(tmp_path, monkeypatch):
    assert train_mod.TrainConfig().epoch_checkpoints is False
    assert asdict(train_mod.TrainConfig())['epoch_checkpoints'] is False
    run(tmp_path, monkeypatch, 3)
    assert not list(tmp_path.glob('epoch_*.pt'))
    assert {p.name for p in tmp_path.iterdir()} >= {'model.pt', 'last.pt', 'optim.pt', 'resume.pt', 'history.json'}


def test_every_epoch_is_kept_before_the_resume_commit(tmp_path, monkeypatch):
    seen = []

    def callback(epoch, history):
        blob = torch.load(tmp_path / train_mod.RESUME_NAME, weights_only=False)
        assert blob['next_epoch'] == epoch + 1 and len(history) == epoch + 1
        assert (tmp_path / train_mod.epoch_checkpoint_name(epoch)).stat().st_mtime_ns <= \
            (tmp_path / train_mod.RESUME_NAME).stat().st_mtime_ns
        seen.append(epoch)

    model = run(tmp_path, monkeypatch, 3, callback=callback, epoch_checkpoints=True)
    assert seen == [0, 1, 2]
    weights = [load_checkpoint(str(tmp_path / f'epoch_{e:03d}.pt')) for e in range(3)]
    assert [w.epoch for w in weights] == [0, 1, 2]
    assert not torch.equal(weights[0].state_dict['weight'], weights[2].state_dict['weight'])
    assert torch.equal(weights[2].state_dict['weight'], model.weight.detach())


def test_resume_continues_and_keeps_all_epochs(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, 2, epoch_checkpoints=True)
    run(tmp_path, monkeypatch, 4, start_epoch=2, epoch_checkpoints=True)
    assert [load_checkpoint(str(tmp_path / f'epoch_{e:03d}.pt')).epoch for e in range(4)] == [0, 1, 2, 3]
    blob = torch.load(tmp_path / train_mod.RESUME_NAME, weights_only=False)
    assert blob['next_epoch'] == 4 and len(blob['history']) == 4


def test_resume_refuses_a_missing_epoch(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, 2, epoch_checkpoints=True)
    (tmp_path / 'epoch_000.pt').unlink()
    with pytest.raises(ValueError, match='lacks epoch checkpoints'):
        run(tmp_path, monkeypatch, 3, start_epoch=2, epoch_checkpoints=True)


def test_resume_refuses_newest_epoch_weights_other_than_the_resume_state(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, 2, epoch_checkpoints=True)
    other = torch.nn.Linear(2, 1)
    save_checkpoint(str(tmp_path / 'epoch_001.pt'), other, arch_name='net_default', epoch=1)
    with pytest.raises(ValueError, match='differ from the resume state'):
        run(tmp_path, monkeypatch, 3, start_epoch=2, epoch_checkpoints=True)


def test_resume_without_the_atomic_state_is_refused(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, 2, epoch_checkpoints=True)
    (tmp_path / train_mod.RESUME_NAME).unlink()
    with pytest.raises(ValueError, match='requires the atomic resume.pt'):
        run(tmp_path, monkeypatch, 3, start_epoch=2, epoch_checkpoints=True)
