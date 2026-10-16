"""Resume must never pair weights with an optimizer from another epoch.

The old contract wrote optim.pt (carrying next_epoch) and last.pt (carrying
the weights) as two files, committing optim.pt FIRST. A job stopped between
the two writes left next_epoch one ahead of the weights on disk, and the
resume path loaded both without checking. The review reproduced exactly that:
optim.next_epoch=2 against last.epoch=0.

These tests pin the new contract: one atomic resume.pt written last, and a
refusal when the legacy pair disagrees.
"""

from pathlib import Path
import os
import sys

import pytest
import torch

REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from HGNDRecoGNN import device as device_mod                     # noqa: E402
from HGNDRecoGNN.training import train as train_mod              # noqa: E402
from HGNDRecoGNN.training.checkpoint import save_checkpoint      # noqa: E402


def _run(tmp_path, monkeypatch, *, epochs=2, start_epoch=0, **cfg_kw):
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(train_mod, "train_epoch", lambda *a, **k: 0.5)
    monkeypatch.setattr(train_mod, "eval_epoch", lambda *a, **k: 0.5)
    cfg = train_mod.TrainConfig(
        epochs=epochs, start_epoch=start_epoch, device="cpu",
        checkpoint_dir=str(tmp_path), verbose=False, **cfg_kw)
    return train_mod.fit(model, train_loader=[], test_loader=[], cfg=cfg,
                         plan=device_mod.DeviceMap(main=torch.device("cpu")))


def _legacy_pair(tmp_path, next_epoch, last_epoch):
    """The old two-file layout, with the generations set independently."""
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10], gamma=0.1)
    torch.save({"optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
                "next_epoch": next_epoch, "min_loss": 0.25, "best_epoch": 0},
               tmp_path / "optim.pt")
    if last_epoch is not None:
        save_checkpoint(str(tmp_path / "last.pt"), model,
                        arch_name="net_default", epoch=last_epoch)


# ── the single committed file ────────────────────────────────────────────

def test_resume_file_is_written_and_self_contained(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, epochs=2)
    blob = torch.load(tmp_path / train_mod.RESUME_NAME, map_location="cpu",
                      weights_only=False)
    for key in ("model", "optimizer", "scheduler", "next_epoch",
                "min_loss", "best_epoch", "rng"):
        assert key in blob, f"resume.pt is missing {key!r}"
    assert blob["next_epoch"] == 2
    assert "torch" in blob["rng"], "RNG state must travel with the resume"


def test_resume_is_committed_last(tmp_path, monkeypatch):
    """resume.pt must be the newest file, so it is the commit point.

    If it were written before last.pt or optim.pt, a kill in between would
    leave it describing an epoch whose weights were never stored.
    """
    _run(tmp_path, monkeypatch, epochs=1)
    r = (tmp_path / train_mod.RESUME_NAME).stat().st_mtime_ns
    for other in ("last.pt", "optim.pt"):
        assert (tmp_path / other).stat().st_mtime_ns <= r, \
            f"{other} was written after resume.pt"


def test_resume_restores_weights_not_just_the_optimizer(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, epochs=1)
    blob = torch.load(tmp_path / train_mod.RESUME_NAME, map_location="cpu",
                      weights_only=False)
    saved_w = blob["model"]["weight"].clone()

    # a fresh model with different weights, resumed from the file
    fresh = torch.nn.Linear(1, 1)
    with torch.no_grad():
        fresh.weight.fill_(999.0)
    monkeypatch.setattr(train_mod, "train_epoch", lambda *a, **k: 0.5)
    monkeypatch.setattr(train_mod, "eval_epoch", lambda *a, **k: 0.5)
    cfg = train_mod.TrainConfig(epochs=1, start_epoch=1, device="cpu",
                                checkpoint_dir=str(tmp_path), verbose=False)
    train_mod.fit(fresh, train_loader=[], test_loader=[], cfg=cfg,
                  plan=device_mod.DeviceMap(main=torch.device("cpu")))
    assert torch.allclose(fresh.weight.detach().cpu(), saved_w), \
        "resume must restore the weights, not leave the caller's"


# ── the legacy pair is checked, not trusted ──────────────────────────────

def test_mismatched_legacy_pair_is_refused(tmp_path, monkeypatch):
    """The review's exact scenario: next_epoch=2 with last.epoch=0."""
    _legacy_pair(tmp_path, next_epoch=2, last_epoch=0)
    with pytest.raises(RuntimeError, match="inconsistent resume state"):
        _run(tmp_path, monkeypatch, epochs=3, start_epoch=2)


def test_consistent_legacy_pair_is_accepted(tmp_path, monkeypatch):
    _legacy_pair(tmp_path, next_epoch=1, last_epoch=0)
    res = _run(tmp_path, monkeypatch, epochs=2, start_epoch=1)
    assert res["best_loss"] == 0.25


def test_absent_last_pt_is_the_init_from_flow_not_a_fault(tmp_path, monkeypatch):
    """scripts/train.py --init-from loads the weights before fit() runs."""
    _legacy_pair(tmp_path, next_epoch=1, last_epoch=None)
    res = _run(tmp_path, monkeypatch, epochs=2, start_epoch=1)
    assert res["best_loss"] == 0.25


def test_mismatch_can_be_overridden_explicitly(tmp_path, monkeypatch):
    _legacy_pair(tmp_path, next_epoch=2, last_epoch=0)
    res = _run(tmp_path, monkeypatch, epochs=3, start_epoch=2,
               resume_allow_mismatch=True)
    assert res is not None, "the escape hatch must still run"


def test_truncated_last_pt_is_detected(tmp_path, monkeypatch):
    """A killed torch.save leaves a file that only fails on load."""
    _legacy_pair(tmp_path, next_epoch=1, last_epoch=0)
    (tmp_path / "last.pt").write_bytes(b"\x80\x02 truncated")
    with pytest.raises(RuntimeError, match="inconsistent resume state"):
        _run(tmp_path, monkeypatch, epochs=2, start_epoch=1)


# ── atomic checkpoint writes ─────────────────────────────────────────────

def test_save_checkpoint_leaves_no_temporary_behind(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(str(path), torch.nn.Linear(1, 1), arch_name="x", epoch=3)
    assert path.exists()
    assert not (tmp_path / "model.pt.tmp").exists()


def test_failed_save_does_not_destroy_the_previous_checkpoint(tmp_path,
                                                              monkeypatch):
    path = tmp_path / "model.pt"
    save_checkpoint(str(path), torch.nn.Linear(1, 1), arch_name="x", epoch=1)
    good = path.read_bytes()

    import HGNDRecoGNN.training.checkpoint as ckpt_mod

    def boom(obj, where):
        Path(where).write_bytes(b"partial")      # a half-written temp file
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(ckpt_mod.torch, "save", boom)
    with pytest.raises(RuntimeError, match="killed mid-write"):
        save_checkpoint(str(path), torch.nn.Linear(1, 1), arch_name="x",
                        epoch=2)
    assert path.read_bytes() == good, "the epoch-1 checkpoint was damaged"
    assert not (tmp_path / "model.pt.tmp").exists(), "temp file left behind"
