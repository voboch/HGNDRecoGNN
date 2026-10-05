"""Regression tests for resumable training state."""

from pathlib import Path
import sys

import torch


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from HGNDRecoGNN import device as device_mod  # noqa: E402
from HGNDRecoGNN.training import train as train_mod  # noqa: E402


def test_fit_restores_best_state_before_resuming(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 15, 25], gamma=0.1,
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "next_epoch": 1,
            "min_loss": 0.25,
            "best_epoch": 0,
        },
        tmp_path / "optim.pt",
    )

    monkeypatch.setattr(train_mod, "train_epoch", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(train_mod, "eval_epoch", lambda *args, **kwargs: 0.5)

    cfg = train_mod.TrainConfig(
        epochs=2,
        start_epoch=1,
        device="cpu",
        checkpoint_dir=str(tmp_path),
        verbose=False,
    )
    result = train_mod.fit(
        model,
        train_loader=[],
        test_loader=[],
        cfg=cfg,
        plan=device_mod.DeviceMap(main=torch.device("cpu")),
    )

    assert result["best_loss"] == 0.25
    assert result["best_epoch"] == 0
    saved = torch.load(tmp_path / "optim.pt", map_location="cpu",
                       weights_only=False)
    assert saved["next_epoch"] == 2
    assert saved["min_loss"] == 0.25
    assert saved["best_epoch"] == 0
