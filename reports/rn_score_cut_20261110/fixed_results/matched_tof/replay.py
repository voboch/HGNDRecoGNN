"""Replay the prior even-file ToF comparison with the fixed tighter SMASH cut."""

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "first_hit_tof_20261110/energy_response"
CHOICE = HERE.parents[1] / "choice.json"


def module(name):
    spec = spec_from_file_location("tof_energy_" + name, SOURCE / f"{name}.py")
    loaded = module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


if __name__ == "__main__":
    raw = CHOICE.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "22c81979ca8064aed4248cb3d11bb4379048c8b2ea22a236688054e15c1a0880"
    )
    assert not (HERE / "bins.json").exists()
    cuts = json.loads(raw)["per_fold_seed_score_cuts"]
    module("analyze").main(HERE, cuts)
    module("plot").main(HERE)
