"""Target-only no-floor patches for the accepted SMASH candidate packs.

The graph and observations remain frozen. The native 0.3 GeV condition is
replayed before removing its energy clause; PDG, Side and ToF stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .graph_dataset import _dfToF
from .smash_transfer import _ordered
from ..analysis.label_energy_scan import threshold_ledger

TARGET_FIELDS = frozenset(("label", "energy"))


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def candidate_keys(pack):
    """Immutable (Row, side, local ClusterID), in the prediction row order."""
    ptr = pack["cluster_ptr"].cpu().numpy()
    lengths = np.diff(ptr)
    return np.column_stack((np.repeat(pack["row"].cpu().numpy(), lengths),
                            np.repeat(pack["side"].cpu().numpy(), lengths),
                            np.arange(int(ptr[-1])) - np.repeat(ptr[:-1], lengths))).astype(np.int64)


def make_patch(frame, pack):
    """Aggregate targets on accepted membership, with exact native replay."""
    frame, rows, sides = _ordered(frame)
    lengths = np.diff(pack["node_ptr"].cpu().numpy())
    ptr = pack["cluster_ptr"].cpu().numpy()
    local = pack["cluster"].cpu().numpy()
    if (lengths.sum() != len(frame) or np.any(lengths <= 0)
            or not np.array_equal(np.repeat(pack["row"].cpu().numpy(), lengths), rows)
            or not np.array_equal(np.repeat(pack["side"].cpu().numpy(), lengths), sides)
            or np.any(local < 0) or np.any(local >= np.repeat(np.diff(ptr), lengths))):
        raise ValueError("Observation order/membership differs from the accepted pack")
    cluster = local + np.repeat(ptr[:-1], lengths)
    if not np.array_equal(np.unique(cluster), np.arange(len(pack["label"]))):
        raise ValueError("Missing accepted cluster")
    ek = frame.Ekin.to_numpy(float)
    distance = np.linalg.norm(frame[["fX", "fY", "fZ"]].to_numpy(float)/100., axis=1)
    time = frame.fTime.to_numpy(float)
    # The inherited +/-0.3 ns ToF gate is fixed; only the energy clause changes.
    prompt = ((np.abs(frame.PDG.to_numpy()) == 2112) & frame.Side.isin([5, 6]).to_numpy()
              & (ek >= _dfToF(distance, time + .3)) & (ek <= _dfToF(distance, time - .3)))
    ledger, targets = threshold_ledger(ek, prompt, cluster, [.3, None])
    tensors = {"candidate_keys": torch.from_numpy(candidate_keys(pack))}
    for floor, prefix in ((.3, "native"), (None, "disabled")):
        label, energy = targets[floor]
        tensors[prefix+"_label"] = torch.as_tensor(label, dtype=pack["label"].dtype)
        tensors[prefix+"_energy"] = torch.as_tensor(energy, dtype=pack["energy"].dtype)
    for field in TARGET_FIELDS:
        if tensor_sha(tensors["native_"+field]) != tensor_sha(pack[field]):
            raise ValueError("Native target replay failed: " + field)
    common = tensors["native_label"].bool()
    if (torch.any(tensors["disabled_label"] < tensors["native_label"])
            or not torch.equal(tensors["disabled_energy"][common], pack["energy"][common])):
        raise ValueError("No-floor patch is not a target-preserving superset")
    return tensors, {"ledger": ledger, "native_replay_exact": True,
                     "candidate_keys_sha256": tensor_sha(tensors["candidate_keys"])}


def apply_patch(pack, patch):
    """Replace exactly two tensors after checking native targets and keys."""
    if not np.array_equal(candidate_keys(pack), patch["candidate_keys"].cpu().numpy()):
        raise ValueError("Patch addresses another candidate order")
    for field in TARGET_FIELDS:
        if tensor_sha(pack[field]) != tensor_sha(patch["native_"+field]):
            raise ValueError("Patch native reference changed: " + field)
    before = {k: tensor_sha(v) for k, v in pack.items() if k not in TARGET_FIELDS}
    out = dict(pack)
    out.update(label=patch["disabled_label"].clone(), energy=patch["disabled_energy"].clone())
    if before != {k: tensor_sha(v) for k, v in out.items() if k not in TARGET_FIELDS}:
        raise ValueError("Target patch changed an observation/structural tensor")
    if (out["label"].shape != pack["label"].shape or out["energy"].shape != pack["energy"].shape
            or out["label"].dtype != pack["label"].dtype or out["energy"].dtype != pack["energy"].dtype
            or not torch.isfinite(out["energy"]).all() or torch.any(out["energy"] < 0)
            or not torch.all((out["label"] == 0) | (out["label"] == 1))
            or torch.any(out["label"] < pack["label"])
            or not torch.equal(out["energy"][pack["label"].bool()], pack["energy"][pack["label"].bool()])):
        raise ValueError("Malformed target patch")
    return out


class PatchStore:
    """A compact target manifest, bound to every accepted source-pack SHA."""
    def __init__(self, directory, expected_manifest_sha256, sha):
        self.root, self.sha = Path(directory), sha
        path = self.root / "summary.json"
        if sha(path) != expected_manifest_sha256:
            raise ValueError("Patch manifest changed")
        self.manifest = json.loads(path.read_text())
        if self.manifest["state"] != "complete" or self.manifest["schema"] != "smash-label-patches-v1":
            raise ValueError("Incomplete patch preparation")

    def load(self, cid, pack_sha256):
        record = self.manifest["files"][cid]
        if pack_sha256 not in record["native_pack_sha256"].values():
            raise ValueError("Native pack was not bound by target preparation")
        path = self.root / record["path"]
        if self.sha(path) != record["sha256"]:
            raise ValueError("Target patch changed")
        return torch.load(path, weights_only=True, map_location="cpu")
