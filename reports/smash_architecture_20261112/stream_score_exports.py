"""Stream already published score outputs over one SSH hop; never run a model."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile

scope = os.environ["HGND_EXPORT_SCOPE"]
if scope not in ("even", "odd"):
    raise ValueError(scope)
base = Path("/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts")
campaign = Path("/scratch/vbocharnikov/hgnd/experiments/smash_architecture_20261112_2d31e3e25c45")
pattern = re.compile(r"score_" + scope + r"_(\d+)_(\d+)\.json")
publications = {}
for path in base.glob("score_" + scope + "_*.json"):
    match = pattern.fullmatch(path.name)
    if match is None:
        continue
    task = int(match[1])
    if task in publications:
        raise ValueError("Duplicate score task: " + str(task))
    raw = path.read_bytes()
    record = json.loads(raw)
    bundle = Path(record["bundle_dir"])
    if record["state"] != "complete" or bundle != campaign / path.stem:
        raise ValueError("Unaccepted publication shape: " + str(path))
    manifest = bundle / "manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != record["manifest_sha256"]:
        raise ValueError("Publication manifest mismatch: " + str(path))
    publications[task] = (path, raw, bundle)
if set(publications) != set(range(18)):
    raise ValueError("Expected all 18 published tasks; found " + str(sorted(publications)))

pins = "".join(hashlib.sha256(raw).hexdigest() + "  " + path.name + "\n"
               for path, raw, _ in publications.values()).encode()
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    info = tarfile.TarInfo("receipt_sha256.txt")
    info.size = len(pins)
    archive.addfile(info, io.BytesIO(pins))
    for task in sorted(publications):
        path, raw, bundle = publications[task]
        info = tarfile.TarInfo(path.name)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
        for name in ("manifest.json", "inputs.tar"):
            archive.add(bundle / name, arcname=path.stem + "/" + name, recursive=False)
