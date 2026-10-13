#!/usr/bin/env python3
"""Reconstruct a transported SMASH/DIGI file with the installed BmnRoot macro.

This stage is optional: a DST containing BmnGlobalTrack should go directly to
proton_pid.py inspect/extract. Generator-only particles and HGND CSV exports
do not contain the charged-detector digitisation needed for reconstruction.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--macro", type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run reconstruction inside a Slurm allocation")
    if args.max_events < 0 or not args.input.is_file():
        raise ValueError("Bad input or event limit")
    if args.output.exists():
        raise FileExistsError(args.output)
    vmc = os.environ.get("VMCWORKDIR")
    if not vmc:
        raise RuntimeError("Source the installed BmnRoot build/config.sh")
    macro = args.macro or Path(vmc)/"macro/run/run_reco_bmn.C"
    if not macro.is_file():
        raise FileNotFoundError(macro)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # These are C++ string literals passed as a subprocess argv element, not
    # interpolated shell code. The four-argument form is Start Guide §4.5.
    call = str(macro)+"("+json.dumps(str(args.input.resolve()))+","+json.dumps(str(args.output.resolve()))+",0,"+str(args.max_events)+")"
    subprocess.run(["root", "-l", "-b", "-q", call], check=True)
    import ROOT
    f = ROOT.TFile.Open(str(args.output), "READ")
    if not f or f.IsZombie():
        raise RuntimeError("Reconstruction did not produce a readable file")
    tree = f.Get("bmndata")
    if not tree or not tree.GetBranch("BmnGlobalTrack") or tree.GetEntries() <= 0:
        raise RuntimeError("Reconstruction output lacks a nonempty global-track DST tree")
    print("DST entries:", tree.GetEntries())
    f.Close()


if __name__ == "__main__":
    main()
