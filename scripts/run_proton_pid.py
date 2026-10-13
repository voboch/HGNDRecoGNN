#!/usr/bin/env python3
"""Inspect/extract BmnRoot DSTs, calibrate the slide PID, and select protons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("inspect", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--load-lib", action="append", default=[])
        p.add_argument("--root-init", type=Path,
                       help="Optional BmnRoot library-loading macro, if rootmap autoload is unavailable")
        p.add_argument("--max-events", type=int, default=100 if name == "inspect" else 0,
                       help="0 means the entire file; inspect defaults to 100 entries for a bounded schema check")
        p.add_argument("--require-mc", action="store_true")
        if name == "extract":
            p.add_argument("--output-prefix", type=Path, required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--pattern", default="*.root")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--tracks", type=Path, nargs="+", required=True)
    p.add_argument("--tof", choices=["400", "700"], required=True)
    p.add_argument("--p-edges", required=True,
                   help="Explicit comma-separated GeV/c bin edges; select using the occupancy QA")
    p.add_argument("--min-slice-tracks", type=int, required=True,
                   help="Required template statistics; documented as a convention and checked in QA")
    p.add_argument("--sample-description", required=True,
                   help="Development file scope, sample(s), run/geometry and why it was chosen")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select")
    p.add_argument("--tracks", type=Path, nargs="+", required=True)
    p.add_argument("--events", type=Path, nargs="+", required=True,
                   help="Event ledgers; include events with zero global tracks")
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--proton-radius", type=float, default=2.0,
                   help="Slide 23 convention: r_p<2; scan purity/efficiency before freezing")
    p.add_argument("--pion-veto", type=float, default=3.0,
                   help="Slide 23 convention: r_pi>3; scan its purity/efficiency tradeoff")
    p.add_argument("--all-vertices", action="store_true",
                   help="Disable reconstructed primary association; diagnostic inclusive sample")
    p.add_argument("--max-chi2-ndf", type=float)
    p.add_argument("--max-dca-cm", type=float)
    p.add_argument("--no-qa", action="store_true")
    p.add_argument("--allow-calibration-overlap", action="store_true",
                   help="Development closure only; permit calibration source files in the evaluation")
    return ap.parse_args()


def apply_selection(args):
    import numpy as np
    import pandas as pd
    from proton_pid.core import calibration_hash, read_tables, select_protons, write_json, KEY

    d = read_tables(args.tracks)
    events = pd.concat([pd.read_csv(p) for p in args.events], ignore_index=True)
    event_key = KEY[:2]
    if events.duplicated(event_key).any():
        raise ValueError("Duplicate event ledger rows")
    if not set(d.source_id).issubset(set(events.source_id)):
        raise ValueError("Missing source in the event denominator ledger")
    ledger_keys = pd.MultiIndex.from_frame(events[event_key])
    if not pd.MultiIndex.from_frame(d[event_key]).isin(ledger_keys).all():
        raise ValueError("Track event is absent from the denominator ledger")
    cal = json.loads(args.calibration.read_text())
    if args.proton_radius <= 0 or args.pion_veto <= 0:
        raise ValueError("PID distance cuts must be positive")
    if ((args.max_chi2_ndf is not None and args.max_chi2_ndf <= 0)
            or (args.max_dca_cm is not None and args.max_dca_cm <= 0)):
        raise ValueError("Quality-cut limits must be positive")
    overlap = set(d.source_id) & set(cal.get("source_ids", []))
    if overlap and not args.allow_calibration_overlap:
        raise ValueError("Calibration files occur in selection input; use held-out files or explicitly label a development closure")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists() or (out / "scored_tracks.csv.gz").exists():
        raise FileExistsError("Selection output exists; use a new run directory")
    scored = select_protons(d, cal, args.proton_radius, args.pion_veto,
                           not args.all_vertices, args.max_chi2_ndf, args.max_dca_cm)
    if not scored.pid_valid.any():
        raise ValueError("No valid combined PID tracks; check dE/dx, TOF and calibration support")
    selected = scored[scored.selected_proton]
    counts = selected.groupby(event_key).size().rename("n_selected_protons")
    events = events.set_index(event_key).join(counts).reset_index()
    events["n_selected_protons"] = events.n_selected_protons.fillna(0).astype(int)
    scored.to_csv(out / "scored_tracks.csv.gz", index=False)
    selected.to_csv(out / "protons.csv.gz", index=False)
    events.to_csv(out / "events.csv.gz", index=False)
    count = events.n_selected_protons.to_numpy(float)
    report = {"method": "parfenov_combined", "tof": cal["tof"],
              "calibration_file": str(args.calibration.resolve()),
              "calibration_sha256": calibration_hash(cal), "calibration_sample": cal["sample_description"],
              "input_tracks": [str(p.resolve()) for p in args.tracks],
              "input_events": [str(p.resolve()) for p in args.events],
              "cuts": {"positive_charge": True, "proton_radius_less_than": args.proton_radius,
                       "pion_radius_greater_than": args.pion_veto,
                       "from_vertex": not args.all_vertices, "max_chi2_ndf": args.max_chi2_ndf,
                       "max_dca_cm": args.max_dca_cm},
              "n_events": len(events), "n_tracks": len(d), "n_pid_valid": int(scored.pid_valid.sum()),
              "n_selected": len(selected), "yield_per_event": float(count.mean()) if len(count) else 0.,
              "yield_event_standard_error": float(count.std(ddof=1)/np.sqrt(len(count))) if len(count)>1 else None,
              "yield_caveat": "raw reconstructed candidate yield; includes fakes/secondaries; not efficiency-corrected",
              "uncertainty_caveat": "event standard error is diagnostic; production-file clustering must be measured before physics use",
              "truth_used_in_selection": False,
              "calibration_source_overlap": sorted(overlap),
              "calibration_truth_used": cal["calibration_provenance"],
              "support_note": "PID rejects momenta outside the common measured calibration centres; no extrapolation",
              "mass_hypothesis_note": "p=|rigidity| and Ekin=sqrt(p^2+m_p^2)-m_p assume charge magnitude one"}
    if "mc_pdg" in scored and scored.mc_pdg.ne(0).any():
        true = scored.mc_pdg.eq(2212)
        report["mc_pid_purity_all_selected"] = float(selected.mc_pdg.eq(2212).mean()) if len(selected) else None
        eligible = true & scored.pid_valid & scored.charge_sign.eq(1)
        if not args.all_vertices:
            eligible &= scored.from_vertex.eq(1)
        report["mc_pid_efficiency_given_valid_reconstructed_track"] = float(scored.loc[eligible, "selected_proton"].mean()) if eligible.any() else None
        report["mc_selected_primary_fraction"] = float((selected.mc_pdg.eq(2212) & selected.mc_mother.eq(-1)).mean()) if len(selected) else None
    write_json(out / "selection.json", report)
    if not args.no_qa:
        from proton_pid.plots import plot_selection
        plot_selection(scored, cal, report, out)
    print(json.dumps(report, indent=2))


def main():
    args = parse_args()
    if args.command in ("inspect", "extract"):
        from proton_pid.root_io import extract
        if args.max_events < 0:
            raise ValueError("--max-events must be nonnegative")
        if args.command == "inspect":
            with tempfile.TemporaryDirectory(prefix="bmn-proton-inspect-") as td:
                meta = extract(args.input, Path(td)/"inspect", args.max_events,
                               args.require_mc, args.load_lib, args.root_init)
            meta["combined_pid_ready"] = bool(meta["tracks"] and meta["dedx_positive"]
                                               and (meta["tof400_valid"] or meta["tof700_valid"]))
            meta["warning"] = ("Calibration and dictionary compatibility still need a single-file batch smoke"
                               if meta["combined_pid_ready"] else
                               "Combined PID unavailable: inspect dE/dx task, run period and TOF branches")
            print(json.dumps(meta, indent=2))
            if not meta["combined_pid_ready"]:
                return 2
        else:
            meta = extract(args.input, args.output_prefix, args.max_events,
                           args.require_mc, args.load_lib, args.root_init)
            print(json.dumps(meta, indent=2))
    elif args.command == "manifest":
        # Record files, not inferred event counts. ROOT schema inspection is a
        # separate first job; .root can name transport, DIGI or DST files.
        files = sorted(args.input_dir.glob(args.pattern))
        files = [p.resolve() for p in files if p.is_file()]
        if not files:
            raise ValueError("No input files matched")
        if args.output.exists():
            raise FileExistsError(args.output)
        rows = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                for p in files]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(str(p) for p in files)+"\n")
        stamp = {"sample": args.sample, "n_files": len(files), "files": rows,
                 "manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
        Path(str(args.output)+".json").write_text(json.dumps(stamp, indent=2)+"\n")
        print(f"{len(files)} files -> {args.output}")
    elif args.command == "calibrate":
        from proton_pid.core import fit_calibration, read_tables, write_json
        from proton_pid.plots import plot_calibration
        if args.output.exists():
            raise FileExistsError(args.output)
        d = read_tables(args.tracks)
        cal = fit_calibration(d, args.tof, [float(x) for x in args.p_edges.split(",")],
                              args.min_slice_tracks, args.sample_description)
        cal["input_tables"] = [str(p.resolve()) for p in args.tracks]
        cal["source_ids"] = sorted(d.source_id.unique().tolist())
        # Attach the actual extraction environment rather than inferring a
        # production tag from a directory name.
        cal["extraction_metadata"] = []
        for p in args.tracks:
            meta = Path(str(p).replace("_tracks.csv.gz", "_meta.json"))
            if meta.exists():
                cal["extraction_metadata"].append(json.loads(meta.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, cal)
        plot_calibration(cal, args.output.parent)
        print("Calibration written; inspect fit means, widths and correlations before freezing:", args.output)
    elif args.command == "select":
        apply_selection(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileExistsError, RuntimeError, ImportError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(2)
