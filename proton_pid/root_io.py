"""Read reconstructed DST quantities through the installed BmnRoot dictionaries.

No dependency on numpy/pandas here: extraction uses the Python paired with the
cluster ROOT build. Statistical calibration can use a separate Python runtime.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path


TRACK_FIELDS = ["source_id", "source_path", "entry", "Row", "event_id", "period",
                "b_fm", "track_index", "inner_index", "charge_sign", "q_over_p",
                "p", "px", "py", "pz", "pt", "theta_rad", "phi_rad",
                "dedx", "charge_estimate", "from_vertex", "inner_hits",
                "chi2_ndf", "dca_cm", "vertex_chi2", "mc_index", "mc_pdg",
                "mc_mother", "mc_p", "mc_px", "mc_py", "mc_pz", "mc_match_fraction"]
for _tof in (400, 700):
    TRACK_FIELDS += [f"tof{_tof}_index", f"tof{_tof}_valid", f"m2_{_tof}",
                     f"beta_{_tof}", f"tof{_tof}_time_ns", f"tof{_tof}_length_cm"]
EVENT_FIELDS = ["source_id", "source_path", "entry", "Row", "event_id", "period",
                "b_fm", "vertex_tracks", "n_global", "n_primary_p_mc", "n_primary_n_mc"]
TRUTH_FIELDS = ["source_id", "entry", "Row", "event_id", "b_fm", "mc_index",
                "mc_pdg", "mc_px", "mc_py", "mc_pz", "mc_vx", "mc_vy", "mc_vz"]


def member(obj, names, default=float("nan")):
    for name in names:
        fun = getattr(obj, name, None)
        if callable(fun):
            return fun()
    return default


def tree_object(tree, names):
    for name in names:
        if tree.GetBranch(name):
            for attr in (name, name.rstrip(".")):
                try:
                    value = getattr(tree, attr)
                    if value or hasattr(value, "GetEntriesFast"):
                        return value
                except AttributeError:
                    pass
    return None


def at(array, index):
    return array.At(index) if array and 0 <= index < array.GetEntriesFast() else None


def mc_array(tree):
    info = tree_object(tree, ("BmnMCInfo.", "BmnMCInfo"))
    if info and hasattr(info, "GetMCTracks"):
        return info.GetMCTracks()
    return tree_object(tree, ("MCTrack", "BmnMCInfo.fMCTracks"))


def tof_values(track, hits, tof):
    suffix = 1 if tof == 400 else 2
    getter = getattr(track, f"GetTof{suffix}HitIndex", None)
    if not getter:
        raise ValueError(f"BmnGlobalTrack lacks GetTof{suffix}HitIndex")
    index = int(getter())  # no argument: kCommon, avoids species-biased matching
    hit = at(hits, index)
    length = float(member(hit, ("GetLength",))) if hit else float("nan")
    time = float(member(hit, ("GetTimeStamp",))) if hit else float("nan")
    beta = length / (29.9792458 * time) if length > 0 and time > 0 else float("nan")
    mass_getter = getattr(track, "GetMass2", None)
    if mass_getter:
        m2 = float(mass_getter(suffix))
    else:
        # Compatibility for older builds: matched hit path and flight time
        # define m^2. GetLength() on the global track stops at the last tracker
        # hit and must not replace the TOF hit's path length.
        par = track.GetParamFirst()
        qp = par.GetQp()
        p = 1 / abs(qp) if qp else float("nan")
        m2 = p*p*(1/(beta*beta)-1) if beta > 0 else float("nan")
    # beta > 1 and small negative m^2 may result from timing resolution;
    # clipping them would deform pion templates. Reject only unavailable data.
    valid = bool(hit and math.isfinite(m2) and m2 > -999 and length > 0 and time > 0)
    return {f"tof{tof}_index": index, f"tof{tof}_valid": int(valid),
            f"m2_{tof}": m2 if valid else float("nan"), f"beta_{tof}": beta,
            f"tof{tof}_time_ns": time, f"tof{tof}_length_cm": length}


def track_row(track, inner, i, identity, mc, tof400, tof700, match):
    par = track.GetParamFirst()
    qp = float(par.GetQp())
    if not math.isfinite(qp) or qp == 0:
        return None
    p = 1 / abs(qp)  # rigidity; unit-charge proton hypothesis
    tx, ty = float(par.GetTx()), float(par.GetTy())
    pz = p / math.sqrt(1 + tx*tx + ty*ty)
    px, py = tx*pz, ty*pz
    mc_index = int(member(inner, ("GetTrkID",), -1)) if inner else -1
    truth = at(mc, mc_index)
    ndf = float(member(track, ("GetNDF",), 0))
    if not inner or not hasattr(inner, "FromVertex"):
        raise ValueError("Inner track / FromVertex missing: primary association cannot be evaluated")
    row = dict(identity)
    row.update({"track_index": i,
                "inner_index": int(member(track, ("GetInnerTrackIndex", "GetGemTrackIndex"), -1)),
                "charge_sign": 1 if qp > 0 else -1, "q_over_p": qp,
                "p": p, "px": px, "py": py, "pz": pz, "pt": math.hypot(px, py),
                "theta_rad": math.atan2(math.hypot(px, py), pz), "phi_rad": math.atan2(py, px),
                "dedx": float(member(track, ("GetDedx",))),
                "charge_estimate": member(track, ("GetQ",), -1),
                "from_vertex": int(bool(inner.FromVertex())),
                "inner_hits": int(member(inner, ("GetNStsHits", "GetNHits"), -1)),
                "chi2_ndf": float(track.GetChi2())/ndf if ndf > 0 else float("nan"),
                "dca_cm": member(track, ("GetDCAInVertex",)),
                "vertex_chi2": member(track, ("GetChi2InVertex",)),
                "mc_index": mc_index, "mc_pdg": int(truth.GetPdgCode()) if truth else 0,
                "mc_mother": int(truth.GetMotherId()) if truth else -999,
                "mc_p": math.sqrt(sum(float(getattr(truth, k)())**2
                                      for k in ("GetPx", "GetPy", "GetPz"))) if truth else float("nan"),
                "mc_px": truth.GetPx() if truth else float("nan"),
                "mc_py": truth.GetPy() if truth else float("nan"),
                "mc_pz": truth.GetPz() if truth else float("nan"),
                "mc_match_fraction": member(match, ("GetTrueOverAllHitsRatio",)) if match else float("nan")})
    row.update(tof_values(track, tof400, 400))
    row.update(tof_values(track, tof700, 700))
    return row


def open_root(path, load_libs=(), root_init=None):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    for lib in load_libs:
        if ROOT.gSystem.Load(lib) < 0:
            raise RuntimeError(f"Cannot load ROOT library {lib}")
    if root_init:
        ROOT.gROOT.Macro(str(Path(root_init).resolve()))
    f = ROOT.TFile.Open(str(path), "READ")
    if not f or f.IsZombie():
        raise ValueError(f"Cannot open {path}")
    tree = f.Get("bmndata")
    if not tree:
        raise ValueError("No bmndata tree")
    if not tree.GetBranch("BmnGlobalTrack"):
        raise ValueError("No BmnGlobalTrack: input may be DIGI/transport rather than a reconstructed DST")
    if not tree.GetBranch("StsVector"):
        raise ValueError("No StsVector: this adapter targets the documented vector-tracking DST; inspect this build")
    return ROOT, f, tree


def extract(path, prefix, max_events=0, require_mc=False, load_libs=(), root_init=None):
    """One file per invocation; all entries go to an event denominator ledger."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [Path(str(prefix)+s) for s in ("_tracks.csv.gz", "_events.csv.gz", "_truth.csv.gz", "_meta.json")]
    if any(p.exists() or Path(str(p)+".part").exists() for p in outputs):
        raise FileExistsError("Output prefix exists; choose a new run directory")
    root, f, tree = open_root(path, load_libs, root_init)
    source = os.path.realpath(str(path))
    source_id = hashlib.sha256(source.encode()).hexdigest()[:20]
    n_entries = int(tree.GetEntries())
    n = min(max_events, n_entries) if max_events > 0 else n_entries
    meta = {"input": source, "source_id": source_id,
            "input_bytes": Path(path).stat().st_size, "input_mtime_ns": Path(path).stat().st_mtime_ns,
            "entries_in_file": n_entries,
            "entries_processed": n, "root_version": root.gROOT.GetVersion(),
            "vmcworkdir": os.environ.get("VMCWORKDIR"), "bmnroot_config": os.environ.get("BMNROOT_CONFIG"),
            "max_events": max_events, "mc_required": require_mc,
            "branches": [{"name": b.GetName(), "class": b.GetClassName()}
                         for b in tree.GetListOfBranches()],
            "tracks": 0, "dedx_positive": 0, "tof400_valid": 0, "tof700_valid": 0,
            "matched_mc_tracks": 0, "invalid_rigidity": 0, "mc_available": None,
            "entry_key": "(source_id, entry); Row == entry, zero-based, matching the "
                         "HGND CSVs written by ncx Convert2Csv.C",
            "mc_matching": "StsVector[global inner index].GetTrkID(); never global.GetRefIndex()",
            "tof_hypothesis": "kCommon", "completed": False}
    parts = [Path(str(p)+".part") for p in outputs[:3]]
    with gzip.open(parts[0], "wt", newline="") as ft, gzip.open(parts[1], "wt", newline="") as fe, gzip.open(parts[2], "wt", newline="") as fm:
        tw, ew, mw = [csv.DictWriter(handle, fields) for handle, fields in
                      ((ft, TRACK_FIELDS), (fe, EVENT_FIELDS), (fm, TRUTH_FIELDS))]
        for writer in (tw, ew, mw):
            writer.writeheader()
        for entry in range(n):
            if tree.GetEntry(entry) <= 0:
                raise ValueError(f"Failed GetEntry({entry})")
            tracks = tree_object(tree, ("BmnGlobalTrack",))
            inners = tree_object(tree, ("StsVector",))
            vertex = tree_object(tree, ("MpdVertex.", "MpdVertex"))
            header = tree_object(tree, ("DstEventHeader.", "DstEventHeader"))
            mc = mc_array(tree)
            mc_present = mc is not None
            if meta["mc_available"] is None:
                meta["mc_available"] = mc_present
            if require_mc and not mc_present:
                raise ValueError("MC tracks absent: cannot produce calibration/efficiency truth ledger")
            identity = {"source_id": source_id, "source_path": source, "entry": entry,
                        # Row is the zero-based TTree entry, exactly as Convert2Csv.C writes it
                        # (`hits.row({ev, ...})` with ev from 0). Writing entry+1
                        # here joined every proton row to the NEXT event's HGND
                        # hits, silently pairing the charged arm with the wrong
                        # neutrons.
                        "Row": entry, "event_id": int(member(header, ("GetEventId",), -1)),
                        "period": int(member(header, ("GetPeriodId",), -1)),
                        "b_fm": float(member(header, ("GetB",), -1))}
            countp = countn = 0
            if mc:
                for imc in range(mc.GetEntriesFast()):
                    tr = mc.At(imc)
                    pdg = int(tr.GetPdgCode())
                    if tr.GetMotherId() == -1 and pdg in (2212, 2112):
                        countp += pdg == 2212
                        countn += pdg == 2112
                        mr = {k: identity[k] for k in ("source_id", "entry", "Row", "event_id", "b_fm")}
                        mr.update({"mc_index": imc, "mc_pdg": pdg,
                                   "mc_px": tr.GetPx(), "mc_py": tr.GetPy(), "mc_pz": tr.GetPz(),
                                   "mc_vx": tr.GetStartX(), "mc_vy": tr.GetStartY(), "mc_vz": tr.GetStartZ()})
                        mw.writerow(mr)
            n_global = tracks.GetEntriesFast() if tracks else 0
            ev = dict(identity)
            ev.update({"vertex_tracks": int(member(vertex, ("GetNTracks",), 0)), "n_global": n_global,
                       "n_primary_p_mc": countp if mc_present else -1,
                       "n_primary_n_mc": countn if mc_present else -1})
            ew.writerow(ev)  # includes zero-track / failed-vertex events
            hits400 = tree_object(tree, ("BmnTof400Hit",))
            hits700 = tree_object(tree, ("BmnTof700Hit",))
            matches = tree_object(tree, ("BmnGlobalTrackMatch",))
            for i in range(n_global):
                tr = tracks.At(i)
                ii = int(member(tr, ("GetInnerTrackIndex", "GetGemTrackIndex"), -1))
                row = track_row(tr, at(inners, ii), i, identity, mc, hits400, hits700, at(matches, i))
                if row is None:
                    meta["invalid_rigidity"] += 1
                    continue
                tw.writerow(row)
                meta["tracks"] += 1
                meta["dedx_positive"] += bool(math.isfinite(row["dedx"]) and row["dedx"] > 0)
                meta["matched_mc_tracks"] += row["mc_pdg"] != 0
                for tof in (400, 700):
                    meta[f"tof{tof}_valid"] += row[f"tof{tof}_valid"]
    f.Close()
    for part, final in zip(parts, outputs[:3]):
        part.rename(final)
    meta["completed"] = True
    outputs[3].write_text(json.dumps(meta, indent=2)+"\n")
    return meta
