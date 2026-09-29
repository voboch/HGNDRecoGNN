#!/usr/bin/env python3
"""Truth nucleon spectra and reconstructed neutron spectra in the HGND acceptance.

The update note quotes hardness ratios without ever showing the spectra they are
ratios of.  These are those spectra.

Truth uncertainties resample production jobs, using the per-job histograms the
reduction writes.  Reconstructed uncertainties resample events: the v3 caches'
source file list cannot be reconstructed -- their graph counts exceed the row
space of the directory they appear to come from -- so the job label is not
available for those predictions.  The two error models are named in the
captions rather than silently mixed.

Follows docs/plotting_style_protocol.md.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_b_full import apply_style, assert_contained, save, FULL, DATASETS, REF

# The energy axis is linear. A log axis is right when the interesting structure
# spans decades; here the hardness window is searched over roughly 1 to 3 GeV,
# and a log axis compresses exactly that range into a third of the panel while
# spending the rest on a sub-GeV region no threshold is placed in. The vertical
# band marks the searched range so the spectra can be read against the choice
# they justify. The y axis stays logarithmic where the yield spans more than
# two decades, per docs/plotting_style_protocol.md section 5.
SCAN_LO, SCAN_HI = 1.0, 3.0


def mark_scan_range(ax, label=True, y=0.04, va="bottom"):
    ax.axvspan(SCAN_LO, SCAN_HI, color="#2b2b2b", alpha=0.07, lw=0, zorder=0)
    if label:
        ax.text(0.5 * (SCAN_LO + SCAN_HI), y, r"$R_{\mathrm{thr}}$ searched",
                transform=ax.get_xaxis_transform(), ha="center", va=va,
                fontsize=6.0, color="#69727C")


def truth_spectra(hist_dir):
    """Per-job (species, Ekin) histograms inside the front-face acceptance."""
    out = {}
    for tag in DATASETS:
        p = Path(hist_dir) / f"{tag}_filehist.npz"
        if not p.exists():
            continue
        z = np.load(p)
        keys = [k for k in z.files if k.endswith("|acc")]
        per = np.stack([z[k] for k in keys])        # (job, species, Ekin)
        out[tag] = {"per_job": per, "edges": z["_e_edges"]}
    return out


def _band(per_job, edges, rng, n_boot=400):
    """Area-normalised mean spectrum over jobs, with a job-level bootstrap.

    `per_job` is (job, Ekin). Each job is normalised before averaging so that a
    job contributing more nucleons does not weight the shape.
    """
    per_job = np.asarray(per_job, float)
    n_job = len(per_job)
    tot = per_job.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    dens = per_job / tot
    mean = dens.mean(axis=0)
    idx = rng.integers(0, n_job, (n_boot, n_job))
    boot = dens[idx].mean(axis=1)
    return mean, boot.std(axis=0, ddof=1)


def fig_truth(hist_dir, out_dir, rng, emin=0.15, emax=5.0, min_counts=500):
    sp = truth_spectra(hist_dir)
    if REF not in sp:
        raise SystemExit("truth spectra need the reference sample")
    edges = sp[REF]["edges"]
    ctr = np.sqrt(np.maximum(edges[:-1], 1e-9) * edges[1:])   # log-bin centres
    wid = np.diff(edges)
    keep = (ctr > emin) & (ctr < emax) & (wid > 0)
    # a bin the reference barely populates carries an error that compresses the
    # whole ratio panel; require enough entries for the point to mean something
    ref_counts = sp[REF]["per_job"].sum(axis=0)          # (species, Ekin)

    fig, ax = plt.subplots(2, 2, figsize=(FULL, FULL / 1.45), sharex="col",
                           gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.07,
                                        "wspace": 0.28})
    payload = {}
    ratio_span = []
    panels = {}
    for col, (sidx, name) in enumerate(((0, "neutrons"), (1, "protons"))):
        keep_s = keep & (ref_counts[sidx] >= min_counts)
        dens, err = {}, {}
        for tag, st in DATASETS.items():
            if tag not in sp:
                continue
            m, e = _band(sp[tag]["per_job"][:, sidx, :], edges, rng)
            dens[tag], err[tag] = m / wid, e / wid
            _k = keep_s & (dens[tag] > 0)
            ax[0, col].errorbar(ctr[_k], dens[tag][_k], yerr=err[tag][_k],
                                color=st["color"], ls=st["ls"], marker=st["marker"],
                                ms=2.6, mfc="none", mew=0.7, elinewidth=0.6,
                                capsize=1.2, label=st["label"])
        ax[0, col].set_yscale("log")
        # label only the panel without a legend in that corner
        mark_scan_range(ax[0, col], label=(col == 1))
        ax[0, col].set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}E_{\mathrm{kin}}$  [GeV$^{-1}$]")
        ax[0, col].set_title(f"primary {name}, HGND front-face acceptance", fontsize=7)
        lo = np.concatenate([(dens[t] - err[t])[keep_s] for t in dens])
        hi = np.concatenate([(dens[t] + err[t])[keep_s] for t in dens])
        lo = lo[lo > 0]
        ax[0, col].set_ylim(lo.min() * 0.6, hi.max() * 1.7)
        if col == 0:
            ax[0, col].legend(loc="lower left", fontsize=6.4)
        fin = []
        for tag, st in DATASETS.items():
            if tag not in dens or tag == REF:
                continue
            r = np.divide(dens[tag], dens[REF], out=np.full_like(dens[tag], np.nan),
                          where=dens[REF] > 0)
            re = r * np.sqrt((err[tag] / np.maximum(dens[tag], 1e-30)) ** 2 +
                             (err[REF] / np.maximum(dens[REF], 1e-30)) ** 2)
            ax[1, col].errorbar(ctr[keep_s], r[keep_s], yerr=re[keep_s],
                                color=st["color"], ls=st["ls"], marker=st["marker"],
                                ms=2.6, mfc="none", mew=0.7, elinewidth=0.6,
                                capsize=1.2)
            fin += [v for v in np.concatenate([(r + re)[keep_s], (r - re)[keep_s]])
                    if np.isfinite(v)]
            payload[f"{name}_{tag}_over_ref"] = r[keep_s].tolist()
        ax[1, col].axhline(1.0, color="#2b2b2b", lw=0.8)
        mark_scan_range(ax[1, col], label=False)
        ax[1, col].set_xlabel(r"$E_{\mathrm{kin}}$  [GeV]")
        ax[1, col].set_ylabel(r"ratio to $S_{\mathrm{pot}}=0$", fontsize=7)
        ratio_span += fin
        panels[col] = None
        ax[0, col].set_xlim(emin, emax); ax[1, col].set_xlim(emin, emax)

    # shared ratio limits across the row, so the species compare by eye
    m = max(abs(np.array(ratio_span) - 1).max(), 0.02) * 1.12 if ratio_span else 0.2
    for col in (0, 1):
        ax[1, col].set_ylim(1 - m, 1 + m)
    for i in range(2):
        for j in range(2):
            assert_contained(ax[i, j], f"truth_{i}{j}")
    save(fig, out_dir, "fig_truth_np_spectra")
    payload["e_centres_note"] = "per-species masks differ; see arrays"
    return payload


def fig_np_relation(hist_dir, out_dir, rng, emin=0.15, emax=5.0, min_counts=500):
    """Where in energy the isovector signal lives, and what that implies.

    The hardness window is a choice of where to split the spectrum, so the
    quantity that should drive it is the relation between the two species as a
    function of energy, not a round number.  Left: n/p against energy for each
    sample.  Right: the same relative to the zero-potential sample, which is the
    isovector signal itself and shows which energies carry it.
    """
    sp = truth_spectra(hist_dir)
    edges = sp[REF]["edges"]
    ctr = np.sqrt(np.maximum(edges[:-1], 1e-12) * edges[1:])
    ref_counts = sp[REF]["per_job"].sum(axis=0)
    keep = ((ctr > emin) & (ctr < emax)
            & (ref_counts[0] >= min_counts) & (ref_counts[1] >= min_counts))

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, FULL / 2.6),
                               gridspec_kw={"wspace": 0.26})
    npr, npe = {}, {}
    for tag, st in DATASETS.items():
        if tag not in sp:
            continue
        pj = sp[tag]["per_job"]
        n_job = len(pj)
        with np.errstate(divide="ignore", invalid="ignore"):
            per = pj[:, 0, :] / np.where(pj[:, 1, :] > 0, pj[:, 1, :], np.nan)
        npr[tag] = np.nanmean(per, axis=0)
        idx = rng.integers(0, n_job, (400, n_job))
        npe[tag] = np.nanmean(per[idx], axis=1).std(axis=0, ddof=1)
        a.errorbar(ctr[keep], npr[tag][keep], yerr=npe[tag][keep], color=st["color"],
                   ls=st["ls"], marker=st["marker"], ms=2.6, mfc="none", mew=0.7,
                   elinewidth=0.6, capsize=1.2, label=st["label"])
    mark_scan_range(a, label=False)
    a.set_xlabel(r"$E_{\mathrm{kin}}$  [GeV]")
    a.set_ylabel(r"$n/p$  in the HGND acceptance")
    a.legend(loc="upper left", fontsize=6.4)
    a.set_title("nucleon ratio against energy", fontsize=7.5)
    v = np.concatenate([(npr[t] + npe[t])[keep] for t in npr]
                       + [(npr[t] - npe[t])[keep] for t in npr])
    v = v[np.isfinite(v)]
    pad = (v.max() - v.min()) * 0.10
    a.set_ylim(v.min() - pad, v.max() + pad); a.set_xlim(emin, emax)

    fin = []
    for tag, st in DATASETS.items():
        if tag not in npr or tag == REF:
            continue
        r = npr[tag] / npr[REF]
        re = r * np.sqrt((npe[tag] / npr[tag]) ** 2 + (npe[REF] / npr[REF]) ** 2)
        b.errorbar(ctr[keep], r[keep], yerr=re[keep], color=st["color"], ls=st["ls"],
                   marker=st["marker"], ms=2.6, mfc="none", mew=0.7,
                   elinewidth=0.6, capsize=1.2, label=st["label"])
        fin += [x for x in np.concatenate([(r + re)[keep], (r - re)[keep]])
                if np.isfinite(x)]
    b.axhline(1.0, color="#2b2b2b", lw=0.8)
    mark_scan_range(b)
    b.set_xlabel(r"$E_{\mathrm{kin}}$  [GeV]")
    b.set_ylabel(r"$(n/p)$ relative to $S_{\mathrm{pot}}=0$")
    b.set_title("the isovector signal, against energy", fontsize=7.5)
    m = max(abs(np.array(fin) - 1).max(), 0.01) * 1.15
    b.set_ylim(1 - m, 1 + m); b.set_xlim(emin, emax)
    for ax_, nm in ((a, "npr_a"), (b, "npr_b")):
        assert_contained(ax_, nm)
    save(fig, out_dir, "fig_np_spectra_relation")
    return {"e_centres": ctr[keep].tolist(),
            "np_ratio": {t: npr[t][keep].tolist() for t in npr}}


def reco_spectra(pred_dir, purity=0.7):
    from reco_experiment import purity_locked_threshold
    out = {}
    for tag in DATASETS:
        p = Path(pred_dir) / tag / "pred_clusters_smash.pkl"
        if not p.exists():
            continue
        c = pd.read_pickle(p)
        t, pur, _ = purity_locked_threshold(c, purity)
        out[tag] = {"sel": c[c.cl_score > t], "threshold": t, "purity": pur,
                    "n_events": int(c.Row.nunique())}
    return out


def fig_reco(pred_dir, out_dir, rng, emin=0.3, emax=5.0, n_boot=300,
             min_counts=300):
    sp = reco_spectra(pred_dir)
    if REF not in sp:
        raise SystemExit("reco spectra need the reference sample")
    edges = np.linspace(emin, emax, 26)
    ctr = 0.5 * (edges[:-1] + edges[1:]); wid = np.diff(edges)

    ref_counts = np.histogram(sp[REF]["sel"].e_pred.to_numpy(), bins=edges)[0]
    keep = ref_counts >= min_counts

    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL / 1.6, FULL / 1.5), sharex=True,
                               gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.07})
    dens, err = {}, {}
    for tag, st in DATASETS.items():
        if tag not in sp:
            continue
        d = sp[tag]["sel"]
        rows = d.Row.to_numpy(); e = d.e_pred.to_numpy()
        uniq = np.unique(rows)
        h, _ = np.histogram(e, bins=edges)
        dens[tag] = h / (len(uniq) * wid)
        # event-level bootstrap: the job label is unavailable for these caches
        idx = {r: i for i, r in enumerate(uniq)}
        ev = np.array([idx[r] for r in rows])
        bs = np.empty((n_boot, len(ctr)))
        for k in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            cnt = np.bincount(pick, minlength=len(uniq))
            w = cnt[ev].astype(float)
            bs[k] = np.histogram(e, bins=edges, weights=w)[0] / (len(uniq) * wid)
        err[tag] = bs.std(axis=0, ddof=1)
        # a log axis cannot show an empty bin; drop them rather than let the
        # mark fall to -inf and off the panel
        ok = keep & (dens[tag] > 0)
        a.errorbar(ctr[ok], dens[tag][ok], yerr=err[tag][ok], color=st["color"],
                   ls=st["ls"], marker=st["marker"], ms=2.8, mfc="none", mew=0.7,
                   elinewidth=0.6, capsize=1.2, label=st["label"])
    a.set_yscale("log")
    a.set_ylabel(r"clusters per event  [GeV$^{-1}$]")
    mark_scan_range(a)
    mark_scan_range(b, label=False)
    a.set_title("reconstructed neutron candidates, purity-locked selection", fontsize=7)
    # limits must contain the error-bar ends, not only the points
    lo = np.concatenate([(dens[t] - err[t])[keep & (dens[t] > 0)] for t in dens])
    hi = np.concatenate([(dens[t] + err[t])[keep & (dens[t] > 0)] for t in dens])
    lo = lo[lo > 0]
    a.set_ylim(lo.min() * 0.6, hi.max() * 1.8)
    a.legend(loc="lower left", fontsize=6.4)
    fin = []
    for tag, st in DATASETS.items():
        if tag not in dens or tag == REF:
            continue
        r = np.divide(dens[tag], dens[REF], out=np.full_like(dens[tag], np.nan),
                      where=dens[REF] > 0)
        re = r * np.sqrt((err[tag] / np.maximum(dens[tag], 1e-30)) ** 2 +
                         (err[REF] / np.maximum(dens[REF], 1e-30)) ** 2)
        b.errorbar(ctr[keep], r[keep], yerr=re[keep], color=st["color"], ls=st["ls"],
                   marker=st["marker"], ms=2.8, mfc="none", mew=0.7,
                   elinewidth=0.6, capsize=1.2)
        fin += [x for x in np.concatenate([(r + re)[keep], (r - re)[keep]])
                if np.isfinite(x)]
    b.axhline(1.0, color="#2b2b2b", lw=0.8)
    b.set_xscale("log"); b.set_xlabel(r"$E_{\mathrm{reco}}$  [GeV]")
    b.set_ylabel(r"ratio to $S_{\mathrm{pot}}=0$", fontsize=7)
    m = max(abs(np.array(fin) - 1).max(), 0.02) * 1.15 if fin else 0.2
    b.set_ylim(1 - m, 1 + m)
    lo_e = max(ctr[keep].min() - 0.15, 0.0); hi_e = ctr[keep].max() + 0.15
    a.set_xlim(lo_e, hi_e); b.set_xlim(lo_e, hi_e)
    for ax, nm in ((a, "reco_a"), (b, "reco_b")):
        assert_contained(ax, nm)
    save(fig, out_dir, "fig_reco_neutron_spectra")
    return {t: {"threshold": sp[t]["threshold"], "purity": sp[t]["purity"],
                "n_events": sp[t]["n_events"]} for t in sp}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth-hist-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    apply_style()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260929)
    meta = {"truth": fig_truth(a.truth_hist_dir, out, rng),
            "np_relation": fig_np_relation(a.truth_hist_dir, out, rng),
            "reco": fig_reco(a.pred_dir, out, rng)}
    json.dump(meta, open(out / "spectra_figure_data.json", "w"), indent=1)
    print("spectra figures complete")


if __name__ == "__main__":
    main()
