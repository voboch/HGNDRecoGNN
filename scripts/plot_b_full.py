#!/usr/bin/env python3
"""Figures for the full-statistics impact-parameter / S_pot sensitivity report.

Follows docs/plotting_style_protocol.md: printed size with no rescale, 8 pt base
type, vector PDF plus a 200 dpi PNG for the weekly report, colour never the only
channel, a ratio sub-panel on every comparison against a reference, and the
Appendix A rule that axis limits are derived from the data and every mark is
asserted to lie inside its panel.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SINGLE, FULL = 3.30, 6.84            # inches (protocol sec. 1)
SCOPE = "full production"            # sample-scope label, set from --scope
DATASETS = {
    "zeroSpot":    {"label": r"$S_{\mathrm{pot}}=0$ MeV",  "color": "#2b2b2b", "marker": "o", "ls": "-"},
    "defaultSpot": {"label": r"$S_{\mathrm{pot}}=18$ MeV", "color": "#B22222", "marker": "s", "ls": "--"},
    "bigSpot":     {"label": r"$S_{\mathrm{pot}}=90$ MeV", "color": "#1f77b4", "marker": "^", "ls": "-"},
}
REF = "zeroSpot"


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "figure.dpi": 200,
        "axes.grid": False, "savefig.bbox": "tight",
    })


def snap_limits(values, pad=0.06, log=False):
    """Axis limits derived from the data, snapped to a 1-3-10 sequence if log.

    Appendix A: hardcoded limits once put 75 % of a spectrum above the panel.
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], float)
    if log:
        v = v[v > 0]
        lo, hi = v.min(), v.max()
        # snap to a 1-3-10 sequence: full decades leave the data filling only
        # ~65 % of the panel height, the 1-3 sequence ~86 % (Appendix A)
        ticks = np.concatenate([[1.0 * 10.0 ** k, 3.0 * 10.0 ** k] for k in range(-12, 13)])
        return float(ticks[ticks <= lo][-1]), float(ticks[ticks >= hi][0])
    lo, hi = v.min(), v.max()
    m = (hi - lo) * pad or abs(hi) * pad or 1.0
    return float(lo - m), float(hi + m)


def assert_contained(ax, name):
    """Every drawn mark and error-bar end must lie inside the panel."""
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    ex, ey = (x1 - x0) * 1e-6, (y1 - y0) * 1e-6
    bad = 0
    for ln in ax.get_lines():
        # axhline/axvline store x (or y) in axes coordinates, not data
        # coordinates; only marks drawn in data space can be checked here
        if ln.get_transform() is not ax.transData:
            continue
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        ok = np.isfinite(xd) & np.isfinite(yd)
        if not ok.any():
            continue
        bad += int(((xd[ok] < x0 - ex) | (xd[ok] > x1 + ex) |
                    (yd[ok] < y0 - ey) | (yd[ok] > y1 + ey)).sum())
    if bad:
        raise AssertionError(f"{name}: {bad} marks fall outside the panel")


def save(fig, out_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


# ---------------------------------------------------------------------------
def fig_b_distributions(data, out_dir, bw=0.5):
    """Fig 1: dN/db for each sample, with the ratio to the reference.

    Uncertainties are job-level: the density is estimated per job file and the
    error is the scatter of those per-file densities over sqrt(N_files).  Per-bin
    Poisson errors would be roughly 2.9x smaller here and would misrepresent the
    agreement, since b is not independent between events of the same job.
    """
    allb = np.concatenate([d.B.to_numpy() for d in data.values()])
    edges = np.arange(0.0, np.ceil(allb.max() / bw) * bw + bw, bw)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    dens, err = {}, {}
    for k, d in data.items():
        per = []
        for _, g in d.groupby("file_idx"):
            c, _ = np.histogram(g.B.to_numpy(), bins=edges)
            per.append(c / (max(c.sum(), 1) * bw))
        per = np.asarray(per)
        dens[k] = per.mean(axis=0)
        err[k] = per.std(axis=0, ddof=1) / np.sqrt(len(per))

    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL, FULL / 1.75), sharex=True,
                               gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.06})
    for k, s in DATASETS.items():           # reference first and darkest
        a.errorbar(ctr, dens[k], yerr=err[k], color=s["color"], ls=s["ls"],
                   marker=s["marker"], ms=3, mfc="none", mew=0.7, elinewidth=0.6,
                   capsize=1.2, label=s["label"])
    a.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}b$  [fm$^{-1}$]")
    # limits must contain the error-bar ends, not just the points (Appendix A)
    a.set_ylim(*snap_limits([v for k in dens
                             for v in np.concatenate([dens[k] + err[k], dens[k] - err[k]])]))
    a.legend(loc="upper left", ncols=1)
    a.set_title("Xe+CsI, $\\sqrt{s_{NN}}=2.87$ GeV, SMASH hard Skyrme, "
                f"minimum bias, {SCOPE}", fontsize=7)

    for k, s in DATASETS.items():
        if k == REF:
            b.axhline(1.0, color=s["color"], lw=0.8)
            continue
        r = np.divide(dens[k], dens[REF], out=np.full_like(dens[k], np.nan),
                      where=dens[REF] > 0)
        re = r * np.sqrt((err[k] / np.maximum(dens[k], 1e-30)) ** 2 +
                         (err[REF] / np.maximum(dens[REF], 1e-30)) ** 2)
        b.errorbar(ctr, r, yerr=re, color=s["color"], ls=s["ls"], marker=s["marker"],
                   ms=3, mfc="none", mew=0.7, elinewidth=0.6, capsize=1.2)
    b.set_xlabel(r"$b$  [fm]")
    b.set_ylabel(r"ratio to $S_{\mathrm{pot}}=0$")
    fin = []
    for k in DATASETS:
        if k == REF:
            continue
        rr = np.divide(dens[k], dens[REF], out=np.full(len(ctr), np.nan),
                       where=dens[REF] > 0)
        ee = rr * np.sqrt((err[k] / np.maximum(dens[k], 1e-30)) ** 2 +
                          (err[REF] / np.maximum(dens[REF], 1e-30)) ** 2)
        fin += [x for x in np.concatenate([rr + ee, rr - ee]) if np.isfinite(x)]
    m = max(abs(np.array(fin) - 1).max(), 0.02) * 1.15
    b.set_ylim(1 - m, 1 + m)
    b.set_xlim(0, edges[-1])
    for ax, nm in ((a, "fig1a"), (b, "fig1b")):
        assert_contained(ax, nm)
    save(fig, out_dir, "fig1_b_distributions")
    return {"bin_width_fm": bw, "edges": edges.tolist(),
            "density": {k: v.tolist() for k, v in dens.items()}}


def fig_job_structure(data, out_dir):
    """Fig 2: per-job-file <b>, the evidence that b carries job-level structure."""
    fig, ax = plt.subplots(figsize=(FULL, FULL / 2.4))
    stats, off, xt, xl = {}, 0, [], []
    for k, s in DATASETS.items():
        g = data[k].groupby("file_idx").B
        m, sd, n = g.mean().to_numpy(), g.std(ddof=1).to_numpy(), g.size().to_numpy()
        sem = sd / np.sqrt(n)
        gm = float(np.average(m, weights=1 / sem ** 2))
        chi2 = float((((m - gm) / sem) ** 2).sum()); ndf = len(m) - 1
        x = off + np.arange(len(m))
        ax.errorbar(x, m, yerr=sem, ls="none", color=s["color"], marker=s["marker"],
                    ms=2.5, mfc="none", mew=0.6, elinewidth=0.5, label=s["label"])
        ax.hlines(gm, x[0], x[-1], color=s["color"], lw=0.9, ls=":")
        xt.append(off + len(m) / 2); xl.append(s["label"])
        stats[k] = {"n_files": int(len(m)), "chi2": chi2, "ndf": ndf,
                    "chi2_per_ndf": chi2 / ndf, "weighted_mean_b": gm,
                    "rms_of_file_means": float(m.std(ddof=1)),
                    "mean_sem": float(sem.mean())}
        ax.text(off + len(m) / 2, ax.get_ylim()[1], "", ha="center")
        off += len(m) + 6
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel(r"$\langle b \rangle$ per job file  [fm]")
    ax.set_xlabel("job files, grouped by sample (dotted line: sample weighted mean)")
    txt = "\n".join(f"{DATASETS[k]['label']}:  $\\chi^2/\\mathrm{{ndf}}={v['chi2_per_ndf']:.1f}$"
                    f"  ({v['n_files']} files)" for k, v in stats.items())
    ax.text(0.015, 0.03, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    allm = np.concatenate([data[k].groupby("file_idx").B.mean().to_numpy() for k in DATASETS])
    # leave headroom under the annotation box so it cannot sit on a point
    lo, hi = snap_limits(allm, pad=0.16)
    ax.set_ylim(lo - (hi - lo) * 0.18, hi)
    assert_contained(ax, "fig2")
    save(fig, out_dir, "fig2_job_structure")
    return stats


def fig_closure(data, rep, out_dir, bw=0.25):
    """Fig 3: the b density before and after reweighting, with the ratio panel."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from b_reweight import ref_density, bin_weights
    allb = np.concatenate([d.B.to_numpy() for d in data.values()])
    edges = np.arange(0.0, np.ceil(allb.max() / bw) * bw + bw, bw)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    counts = {k: np.histogram(data[k].B.to_numpy(), bins=edges)[0].astype(float)
              for k in DATASETS}
    ref = ref_density([counts[k] for k in DATASETS])
    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL, FULL / 1.75), sharex=True,
                               gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.06})
    raw, rw = {}, {}
    for k in DATASETS:
        c = counts[k]
        w = bin_weights(c, ref)
        raw[k] = c / max(c.sum(), 1)
        cw = w * c
        rw[k] = cw / max(cw.sum(), 1)
    for k, s in DATASETS.items():
        a.step(ctr, raw[k] / bw, where="mid", color=s["color"], ls=s["ls"], lw=0.9,
               label=s["label"] + " (raw)")
    a.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}b$  [fm$^{-1}$]")
    a.set_ylim(*snap_limits([v for k in raw for v in raw[k] / bw]))
    a.legend(loc="upper left", fontsize=6.5)
    a.set_title("Impact-parameter reweighting onto the common reference density", fontsize=7)
    for k, s in DATASETS.items():
        if k == REF:
            b.axhline(1.0, color="#2b2b2b", lw=0.8)
        rr = np.divide(rw[k], ref, out=np.full_like(ref, np.nan), where=ref > 0)
        b.plot(ctr, rr, color=s["color"], ls=s["ls"], lw=0.9, marker=s["marker"],
               ms=2.5, mfc="none", mew=0.6, label=s["label"] + " (reweighted)")
    b.set_xlabel(r"$b$  [fm]"); b.set_ylabel("reweighted / reference")
    fin = np.array([x for k in DATASETS
                    for x in np.divide(rw[k], ref, out=np.full_like(ref, np.nan), where=ref > 0)
                    if np.isfinite(x)])
    m = max(abs(fin - 1).max(), 0.005) * 1.3
    b.set_ylim(1 - m, 1 + m); b.set_xlim(0, edges[-1])
    for ax, nm in ((a, "fig3a"), (b, "fig3b")):
        assert_contained(ax, nm)
    save(fig, out_dir, "fig3_reweight_closure")


def fig_observable(rep, out_dir, obs="R_n_band",
                   ylab=r"$R_n = N(E_{\mathrm{kin}}>2\,\mathrm{GeV})/N(E_{\mathrm{kin}}<1\,\mathrm{GeV})$"):
    """Fig 4: one observable vs centrality class, reweighted, with ratio panel."""
    rows = rep["classes"][obs]
    x = np.arange(len(rows)) + 0.5
    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL, FULL / 1.7), sharex=True,
                               gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.06})
    vals = {k: np.array([r["central"][k] for r in rows], float) for k in DATASETS}
    for k, s in DATASETS.items():
        a.plot(x, vals[k], color=s["color"], ls=s["ls"], marker=s["marker"], ms=3.5,
               mfc="none", mew=0.8, label=s["label"])
    a.set_ylabel(ylab, fontsize=7)
    a.set_ylim(*snap_limits(np.concatenate(list(vals.values()))))
    a.legend(loc="best", fontsize=6.5)
    a.set_title("Reweighted to a common $b$ density; uncertainties from job-file bootstrap",
                fontsize=7)
    rel = np.array([r["rel_90_over_0"] for r in rows], float)
    rer = np.array([r["rel_err"] for r in rows], float)
    b.axhline(0.0, color="#2b2b2b", lw=0.8)
    b.errorbar(x, rel, yerr=rer, color=DATASETS["bigSpot"]["color"], ls="none",
               marker="^", ms=3.5, mfc="none", mew=0.8, elinewidth=0.7, capsize=1.5)
    b.set_ylabel(r"$S_{90}/S_{0}-1$", fontsize=7)
    b.set_xlabel("centrality class  [% most central, from pooled $b$ quantiles]")
    b.set_xticks(x)
    b.set_xticklabels([f"{i*10}–{(i+1)*10}" for i in range(len(rows))], fontsize=6.5)
    ok = np.isfinite(rel) & np.isfinite(rer)
    m = max(np.abs(rel[ok] + rer[ok]).max(), np.abs(rel[ok] - rer[ok]).max()) * 1.2
    b.set_ylim(-m, m); b.set_xlim(0, len(rows))
    for ax, nm in ((a, f"fig4a_{obs}"), (b, f"fig4b_{obs}")):
        assert_contained(ax, nm)
    save(fig, out_dir, f"fig4_{obs}_centrality")


def fig_null(null_rep, rep, out_dir):
    """Fig 5: the null test -- |sigma| from same-sample splits vs the real result."""
    obs = list(null_rep.keys())
    y = np.arange(len(obs))
    fig, ax = plt.subplots(figsize=(FULL, FULL / 2.6))
    med = [null_rep[o]["median_sigma"] for o in obs]
    p90 = [null_rep[o]["p90_sigma"] for o in obs]
    mx = [null_rep[o]["max_sigma"] for o in obs]
    real = [abs(rep["integrated"][o]["sigma"]) for o in obs]
    for i, o in enumerate(obs):
        ax.plot([med[i], mx[i]], [i, i], color="#2b2b2b", lw=3, alpha=0.25,
                solid_capstyle="butt")
        ax.plot([med[i], p90[i]], [i, i], color="#2b2b2b", lw=3, alpha=0.55,
                solid_capstyle="butt")
        ax.plot(med[i], i, marker="o", ms=4, color="#2b2b2b", mfc="white", mew=0.9)
        ax.plot(real[i], i, marker="^", ms=5, color="#1f77b4", mfc="none", mew=1.1)
    ax.axvline(3.0, color="#B22222", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(obs, fontsize=7)
    ax.set_xlabel(r"$|\sigma|$ of the $S_{90}/S_{0}$ difference")
    ax.set_xlim(0, max(max(mx), max(real)) * 1.1)
    ax.set_ylim(-0.6, len(obs) - 0.4)
    ax.text(0.99, 0.02,
            "circles / bars: same-sample split null (median, p90, max)\n"
            "triangles: measured $S_{90}$ vs $S_{0}$;  dashed: 3$\\sigma$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    assert_contained(ax, "fig5")
    save(fig, out_dir, "fig5_null_calibration")


def fig_spectra(hist_dir, out_dir, emin=0.1, emax=8.0):
    """Fig 6: HGND-band kinetic-energy spectra for each species, with ratio panels.

    This is the shape behind the hardness ratio: it shows whether R moves because
    the spectrum genuinely tilts or because a threshold sits on a steep edge.

    The theta selection here is the histogram grid's nearest bins, 9.0-13.5 deg,
    slightly wider than the 8.9-13.1 deg band used for the quoted R values, which
    is applied per particle before binning.  The spectrum is shown from 0.1 GeV:
    below that the histogram's own lower edge dominates the ratio and compresses
    the 1-8 GeV region where the hardness thresholds sit.
    """
    Z = {k: np.load(Path(hist_dir) / f"{k}_filehist.npz") for k in DATASETS
         if (Path(hist_dir) / f"{k}_filehist.npz").exists()}
    if REF not in Z:
        raise SystemExit("spectra figure needs the reference sample")
    ee = Z[REF]["_e_edges"]; th = Z[REF]["_th_edges"]
    ec = 0.5 * (ee[:-1] + ee[1:]); wid = np.diff(ee)
    band = (th[:-1] >= 8.9) & (th[:-1] < 13.1)
    keep = (ec > emin) & (ec < emax)

    fig, ax = plt.subplots(2, 2, figsize=(FULL, FULL / 1.45), sharex="col",
                           gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.07,
                                        "wspace": 0.28})
    out = {}
    for col, (sp, nm) in enumerate(((0, "neutrons"), (1, "protons"))):
        dens = {}
        for k, z in Z.items():
            h = z["_global"][sp][:, :, band].sum(axis=(0, 2)).astype(float)
            dens[k] = (h / h.sum()) / wid
        for k, st in DATASETS.items():
            if k not in dens:
                continue
            ax[0, col].plot(ec[keep], dens[k][keep], color=st["color"], ls=st["ls"],
                            marker=st["marker"], ms=2.5, mfc="none", mew=0.7,
                            label=st["label"])
        ax[0, col].set_yscale("log"); ax[0, col].set_xscale("log")
        ax[0, col].set_ylabel(
            r"$(1/N)\,\mathrm{d}N/\mathrm{d}E_{\mathrm{kin}}$  [GeV$^{-1}$]")
        ax[0, col].set_title(rf"{nm}, HGND band ($9.0^\circ<\theta<13.5^\circ$)",
                             fontsize=7)
        ax[0, col].set_ylim(*snap_limits([dens[k][keep] for k in dens], log=True))
        if col == 0:
            ax[0, col].legend(loc="lower left", fontsize=6.5)
        for k, st in DATASETS.items():
            if k == REF or k not in dens:
                continue
            r = np.divide(dens[k], dens[REF], out=np.full_like(dens[k], np.nan),
                          where=dens[REF] > 0)
            ax[1, col].plot(ec[keep], r[keep], color=st["color"], ls=st["ls"],
                            marker=st["marker"], ms=2.5, mfc="none", mew=0.7)
            out[f"{nm}_{k}"] = r[keep].tolist()
        ax[1, col].axhline(1.0, color="#2b2b2b", lw=0.8)
        ax[1, col].set_xscale("log")
        ax[1, col].set_xlabel(r"$E_{\mathrm{kin}}$  [GeV]")
        ax[1, col].set_ylabel(r"ratio to $S_{\mathrm{pot}}=0$", fontsize=7)
        ax[1, col].set_xlim(emin, emax); ax[0, col].set_xlim(emin, emax)

    # shared ratio limits across the row, so the two species compare by eye
    fin = np.array([v for vv in out.values() for v in vv if np.isfinite(v)])
    m = max(abs(fin - 1).max(), 0.02) * 1.15 if len(fin) else 0.2
    for col in (0, 1):
        ax[1, col].set_ylim(1 - m, 1 + m)
    for i in range(2):
        for j in range(2):
            assert_contained(ax[i, j], f"fig6_{i}{j}")
    save(fig, out_dir, "fig6_hgnd_band_spectra")
    return {"e_centers": ec[keep].tolist(), "ratios": out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--null", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples", default=None,
                   help="comma-separated subset, when one production is still reducing")
    p.add_argument("--scope", default="full production",
                   help="sample-scope label shown in figure titles")
    a = p.parse_args()
    global SCOPE, DATASETS
    SCOPE = a.scope
    if a.samples:
        keep = a.samples.split(",")
        DATASETS = {k: v for k, v in DATASETS.items() if k in keep}
    apply_style()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    data = {k: pd.read_pickle(a.events_dir / f"{k}_events.pkl") for k in DATASETS}
    rep = json.load(open(a.report))
    binned = fig_b_distributions(data, a.output_dir)
    js = fig_job_structure(data, a.output_dir)
    fig_closure(data, rep, a.output_dir)
    labels = {
        "Rn_over_Rp_band": r"$R_n/R_p$  (HGND band)",
        "R_n_band": r"$R_n=N(E_{\mathrm{kin}}>2)/N(E_{\mathrm{kin}}<1)$  (HGND band)",
        "R_p_band": r"$R_p=N(E_{\mathrm{kin}}>2)/N(E_{\mathrm{kin}}<1)$  (HGND band)",
        "R_n_mid": r"$R_n$  ($|y_{\mathrm{cm}}|<0.5$)",
        "np_band": r"$n/p$  (HGND band)",
    }
    for o, lab in labels.items():
        if o in rep.get("classes", {}):
            fig_observable(rep, a.output_dir, o, lab)
    try:
        fig_spectra(a.events_dir, a.output_dir)
    except SystemExit as e:
        print(f"  spectra figure skipped: {e}")
    if a.null and a.null.exists():
        fig_null(json.load(open(a.null)), rep, a.output_dir)
    json.dump({"job_structure": js, "b_binned": binned},
              open(a.output_dir / "figure_data.json", "w"), indent=1)
    print("figures complete")


if __name__ == "__main__":
    main()
