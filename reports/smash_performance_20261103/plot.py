"""Report figures from accepted fixed predictions and saved diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import sys
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_b_full import apply_style

FIG = HERE / "figures"
D = json.loads((HERE / "data.json").read_text())
Z = np.load(HERE / "counts.npz")
A = json.loads((ROOT / "results/smash_train_20261028/analysis/summary.json").read_text())
E = np.array(D["energy_edges_GeV"])
EDGES = np.r_[0., E]  # Visible spectrum range; separate negative/overflow bins retained.
SEEDS = ("42", "123", "456")
TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
POTS = (0, 18, 90)
COLORS = {"A0": "#1f77b4", "A1": "#1f77b4", "B0": "#b22222", "B1": "#b22222", "Bcomp": "#697442"}
STYLES = {"42": "-", "123": "--", "456": ":"}
NB = 400  # Inherited analysis precision, not a significance optimisation.
RNG = np.random.default_rng(20261028)
W = np.zeros((NB, len(D["content"])))
for fold in range(3):
    for tag in TAGS:
        ix = np.flatnonzero((np.array(D["fold"]) == fold) & (np.array(D["tag"]) == tag))
        W[:, ix] = RNG.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), NB)
MANIFEST = []
BOUNDS = []
STATS = {}


def vals(arm, seed, name):
    return Z[f"{arm}__{seed}__{name}"]


def div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.asarray(a, float) / np.asarray(b, float)
        return np.where(np.isfinite(q), q, np.nan)


def save(fig, name, title, scope, uncertainty):
    fig.canvas.draw()
    for i, ax in enumerate(fig.axes):
        # Check data in the declared horizontal view, including error-bar ends
        # and filled-band vertices. Spectrum tails outside that view are exported
        # separately; no vertical truncation is allowed to hide them in-view.
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        eps = max(y1-y0, 1.)*1e-7
        vertices = []
        for line in ax.lines:
            if line.get_transform() is ax.transData:
                vertices.append(np.column_stack((line.get_xdata(), line.get_ydata())))
        for artist in [*ax.patches, *ax.collections]:
            if artist.get_transform() is ax.transData:
                paths = artist.get_paths() if hasattr(artist, "get_paths") else [artist.get_path()]
                for path in paths:
                    v = path.vertices
                    if path.codes is not None: v = v[path.codes != 79]  # CLOSEPOLY has a dummy vertex.
                    vertices.append(v)
        checked = 0
        for v in vertices:
            v = np.asarray(v, float)
            valid = np.isfinite(v).all(axis=1) & (v[:, 0] >= x0) & (v[:, 0] <= x1)
            if ax.get_yscale() == "log": valid &= v[:, 1] > 0
            y = v[valid, 1]; checked += len(y)
            assert ((y >= y0-eps) & (y <= y1+eps)).all(), (name, i, y.min(), y.max(), (y0,y1))
        BOUNDS.append({"figure": name, "panel": i, "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()), "ratio_data_bounds": getattr(ax, "_report_data_bounds", None), "data_vertices_checked": checked, "vertical_truncation": False})
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=190, metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else {})
    plt.close(fig)
    MANIFEST.append({"name": name, "title": title, "scope": scope, "uncertainty": uncertainty, "source": "counts.npz / accepted analysis summary"})


def decorate(axes):
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(False)


def heading(fig, text, extra=""):
    fig.suptitle(textwrap.fill(text, 95) + "\nSMASH Xe+Cs, 2.87 GeV/nucleon · BM@N HGND simulation"
                 + ("\n" + textwrap.fill(extra, 95) if extra else ""), fontsize=8.4)


def band(ax, edges, central, sd, color, label, ls="-"):
    y = np.array(central, float)
    s = np.array(sd, float)
    ax.stairs(y, edges, baseline=None, color=color, label=label, ls=ls, lw=1.1)
    # Filled steps are bounded to positive values only for logarithmic density axes.
    lo = y - s
    if ax.get_yscale() == "log":
        lo = np.where(lo > 0, lo, np.nan)
    ax.fill_between(edges, np.r_[lo, lo[-1]], np.r_[y+s, (y+s)[-1]], step="post", color=color, alpha=.12)


def ratio_limits(ax, samples):
    lo, hi = 0., 1.
    for c, sd in samples:
        lower, upper = np.asarray(c)-np.asarray(sd), np.asarray(c)+np.asarray(sd)
        if np.isfinite(lower).any(): lo = min(lo, float(np.nanmin(lower)))
        if np.isfinite(upper).any(): hi = max(hi, float(np.nanmax(upper)))
    pad = .08 * (hi-lo)
    ax.set_ylim(lo-pad, hi+pad)
    ax._report_data_bounds = [lo, hi]


def primary():
    z = np.load(HERE / "primary_spectra.npz")
    edges = z["edges"]
    populated = np.any(np.stack([z[tag+"__central"] for tag in TAGS]) > 0, axis=(0, 1))
    last = np.flatnonzero(populated)[-1]
    ix = np.flatnonzero((edges[:-1] >= .3) & (np.arange(len(edges)-1) <= last))
    assert all(np.all(z[tag+"__central"][:, last+1:] == 0) for tag in TAGS)
    view = edges[np.r_[ix, ix[-1]+1]]
    widths = np.diff(view)
    fig, axes = plt.subplots(2, 2, figsize=(6.84, 5.2), sharex="col", gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for species, label in enumerate(("Primary neutrons", "Primary protons; straight-ray window")):
        ax, ratio = axes[:, species]
        ax.set_yscale("log")
        for tag, u, color, ls in zip(TAGS, POTS, ("#2b2b2b", "#6688aa", "#173e65"), ("-", "--", "-.")):
            c = z[tag+"__central"][species, ix] / widths
            r = z[tag+"__replicas"][:, species, ix] / widths
            band(ax, view, c, r.std(axis=0, ddof=1), color, f"U={u} MeV", ls)
        for hi, lo, color, ls in ((1, 0, "#6688aa", "--"), (2, 0, "#173e65", "-"), (2, 1, "#984a27", ":")):
            c = div(z[TAGS[hi]+"__central"][species, ix], z[TAGS[lo]+"__central"][species, ix])
            r = div(z[TAGS[hi]+"__replicas"][:, species, ix], z[TAGS[lo]+"__replicas"][:, species, ix])
            valid = np.isfinite(r).all(axis=0)
            c[~valid] = np.nan
            band(ratio, view, c, np.nanstd(r, axis=0, ddof=1), color, f"{POTS[hi]}/{POTS[lo]}", ls)
        for a in (ax, ratio):
            a.set_xlim(.3, view[-1])
            a.axvline(1., color=".65", ls=":", lw=.7)
            a.axvline(2., color=".65", ls=":", lw=.7)
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("Particles / event / GeV")
        ax.legend(fontsize=7, frameon=False)
        ratio.axhline(1, color=".3", ls="--", lw=.7)
        ratio.set_ylabel("Potential ratio")
        ratio.set_xlabel("Generated kinetic energy (GeV)")
        ratio.legend(fontsize=7, frameon=False, ncol=3)
    heading(fig, "MC truth: primary n/p spectra in the fixed geometric window", "Full primary inventory: 2,345,993 events · censored below 0.3 GeV · no B selection")
    decorate(axes)
    save(fig, "01_primary_spectra", "MC-truth primary spectra", "Full certified primary files, not processed spans or observed clusters", "SD of 400 whole-file resamples; densities divide by exact bin width")


def norm_hist(arm, field, tag, seed_mean=True):
    ix = np.array(D["tag"]) == tag
    n = vals(arm, "42", "n_pos")[ix]
    central, reps = [], []
    for seed in SEEDS:
        h = vals(arm, seed, field)[ix][:, 1:-1]
        central.append(h.sum(axis=0) / n.sum())
        reps.append((W[:, ix] @ h) / (W[:, ix] @ n)[:, None])
    return np.mean(central, axis=0), np.mean(reps, axis=0)


def common_spectra():
    fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.7), sharex="col", gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    widths = np.diff(EDGES)
    for i, tag in enumerate(TAGS):
        ax, ratio = axes[:, i]
        ax.set_yscale("log")
        ratio_samples = []
        t, tr = norm_hist("B1", "T", tag)
        band(ax, EDGES, t/widths, tr.std(axis=0, ddof=1)/widths, "#2b2b2b", "MC target")
        for arm, color, ls in (("A1", "#1f77b4", "-"), ("B0", "#b22222", "--"), ("B1", "#b22222", "-")):
            p, pr = norm_hist(arm, "P", tag)
            band(ax, EDGES, p/widths, pr.std(axis=0, ddof=1)/widths, color, arm, ls)
            c, rr = div(p, t), div(pr, tr)
            sd = np.nanstd(rr, axis=0, ddof=1)
            ratio_samples.append((c, sd))
            band(ratio, EDGES, c, sd, color, arm+"/target", ls)
        ax.set_title(f"U_sym={POTS[i]} MeV", fontsize=8)
        ax.set_ylabel("Density per MC-positive / GeV")
        ax.legend(fontsize=7, frameon=False)
        ratio.axhline(1, color=".3", ls="--", lw=.7)
        ratio_limits(ratio, ratio_samples)
        ratio.set_ylabel("Histogram ratio")
        ratio.set_xlabel("Cluster energy (GeV)")
        ax.set_xlim(0, 6.5)
    heading(fig, "Cluster MC-target and reconstructed-energy spectra: SAME positive candidates", "Mean of 3 fixed-model histograms, not triple statistics · tails tabulated · no B selection")
    decorate(axes)
    save(fig, "02_common_cluster_spectra", "Common-positive cluster energy spectra", "Same MC-positive candidates; common denominator; three models averaged descriptively", "File SD of the mean fixed-model histogram; histogram ratio is not detector efficiency")


def selected_spectra(arm, number):
    fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.7), sharex="col", gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    widths = np.diff(EDGES)
    for i, tag in enumerate(TAGS):
        ax, ratio = axes[:, i]
        ax.set_yscale("log")
        central, replica = {}, {}
        ratio_samples = []
        for field, color, ls in (("C2", "#2b2b2b", "-"), ("C3", "#1f77b4", "--"), ("C4", "#b22222", "-")):
            c, r = norm_hist(arm, field, tag)
            central[field], replica[field] = c, r
            band(ax, EDGES, c/widths, r.std(axis=0, ddof=1)/widths, color, field, ls)
        for h, l, color, ls in (("C3", "C2", "#1f77b4", "--"), ("C4", "C3", "#b22222", "-")):
            c = div(central[h], central[l])
            r = div(replica[h], replica[l])
            sd = np.nanstd(r, axis=0, ddof=1)
            ratio_samples.append((c, sd))
            band(ratio, EDGES, c, sd, color, h+"/"+l, ls)
        ax.set_title(f"{arm}; U_sym={POTS[i]} MeV", fontsize=8)
        ax.set_ylabel("Selected / all MC-positive / GeV")
        ax.legend(fontsize=7, frameon=False)
        ratio.axhline(1, color=".3", ls="--", lw=.7)
        ratio_limits(ratio, ratio_samples)
        ratio.set_xlabel("Cluster energy (GeV)")
        ratio.set_ylabel("Stage ratio")
        ratio.legend(fontsize=7, frameon=False)
        ax.set_xlim(0, 6.5)
    heading(fig, f"{arm}: selected cluster spectra, frozen per-fold score cuts", "C2: MC target of selected positives · C3: same objects, reco energy · C4: all selected")
    decorate(axes)
    save(fig, f"{number:02d}_selected_{arm}", f"Selected spectra {arm}", "C2/C3 identical positive IDs; C4 adds native-label negatives; no B selection", "400 shared file draws; common denominator; full-range C2/C3 integrals checked")


def interp(x, y, grid):
    ix = np.isfinite(x) & np.isfinite(y)
    x, y = x[ix], y[ix]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique, first = np.unique(x, return_index=True)
    maxima = np.maximum.reduceat(y, first)
    return np.interp(grid, unique, maxima)


def curves():
    fig, axes = plt.subplots(3, 2, figsize=(6.84, 6.6), layout="constrained")
    grid = np.linspace(0, 1, 201)  # Display coordinates only; AP/AUC use accepted exact values.
    for fold in range(3):
        ix = np.array(D["fold"]) == fold
        prevalence = vals("A1", "42", "n_pos")[ix].sum() / vals("A1", "42", "n")[ix].sum()
        axes[fold, 0].plot([0, 1], [0, 1], color=".5", ls=":", lw=.7)
        axes[fold, 1].axhline(prevalence, color=".5", ls=":", lw=.7)
        for arm in ("A1", "B1"):
            for seed in SEEDS:
                p, n = vals(arm, seed, "n_pos")[ix], vals(arm, seed, "n")[ix] - vals(arm, seed, "n_pos")[ix]
                tp, fp = vals(arm, seed, "TP")[ix], vals(arm, seed, "FP")[ix]
                r, f = tp.sum(axis=0)/p.sum(), fp.sum(axis=0)/n.sum()
                precision = div(tp.sum(axis=0), tp.sum(axis=0)+fp.sum(axis=0))
                br, bf = (W[:, ix] @ tp)/(W[:, ix] @ p)[:, None], (W[:, ix] @ fp)/(W[:, ix] @ n)[:, None]
                bp = div(W[:, ix] @ tp, W[:, ix] @ (tp+fp))
                roc = interp(np.r_[0, f, 1], np.r_[0, r, 1], grid)
                pr = interp(np.r_[0, r, 1], np.r_[1, precision, prevalence], grid)
                rb = np.array([interp(np.r_[0, x, 1], np.r_[0, y, 1], grid) for x, y in zip(bf, br)])
                replica_prevalence = (W[:, ix] @ p) / (W[:, ix] @ (p+n))
                pb = np.array([interp(np.r_[0, x, 1], np.r_[1, y, pi], grid) for x, y, pi in zip(br, bp, replica_prevalence)])
                for col, c, b in ((0, roc, rb), (1, pr, pb)):
                    sd = b.std(axis=0, ddof=1)
                    axes[fold, col].plot(grid, c, color=COLORS[arm], ls=STYLES[seed], lw=.9, label=f"{arm} s{seed}")
                    axes[fold, col].fill_between(grid, np.maximum(c-sd, 0), np.minimum(c+sd, 1), color=COLORS[arm], alpha=.07)
                old = A["arms"][arm][seed]["classification"]["pooled"][f"fold_{fold}"]["all"]
                fpr = (old["selected"] - old["selected_signal"]) / (old["n"] - old["signal"])
                axes[fold, 0].plot(fpr, old["efficiency"], marker="o", ms=3, color=COLORS[arm])
                axes[fold, 1].plot(old["efficiency"], old["purity"], marker="o", ms=3, color=COLORS[arm])
        for col, xlabel, ylabel in ((0, "False-positive rate", "True-positive rate"), (1, "Recall", "Count precision")):
            axes[fold, col].set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel, title=f"Fold {fold}")
        axes[fold, 1].legend(fontsize=7, ncol=2, frameon=False, loc="lower left")
    heading(fig, "Reconstructed score vs native MC label: test ROC / PR, folds kept separate", "Dots: frozen operating points · 401 accepted score-grid display · bands: file SD")
    decorate(axes)
    save(fig, "05_roc_pr", "Test ROC and precision-recall", "Each fold/seed separately; no pooled AUC or test retuning", "400 potential-by-fold file draws; displayed interpolation of fixed 401-grid; accepted AUC/AP remain exact")


def score_distributions():
    edges = np.array(D["score_edges"])
    fig, axes = plt.subplots(1, 3, figsize=(6.84, 3.4), layout="constrained")
    for fold, ax in enumerate(axes):
        ix = np.array(D["fold"]) == fold
        ax.set_yscale("log")
        for arm in ("A1", "B1"):
            for field, ls, label in (("score_pos", "-", "positive"), ("score_neg", "--", "negative")):
                h = np.mean([vals(arm, s, field)[ix] for s in SEEDS], axis=0)
                den = h.sum(axis=1)
                c = h.sum(axis=0) / den.sum() / np.diff(edges)
                r = (W[:, ix] @ h) / (W[:, ix] @ den)[:, None] / np.diff(edges)
                band(ax, edges, c, r.std(axis=0, ddof=1), COLORS[arm], arm+" "+label, ls)
        ax.set(xlim=(0, 1), xlabel="Reconstructed cluster score", ylabel="Density within MC class", title=f"Fold {fold}")
        ax.legend(fontsize=7, frameon=False)
    heading(fig, "Score distributions: reconstructed scores, native MC-positive / negative labels", "Mean of 3 fixed-model histograms; file SD; no B selection")
    decorate(axes)
    save(fig, "06_score_distributions", "Score distributions", "Fold-separated MC classes; background target zero is not a physical energy", "400 file draws; equal three-model average")


def profiles(population, number):
    fig, axes = plt.subplots(2, 2, figsize=(6.84, 5.2), layout="constrained")
    specs = (("median_response", "Median E_reco / E_target", 1., 1.),
             ("half_width_16_84", "Half 16-84% residual width", 1., 0.),
             ("bias_GeV", "Mean bias (MeV)", 1000., 0.),
             ("mae_GeV", "MAE (MeV)", 1000., 0.))
    for ax, (metric, ylabel, factor, reference) in zip(axes.flat, specs):
        for arm in ("A0", "A1", "B0", "B1"):
            rows = []
            for seed in SEEDS:
                lookup = {(r["e_lo"], r["e_hi"]): r for r in A["arms"][arm][seed]["energy"][population]["pooled"]}
                rows.append([lookup.get((lo, hi), {}).get(metric, np.nan) for lo, hi in zip(E[:-1], E[1:])])
            values = np.asarray(rows) * factor
            central, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)
            band(ax, E, central, sd, COLORS[arm], arm, "--" if arm.endswith("0") else "-")
        ax.axhline(reference, color=".4", ls=":", lw=.8)
        ax.set(xlim=(.3, 6.5), xlabel="True maximum qualifying MCTrack energy (GeV)", ylabel=ylabel)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    heading(fig, "Test cluster energy response: " + ("all MC-positive candidates" if population == "mc_positive" else "selected MC-positive candidates"), "Bands: SD over 3 trained seeds, NOT file error or 16-84% distribution width · bins with n<25 masked")
    decorate(axes)
    save(fig, f"{number:02d}_energy_{population}", "Test energy response " + population, "Target is maximum initial MCTrack energy; selected populations differ by arm", "Mean ± SD of 3 seed metrics; median/width sampling error not estimated; MIN_BIN=25")


def energy_selection():
    fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.9), layout="constrained")
    for i, tag in enumerate(TAGS):
        ix = np.array(D["tag"]) == tag
        for arm in ("A1", "B1"):
            for row, num, den in ((0, "C2", "T"), (1, "C3", "C4")):
                c, r = [], []
                for seed in SEEDS:
                    n, d = vals(arm, seed, num)[ix][:, 1:-1], vals(arm, seed, den)[ix][:, 1:-1]
                    c.append(div(n.sum(axis=0), d.sum(axis=0)))
                    r.append(div(W[:, ix] @ n, W[:, ix] @ d))
                central, reps = np.mean(c, axis=0), np.mean(r, axis=0)
                band(axes[row, i], EDGES, central, np.nanstd(reps, axis=0, ddof=1), COLORS[arm], arm)
            axes[0, i].set_title(f"U_sym={POTS[i]} MeV", fontsize=8)
        axes[0, i].set(xlim=(.3, 6.5), ylim=(0, 1.03), xlabel="True cluster target energy (GeV)", ylabel="C2 / C1: candidate efficiency")
        axes[1, i].set(xlim=(0, 6.5), ylim=(0, 1.03), xlabel="Reconstructed energy (GeV)", ylabel="C3 / C4: native-label purity")
        axes[1, i].axhline(.7, color=".5", ls=":", lw=.7)
        axes[0, i].legend(fontsize=7, frameon=False)
    heading(fig, "Energy-dependent selection quality at the frozen cuts", "MC-label assisted diagnostics; NOT efficiency vs all generated neutrons; bands: file SD")
    decorate(axes)
    save(fig, "09_energy_selection", "Energy-dependent efficiency and purity", "Mean of fixed-model ratios; actual MC-positive candidate denominator", "400 shared whole-file draws")


def size_quality():
    fig, axes = plt.subplots(2, 2, figsize=(6.84, 5.1), layout="constrained")
    metrics = (("auc", "ROC-AUC"), ("purity", "Count purity"), ("efficiency", "Candidate efficiency"), ("mae_GeV", "Energy MAE (MeV)"))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for arm in ("A1", "B1"):
            by_seed = []
            for seed in SEEDS:
                by_fold = [A["arms"][arm][seed]["classification"]["pooled"][f"fold_{f}"]["size"] for f in range(3)]
                by_seed.append([np.mean([rows[k][metric] for rows in by_fold]) for k in range(4)])
            v = np.array(by_seed) * (1000 if metric == "mae_GeV" else 1)
            ax.errorbar(np.arange(4) + (-.05 if arm == "A1" else .05), v.mean(axis=0), yerr=v.std(axis=0, ddof=1),
                        fmt="s" if arm == "A1" else "^", capsize=3, ms=5, color=COLORS[arm], label=arm)
        ax.set_xticks(range(4), ["1", "2-3", "4-7", "8+"])
        ax.set(xlabel="Observed hits per cluster", ylabel=ylabel)
        ax.legend(fontsize=8, frameon=False)
    heading(fig, "Reconstruction quality vs observed cluster size", "Equal mean of fold metrics per seed; bars: 3-seed SD; MC labels/targets; no B selection")
    decorate(axes)
    save(fig, "10_cluster_size", "Quality versus cluster size", "All accepted test folds; candidate-level metrics", "Mean ± SD over three seed-labelled fold averages")


def migration():
    fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.9), layout="constrained")
    for row, arm in enumerate(("A1", "B1")):
        for col, seed in enumerate(SEEDS):
            count = vals(arm, seed, "migration").sum(axis=0)
            den = vals(arm, seed, "true_energy_n").sum(axis=0)
            freq = div(count, den[:, None])
            # Only the displayed >=0.3 GeV cells are drawn. The denominator
            # still includes reconstructed tails, exported in counts.npz.
            mesh = axes[row, col].pcolormesh(E[1:], E[1:], freq[1:, 1:], cmap="viridis", norm=Normalize(0, 1), rasterized=False)
            axes[row, col].plot([.3, 6.5], [.3, 6.5], color="white", ls=":", lw=.6)
            axes[row, col].set(xlim=(.3, 6.5), ylim=(.3, 6.5), xlabel="Reco cluster energy (GeV)", ylabel="True MC target (GeV)", title=f"{arm}, seed {seed}")
    colorbar = fig.colorbar(mesh, ax=list(axes.flat), shrink=.85, label="Fraction of true-bin positives")
    colorbar.solids.set_rasterized(False)
    heading(fig, "MC-assisted energy migration matrices, fixed test candidates", "Rows include all positives in denominator; reco tails outside range retained separately; finite-sample frequencies")
    decorate(axes)
    save(fig, "11_migration", "Cluster energy migration", "All MC-positive candidates, every seed; row sum can be <1 because reco tails are tabulated", "Finite-sample conditional frequencies; no per-cell CI, not an unfolded response")


def paired_and_label_figures():
    pairs = json.loads((ROOT / "reports/status_20261103_support/all_pair_contrasts.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.9), sharex="col", layout="constrained")
    for col, (key, row) in enumerate(pairs["pairs"].items()):
        t = row["MC_truth_T1"]
        axes[0, col].axhspan(t["percent"]-t["whole_file_bootstrap_sd_percent"], t["percent"]+t["whole_file_bootstrap_sd_percent"], color=".2", alpha=.15)
        axes[0, col].axhline(t["percent"], color=".2", lw=.8, label="T1 MC truth")
        axes[1, col].axhline(0, color=".3", ls="--", lw=.8)
        for arm, dx, marker in (("A1", -.08, "s"), ("B1", .08, "^")):
            for i, seed in enumerate(SEEDS):
                rec = row["reconstructed"][arm][seed]
                for k, field in ((0, "C4"), (1, "C4_minus_T1_points")):
                    val = rec[field]
                    axes[k, col].errorbar(i+dx, val["percent"], yerr=val["whole_file_bootstrap_sd_percent"], fmt=marker, ms=4, capsize=2, color=COLORS[arm], label=arm if i == 0 else None)
        axes[0, col].set_title(f"U_sym: {key} MeV", fontsize=8)
        axes[0, col].legend(fontsize=7, frameon=False, loc="best")
        axes[1, col].set_xticks(range(3), SEEDS)
        axes[1, col].set_xlabel("Training seed")
    axes[0, 0].set_ylabel("Hardness contrast (%)")
    axes[1, 0].set_ylabel("C4 - T1 (points)")
    heading(fig, "All potential-pair hardness contrasts: MC truth T1 and reconstructed C4", "568 cross-fit files; no B selection; bars: shared 400-file SD")
    decorate(axes)
    save(fig, "17_all_potential_pairs", "Every potential pair", "Accepted processed-span truth and reconstructed C4; same saved paired replicas", "400 shared saved whole-file draws")

    raw = json.loads((ROOT / "reports/energy_label_results_20261103/audit_summary.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.9), sharex=True, layout="constrained")
    for col, tag in enumerate(TAGS):
        b = raw["bootstrap"][tag]
        for row, field, err in ((0, "positive_fraction_percent", "bootstrap_sd_percent"), (1, "paired_delta_vs_native_points", "paired_delta_sd_points")):
            axes[row, col].errorbar(range(6), b[field], yerr=b[err], fmt="o", capsize=2, ms=4)
            axes[row, col].set_xticks(range(6), ["off", ".1", ".2", ".3", ".4", ".5"])
        axes[0, col].set_title(f"U_sym={POTS[col]} MeV", fontsize=8)
        axes[1, col].axhline(0, color=".4", ls="--", lw=.7)
        axes[1, col].set_xlabel("MC-label floor (GeV)")
    axes[0, 0].set_ylabel("Positive clusters (%)")
    axes[1, 0].set_ylabel("Change from 0.3 (points)")
    heading(fig, "MC-truth label census: fixed observed graphs, S286 only", "No new trained model; no B selection; bars: 400 shared group-draw SD")
    decorate(axes)
    save(fig, "18_label_floor_scan", "S286 label-floor scan", "Accepted S286 observations, no score/model calls", "400 potential-by-part shared group draws")

    r = json.loads((ROOT / "reports/energy_label_results_20261103/results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.6), layout="constrained")
    for j, name, color, marker, dx in ((0, "All added", ".15", "o", -.10), (1, "Direct qualifying only", "#1f77b4", "^", 0), (2, "Nonprimary qualifying only", "#b22222", "s", .10)):
        rs = list(r["by_potential"].values())
        axes[0].errorbar(np.arange(3)+dx, [x["delta_positive_fraction_points"][j] for x in rs], yerr=[x["delta_positive_fraction_bootstrap_sd_points"][j] for x in rs], fmt=marker, ms=4, capsize=2, color=color, label=name)
    axes[0].set_xticks(range(3), ["0", "18", "90"])
    axes[0].set(xlabel="U_sym (MeV)", ylabel="Added / all clusters (points)", ylim=(0, .42))
    axes[0].legend(fontsize=7, frameon=False)
    v = np.array(r["total"]["new_target_fraction_points"][:3])
    sd = np.array(r["total"]["new_target_fraction_bootstrap_sd_points"][:3])
    axes[1].stairs(v, [0, .1, .2, .3], baseline=None, color=".15")
    axes[1].errorbar([.05, .15, .25], v, yerr=sd, fmt="none", capsize=2, color=".15")
    axes[1].axvline(.3, color="#b22222", ls="--")
    axes[1].set(xlabel="Added MC target energy (GeV)", ylabel="Added / all clusters (points)", ylim=(0, .21), xlim=(0, .32))
    heading(fig, "S286: added soft positives and qualifying-hit origins", "MC truth label diagnostics; same graphs; no B selection; bars: group SD")
    decorate(axes)
    save(fig, "19_soft_origins", "S286 soft targets and origins", "Qualifying-hit category, not full-cluster purity or primary efficiency", "400 shared group draws")


def validation_and_ladder():
    runs = {}
    for task in range(9):
        p = ROOT / f"results/smash_train_20261028/train_4466016/task_{task}/complete.json"
        j = json.loads(p.read_text())
        runs[(j["fold"], str(j["seed"]))] = (j, ROOT / f"results/smash_train_20261028/train_4466016/extracted/task_{task}/train")
    seedcolors = {"42": "#173e65", "123": "#a25422", "456": "#526d35"}
    fig, axes = plt.subplots(3, 2, figsize=(6.84, 5.8), layout="constrained")
    for fold in range(3):
        for col, metric, head in ((0, "validation_cluster_cls", "e_cls"), (1, "validation_cluster_energy", "e_en")):
            for seed in SEEDS:
                j, _ = runs[(fold, seed)]
                y = j["selection"][metric]
                axes[fold, col].plot(range(30), y, color=seedcolors[seed], ls=STYLES[seed], label="seed "+seed)
                axes[fold, col].axvline(j["selection"][head], color=seedcolors[seed], ls=":", lw=.8)
            axes[fold, col].set(xlabel="Epoch (zero based)", ylabel="Validation BCE" if col == 0 else "Validation positive SmoothL1", title=f"Fold {fold}; {head}")
        axes[fold, 0].legend(fontsize=7, frameon=False)
    heading(fig, "Training selection: population validation losses, all nine models", "Vertical lines: frozen selected epochs; curves are fixed seed trajectories, not uncertainty bands")
    decorate(axes)
    save(fig, "12_head_histories", "Validation head histories", "Native MC labels/targets; independent per-head first argmin", "Three fixed seed trajectories, no CI")

    fig, axes = plt.subplots(3, 3, figsize=(6.84, 5.8), layout="constrained")
    for fold in range(3):
        for col, key, arm in ((0, "A", "A1"), (1, "B", "B1"), (2, "Bcomp", "Bcomp")):
            for seed in SEEDS:
                j, p = runs[(fold, seed)]
                scan = pd.read_csv(p / f"validation/threshold_scan_{key}.csv")
                axes[fold, col].plot(scan.threshold, scan.purity_count, color=seedcolors[seed], lw=.9)
                axes[fold, col].plot(scan.threshold, scan.efficiency_count, color=seedcolors[seed], ls="--", lw=.9)
                axes[fold, col].axvline(j["cuts"][arm], color=seedcolors[seed], ls=":", lw=.7)
            axes[fold, col].axhline(.7, color=".3", ls=":", lw=.7)
            axes[fold, col].set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Score cut", ylabel="Purity / efficiency", title=f"Fold {fold}; {arm}")
    heading(fig, "VALIDATION threshold selection; no test tuning", "Solid: purity; dashed: efficiency; vertical: frozen cut; 42 blue, 123 brown, 456 green")
    decorate(axes)
    save(fig, "13_threshold_scans", "Validation threshold scans", "401 fixed score cuts; first count purity >=0.70; potentials pooled", "Three fixed seed curves, no CI")

    fig, axes = plt.subplots(3, 3, figsize=(6.84, 5.8), layout="constrained")
    for fold in range(3):
        for col, key, arm in ((0, "A", "A1"), (1, "B", "B1"), (2, "Bcomp", "Bcomp")):
            for seed in SEEDS:
                j, p = runs[(fold, seed)]
                z = np.load(p / f"validation/predictions_{key}.npz")
                m = (z["cl_label"] == 1) & (z["e_true"] > 0)
                et, ep = z["e_true"][m].astype(float), z["e_pred"][m].astype(float)
                cal = j["calibrations"][arm]
                x, y = np.array(cal["x_e_pred"]), np.array(cal["y_e_true"])
                calibrated = np.interp(ep, x, y)
                calibrated = np.where(ep < x[0], ep*y[0]/x[0], calibrated)
                calibrated = np.where(ep > x[-1], ep*y[-1]/x[-1], calibrated)
                for values, ls in ((ep, "--"), (calibrated, "-")):
                    med = [np.median(values[(et >= lo) & (et < hi)] / et[(et >= lo) & (et < hi)])
                           if ((et >= lo) & (et < hi)).sum() >= 25 else np.nan for lo, hi in zip(E[:-1], E[1:])]
                    axes[fold, col].stairs(med, E, baseline=None, color=seedcolors[seed], ls=ls, lw=.9)
            axes[fold, col].axhline(1, color=".3", lw=.7)
            axes[fold, col].set(xlim=(.3, 6.5), xlabel="True MC target (GeV)", ylabel="Median reco / target", title=f"Fold {fold}; {arm}")
    heading(fig, "VALIDATION calibration diagnostic: dashed raw, solid calibrated", "All positive validation candidates; inherited bins; three fixed seed curves, not test closure")
    decorate(axes)
    save(fig, "14_calibration_validation", "Validation calibration response", "Accepted maps reapplied, no fit; all positive validation candidates", "Three fixed-model curves; sampling error not estimated")

    fig, axes = plt.subplots(3, 2, figsize=(6.84, 5.7), layout="constrained")
    for row, seed in enumerate(SEEDS):
        for col, pair in enumerate(("18-0", "90-0")):
            for arm, dx, marker in (("A1", -.012, "s"), ("B1", .012, "^")):
                rows = [r for r in A["arms"][arm][seed]["delta_bias"]["mc_positive"][pair] if not r["window"] and r["delta_bias_MeV"] is not None]
                axes[row, col].errorbar([.5*(r["e_lo"]+r["e_hi"])+dx for r in rows], [r["delta_bias_MeV"] for r in rows], yerr=[r["delta_bias_sd_MeV"] for r in rows], fmt=marker, ms=3.5, capsize=2, color=COLORS[arm], label=arm)
            axes[row, col].axhline(0, color=".4", ls=":", lw=.7)
            axes[row, col].axvspan(1.75, 3, color=".9", zorder=0)
            axes[row, col].set(xlabel="True MC target bin (GeV)", ylabel="Delta bias (MeV)", title=f"{pair} MeV; seed {seed}")
            axes[row, col].legend(fontsize=7, frameon=False)
    heading(fig, "Potential-dependent TEST energy bias on MC-positive clusters", "Bars: 400-file SD; shaded [1.75,3) is the inherited critic window; tiny x offsets for visibility")
    decorate(axes)
    save(fig, "15_delta_bias", "Potential-dependent test bias", "All MC-positive test clusters, each seed and both potential differences", "400 whole-file bootstrap SD")

    ladder = json.loads((ROOT / "results/smash_train_20261028/heldout_rn/summary.json").read_text())["variants"]["primary"]["pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.7), layout="constrained")
    rungs = ("T0", "T1", "T2", "C1", "C2", "C3", "C4")
    for a, arm in enumerate(("A1", "B1")):
        for i, seed in enumerate(SEEDS):
            rec = ladder["arms"][arm]["endpoint"]
            axes[0].errorbar(np.arange(7)+(a-.5)*.22+(i-1)*.055, [rec[f"{r}/seed_{seed}"]["central_percent"] for r in rungs], yerr=[rec[f"{r}/seed_{seed}"]["bootstrap_sd_percent"] for r in rungs], fmt="s" if arm == "A1" else "^", ms=3, capsize=2, color=COLORS[arm], label=arm if i == 0 else None)
    for i, seed in enumerate(SEEDS):
        rec = ladder["pairs"]["B1-A1"]["endpoint"]
        axes[1].errorbar(np.arange(7)+(i-1)*.08, [rec[f"{r}/seed_{seed}"]["central_percent"] for r in rungs], yerr=[rec[f"{r}/seed_{seed}"]["bootstrap_sd_percent"] for r in rungs], fmt=("o", "s", "^")[i], ms=3, capsize=2, color=".2", label="seed "+seed)
    for ax in axes:
        ax.set_xticks(range(7), rungs)
        ax.axhline(0, color=".5", ls=":", lw=.7)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("90/0 hardness contrast (%)")
    axes[1].set_ylabel("B1 - A1, paired (points)")
    heading(fig, "Truth-to-reconstruction ladder: T0-T2 MC truth; C4 reconstructed", "568 reused cross-fit files; no B selection; bars: 400 shared whole-file SD")
    decorate(axes)
    save(fig, "16_heldout_ladder", "Truth-to-reconstruction ladder", "Retrospective cross-fit; processed-span primary truth and cluster stages", "400 shared whole-file draws; paired B1-A1")


def inventory_and_tables():
    tail = []
    for arm in ("A1", "B0", "B1"):
        for seed in SEEDS:
            t, p, c2, c3, c4 = [vals(arm, seed, k).sum(axis=0) for k in ("T", "P", "C2", "C3", "C4")]
            tail.append({"arm": arm, "seed": seed, "MC_positive": int(t.sum()), "pred_negative": int(p[0]),
                         "pred_below_0p1": int(p[:2].sum()), "pred_ge_6p5": int(p[-1]),
                         "selected_positive_C2_C3": int(c2.sum()), "selected_all_C4": int(c4.sum()),
                         "C3_below_0": int(c3[0]), "C3_ge_6p5": int(c3[-1]), "C4_below_0": int(c4[0]), "C4_ge_6p5": int(c4[-1])})
            assert c2.sum() == c3.sum()
    STATS["tails"] = tail
    STATS["draws"] = {"n": NB, "seed": 20261028, "strata": "potential x fold", "sha256": hashlib.sha256(W.astype('<i8').tobytes()).hexdigest()}
    accepted = {}
    for name, (source, title, scope, err) in accepted.items():
        for ext in ("pdf", "png"):
            shutil.copyfile(ROOT / "results/smash_train_20261028" / f"{source}.{ext}", FIG / f"{name}.{ext}")
        MANIFEST.append({"name": name, "title": title, "scope": scope, "uncertainty": err, "source": "results/smash_train_20261028/"+source,
                         "source_pdf_sha256": hashlib.sha256((FIG / f"{name}.pdf").read_bytes()).hexdigest()})
    others = (("20_crossfit", "reports/progress_20261021_20261029/figures/02_crossfit", "Cross-fit roles", "split scheme, no measurement", "exact role diagram"),
              ("21_bias_folds", "reports/progress_20261021_20261029/figures/08_bias_folds", "Energy bias by fold", "test MC-positive [1.75,3) window", "file SD and seed spread separated"),
              ("22_census", "results/smash_train_20261028/figures/fig_census_shared_records", "Input-group census", "Content/event grouping", "exact finite inventory counts"),
              ("23_split_precision", "results/smash_train_20261028/figures/fig_split_projected_sd", "Split design precision projection", "design diagnostic before training", "projected counting precision; not measured model uncertainty"))
    for name, source, title, scope, err in others:
        for ext in ("pdf", "png"):
            shutil.copyfile(ROOT / f"{source}.{ext}", FIG / f"{name}.{ext}")
        MANIFEST.append({"name": name, "title": title, "scope": scope, "uncertainty": err, "source": source,
                         "source_pdf_sha256": hashlib.sha256((FIG / f"{name}.pdf").read_bytes()).hexdigest()})
    (HERE / "figure_manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    (HERE / "figure_bounds.json").write_text(json.dumps(BOUNDS, indent=2) + "\n")
    (HERE / "plot_statistics.json").write_text(json.dumps(STATS, indent=2) + "\n")


if __name__ == "__main__":
    apply_style()
    # Exact report canvas: no automatic bbox resizing after print-size layout.
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.bbox": None})
    FIG.mkdir(exist_ok=True)
    primary()
    common_spectra()
    selected_spectra("A1", 3)
    selected_spectra("B1", 4)
    curves()
    score_distributions()
    profiles("mc_positive", 7)
    profiles("selected_positive", 8)
    energy_selection()
    size_quality()
    migration()
    paired_and_label_figures()
    validation_and_ladder()
    inventory_and_tables()
    print("Created", len(MANIFEST), "figure families")
