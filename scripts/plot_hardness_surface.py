#!/usr/bin/env python3
"""The hardness-window scan surface, so the chosen window is visibly a choice.

Quoting "a broad optimum near 2.2 GeV" asks the reader to take on trust both
that an optimum exists and that it is broad.  These maps show the whole scanned
plane instead: the response and its significance over the centre `Rthr` and the
half-gap `delta`, with the windows that fail to order the three samples masked
out and the selected point marked.

Colour follows docs/plotting_style_protocol.md section 2: a diverging map for
the signed response, centred on zero with symmetric limits, and a sequential map
for significance, which has no meaningful midpoint.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_b_full import apply_style, save, FULL


def grid(surface, field):
    """Scatter the scan records onto the (delta, Rthr) plane."""
    r = np.array(sorted({s["Rthr"] for s in surface}))
    d = np.array(sorted({s["delta"] for s in surface}))
    z = np.full((len(d), len(r)), np.nan)
    ordered = np.zeros((len(d), len(r)), bool)
    ri = {v: i for i, v in enumerate(r)}
    di = {v: i for i, v in enumerate(d)}
    for s in surface:
        z[di[s["delta"]], ri[s["Rthr"]]] = s[field]
        ordered[di[s["delta"]], ri[s["Rthr"]]] = s["ordered"]
    return r, d, z, ordered


def _edges(v):
    v = np.asarray(v, float)
    if len(v) == 1:
        return np.array([v[0] - 0.05, v[0] + 0.05])
    step = np.diff(v).mean()
    return np.concatenate([v - step / 2, [v[-1] + step / 2]])


def panel(ax, r, d, z, ordered, *, cmap, label, diverging, chosen=None,
          mask_unordered=True):
    zz = np.ma.masked_invalid(z)
    if mask_unordered:
        zz = np.ma.masked_where(~ordered, zz)
    if diverging:
        m = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
        kw = dict(cmap=cmap, vmin=-m, vmax=m)
    else:
        kw = dict(cmap=cmap)
    im = ax.pcolormesh(_edges(r), _edges(d), zz, shading="auto", **kw)
    # windows that do not order the samples are drawn as absence, not as a
    # colour a reader could mistake for a low value
    if mask_unordered and (~ordered).any():
        ax.pcolormesh(_edges(r), _edges(d),
                      np.ma.masked_where(ordered, np.ones_like(z)),
                      shading="auto", cmap=matplotlib.colors.ListedColormap(["#DADDE1"]),
                      vmin=0, vmax=1, zorder=0)
    if chosen is not None:
        ax.plot(chosen[0], chosen[1], marker="*", ms=13, mfc="none", mew=1.4,
                color="#111111", ls="none", zorder=5)
    ax.set_xlabel(r"$R_{\mathrm{thr}}$  [GeV]")
    ax.set_ylabel(r"$\delta$  [GeV]")
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True)
    p.add_argument("--reco", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    apply_style()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    T = json.load(open(a.truth))
    R = json.load(open(a.reco))
    fig, ax = plt.subplots(2, 2, figsize=(FULL, FULL / 1.15),
                           gridspec_kw={"hspace": 0.34, "wspace": 0.30})

    rt, dt, zt, ot = grid(T["surface"], "rel")
    ch = (T["chosen"]["Rthr"], T["chosen"]["delta"]) if T.get("chosen") else None
    panel(ax[0, 0], rt, dt, zt * 100, ot, cmap="RdBu_r",
          label=r"$S_{90}/S_{0}-1$  [%]", diverging=True, chosen=ch)
    ax[0, 0].set_title("truth, response", fontsize=7.5)

    _, _, st, _ = grid(T["surface"], "sigma")
    panel(ax[0, 1], rt, dt, st, ot, cmap="viridis", label=r"significance  [$\sigma$]",
          diverging=False, chosen=ch)
    ax[0, 1].set_title("truth, significance", fontsize=7.5)

    rr, dr, zr, orr = grid(R["surface"], "rel")
    panel(ax[1, 0], rr, dr, zr * 100, orr, cmap="RdBu_r",
          label=r"$S_{90}/S_{0}-1$  [%]", diverging=True, mask_unordered=False)
    ax[1, 0].set_title("reconstructed, response", fontsize=7.5)

    _, _, sr, _ = grid(R["surface"], "sigma")
    panel(ax[1, 1], rr, dr, sr, orr, cmap="viridis", label=r"significance  [$\sigma$]",
          diverging=False, mask_unordered=False)
    ax[1, 1].set_title("reconstructed, significance", fontsize=7.5)

    # Notes go inside their own panels: placed above the axes they drifted onto
    # the row above and sat on its x-axis label.
    n_adm = int(np.isfinite(zr).sum())
    for axx in (ax[1, 0], ax[1, 1]):
        axx.text(0.03, 0.04, f"none of {n_adm} admissible windows\norders the "
                             f"three samples",
                 transform=axx.transAxes, ha="left", va="bottom", fontsize=6.3,
                 color="#9E2222",
                 bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6))
    # top-left: the star sits low and right in this panel
    ax[0, 1].text(0.03, 0.96, "star: selected on\ndevelopment units",
                  transform=ax[0, 1].transAxes, ha="left", va="top",
                  fontsize=6.3, color="#222222",
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6))
    ax[0, 0].text(0.03, 0.04, "grey: does not order",
                  transform=ax[0, 0].transAxes, ha="left", va="bottom",
                  fontsize=6.3, color="#69727C",
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6))
    fig.suptitle("Hardness-window scan: response and significance over the "
                 "centre and half-gap", fontsize=8, y=0.965)
    save(fig, out, "fig_hardness_window_surface")

    summary = {
        "truth": {"n_scanned": len(T["surface"]),
                  "n_ordered": int(sum(s["ordered"] for s in T["surface"])),
                  "chosen": T.get("chosen"), "held_out": T.get("held_out")},
        "reco": {"n_scanned": len(R["surface"]),
                 "n_ordered": int(sum(s["ordered"] for s in R["surface"]))},
    }
    json.dump(summary, open(out / "hardness_surface_summary.json", "w"), indent=1)
    print("surface figure complete")


if __name__ == "__main__":
    main()
