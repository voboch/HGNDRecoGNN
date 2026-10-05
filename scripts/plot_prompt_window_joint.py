#!/usr/bin/env python3
"""The coincidence window and the prompt-hit definition, calibrated together.

Two constants control how much of a neutron's deposit becomes signal, and
neither was examined before this scan:

  t_win    the coincidence window that links two hits into one cluster;
  n_sigma  the width of the time-of-flight match that makes a hit *prompt*,
           fixed at 2 in the production.

They are not independent. Widening either admits more of the event into the
signal definition, so choosing one while holding the other at an unexamined
value is not a calibration. This plots the reference report's own clustering
criteria, plus the two quantities this analysis actually needs, over both.

The background-per-cluster rate is deliberately *not* used as a criterion: it
falls as `n_sigma` widens purely because hits move from the background column
to the signal column, so it is definitional across this axis rather than a
measure of quality.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True, help="joint time_window_scan.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--iso-target", type=float, default=97.0,
                   help="isolation the reference report claims, in percent")
    p.add_argument("--band", default="rthr2.2_d0.4",
                   help="which hardness band to report the induced bias for; "
                        "the default is the one the window optimiser selected")
    p.add_argument("--split-target", type=float, default=98.0,
                   help="unsplit fraction implied by the report's '<2%% split'")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    meta = json.load(open(a.scan))
    recs = meta["scan"]
    sig = sorted({r["nsigma"] for r in recs if r.get("nsigma") is not None})
    if not sig:
        raise SystemExit("this scan has no --prompt-sigmas axis")
    tw = sorted({r["twindow"] for r in recs})
    print(f"{len(sig)} prompt widths x {len(tw)} windows")

    def grid(field, scale=1.0):
        g = np.full((len(sig), len(tw)), np.nan)
        for r in recs:
            if r.get("nsigma") is None:
                continue
            g[sig.index(r["nsigma"]), tw.index(r["twindow"])] = r[field] * scale
        return g

    iso = grid("isolation_dombay", 100)
    uns = grid("unsplit_dombay", 100)
    cnt = grid("count_bias", 100)
    # the reference band the optimiser selected, not whichever key sorts first
    rb = np.full_like(iso, np.nan)
    band_key = a.band if a.band in recs[0]["bands"] else None
    if band_key is None:
        raise SystemExit(f"--band {a.band} not in scan; have "
                         f"{sorted(recs[0]['bands'])}")
    for r in recs:
        if r.get("nsigma") is None:
            continue
        b = r["bands"][band_key]
        rb[sig.index(r["nsigma"]), tw.index(r["twindow"])] = 100 * b["R_bias"]
    print(f"R bias reported for band {band_key} "
          f"(E < {recs[0]['bands'][band_key]['rthr'] - recs[0]['bands'][band_key]['delta']:.2f}"
          f" | E >= {recs[0]['bands'][band_key]['rthr'] + recs[0]['bands'][band_key]['delta']:.2f} GeV)")

    _report(sig, tw, iso, uns, cnt, rb, a)
    _plot(sig, tw, iso, uns, cnt, rb, meta, a)
    return 0


def _report(sig, tw, iso, uns, cnt, rb, a):
    print(f"\n{'n_sig':>6} {'t_win':>6} {'isolation':>10} {'unsplit':>9} "
          f"{'count bias':>11} {'R bias':>8}  admissible")
    best = None
    for i, ns in enumerate(sig):
        for j, t in enumerate(tw):
            ok = iso[i, j] >= a.iso_target and uns[i, j] >= a.split_target
            mark = "  yes" if ok else ""
            print(f"{ns:>6.1f} {t:>6.2f} {iso[i,j]:>9.2f}% {uns[i,j]:>8.2f}% "
                  f"{cnt[i,j]:>+10.2f}% {rb[i,j]:>+7.2f}%{mark}")
            if ok and (best is None or abs(cnt[i, j]) < best[0]):
                best = (abs(cnt[i, j]), ns, t, iso[i, j], uns[i, j], rb[i, j])
    print()
    if best is None:
        print(f"  NO (n_sigma, t_win) meets both reference criteria "
              f"(isolation >= {a.iso_target}%, unsplit >= {a.split_target}%).")
    else:
        _, ns, t, i_, u_, r_ = best
        print(f"  both criteria met and the count closest to closing: "
              f"n_sigma = {ns:g}, t_win = {t:g} ns "
              f"(isolation {i_:.2f}%, unsplit {u_:.2f}%, R bias {r_:+.2f}%)")


def _plot(sig, tw, iso, uns, cnt, rb, meta, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_b_full import apply_style, hep_label, no_grid, FULL
    apply_style()

    fig, ax = plt.subplots(1, 4, figsize=(FULL * 1.25, FULL / 3.1))
    cmap = plt.get_cmap("viridis")
    cols = [cmap(x) for x in np.linspace(0.08, 0.86, len(sig))]
    mks = ["o", "s", "^", "D", "v", "P"]

    panels = [
        (ax[0], iso, "isolation  [%]", a.iso_target),
        (ax[1], uns, "unsplit  [%]", a.split_target),
        (ax[2], cnt, "count bias  [%]", 0.0),
        (ax[3], rb, r"bias on $R$  [%]", 0.0),
    ]
    for k, (axx, g, ylab, ref) in enumerate(panels):
        for i, ns in enumerate(sig):
            axx.plot(tw, g[i], marker=mks[i % len(mks)], ms=3.2, mfc="none",
                     mew=0.9, lw=1.0, color=cols[i],
                     label=rf"{ns:g}$\sigma$")
        if ref is not None:
            axx.axhline(ref, color="#B22222", ls=":", lw=1.0)
        axx.set_xlabel(r"$t_{\rm win}$  [ns]")
        axx.set_ylabel(ylab)
    ax[0].legend(title="prompt match", loc="lower left", fontsize=5.8,
                 title_fontsize=5.8)
    no_grid(*ax)
    hep_label(ax[3],
              system=meta.get("system", ""),
              extra=meta.get("note", "MC truth"), loc="upper left",
              fontsize=6.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, f"fig_prompt_window_joint.{ext}"),
                    dpi=200)
    print(f"\nwrote {a.out_dir}/fig_prompt_window_joint.png")


if __name__ == "__main__":
    sys.exit(main())
