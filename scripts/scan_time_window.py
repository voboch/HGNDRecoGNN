#!/usr/bin/env python3
"""Justify the coincidence time window `twindow` used in graph construction.

`rlocal = 3.6` (cell units) and `twindow` set the two radius graphs whose
*intersection* defines the hit graph, and whose connected components define
clusters.  `rlocal` follows the reference notebook; `twindow` does not — the
reference uses 1.0 ns and this pipeline has used 1.5 ns since the first run,
with no recorded reason.  This script supplies the missing measurement.

The window trades two failure modes against each other:

  * too narrow  — a single neutron's hits fall outside the window and its
    cluster *splits*, so the cluster energy target (max of Ekin over the
    cluster) is carried by a fragment and the rest becomes fake;
  * too wide    — hits from different particles fall inside the window and
    clusters *merge*, so one cluster carries two neutrons and the energy
    target keeps only the harder one.

Both are measured directly against MC truth (`Id`), on the same
connected-component clustering the dataset builder uses, so the choice can be
read off a curve instead of asserted.

Choosing the objective
----------------------
The obvious composite, `(1 - split)(1 - merge)`, does not work: splitting falls
from 43 % to 18 % across 0.5-3.0 ns while merging only rises from 3 % to 7 %,
so the product increases monotonically and every window is beaten by a wider
one.  Weighting the two failures equally is what breaks it, and there is no
window-independent reason to pick any other pair of weights.

What the analysis actually needs is an unbiased *count* in each energy band,
because the spectral hardness is a ratio of two such counts.  The two failures
enter that count with opposite sign and so define an optimum:

  * a split neutron whose hits land in two clusters, both holding a labelled
    hit, is reconstructed *twice* — both clusters carry `cllabel = 1` and the
    same `clenergy`, inflating the count;
  * a merged pair shares one cluster whose `clenergy` is the max over its
    labelled hits, so the softer neutron is *lost* — deflating the count.

So the objective is the closure of the signal-cluster count against the true
neutron count, and, since hardness is a ratio, the induced bias on R itself,
where the two band errors partly cancel.  Both are reported per energy band,
because a window that is unbiased overall can still tilt the spectrum.

Output: a JSON table and a figure (failure rates, count closure, R bias).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

RLOCAL_DEFAULT = 3.6

# Hardness windows to report the induced bias for. The first is the point the
# threshold optimiser selected; the others bracket it so a window that only
# looks unbiased at one split is not mistaken for an unbiased one.
BANDS: dict[str, tuple[float, float]] = {
    "rthr2.2_d0.4": (2.2, 0.4),
    "rthr1.6_d0.3": (1.6, 0.3),
    "rthr2.6_d0.4": (2.6, 0.4),
}
REF_BAND = "rthr2.2_d0.4"
COLS = ["Row", "fX", "fY", "fZ", "LayerId", "RowId", "ColumnId", "fTime",
        "Id", "PDG", "n0_label", "Ekin", "Side"]

# Time resolution assumed in the production, and the floor this pipeline adds
# to the reference's neutron label.
SIGMA_T_NS = 0.15
EKIN_FLOOR = 0.3


def _etof(dist_m: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Kinetic energy a neutron would need to arrive at `t_ns` over `dist_m`."""
    E0, c = 939.55, 299792458.0
    beta = np.clip(dist_m / np.maximum(t_ns * 1e-9, 1e-30) / c, 1e-14, 1 - 1e-14)
    return np.clip(E0 * (1.0 / np.sqrt(1.0 - beta ** 2) - 1.0), 0.0, 1e4) * 1e-3


def prompt_label(df: pd.DataFrame, nsigma: float) -> np.ndarray:
    """The prompt-neutron label at an arbitrary time-matching width.

    A hit is prompt when its MC kinetic energy is consistent with the energy
    implied by its arrival time and position, within `nsigma` of the assumed
    time resolution. The production fixes this at 2 sigma and nothing records
    why, so it is scanned here alongside the coincidence window: the two are
    not independent, since widening either admits more hits into the signal
    definition.
    """
    dist = np.linalg.norm(
        np.asarray([df.fX.to_numpy(), df.fY.to_numpy(), df.fZ.to_numpy()]) / 100.0,
        axis=0)
    t = df.fTime.to_numpy(float)
    dt = nsigma * SIGMA_T_NS
    hi = _etof(dist, t - dt)      # earlier arrival -> higher energy
    lo = _etof(dist, t + dt)
    ek = df.Ekin.to_numpy(float)
    return ((df.PDG.to_numpy() == 2112)
            & (ek > EKIN_FLOOR)
            & (ek <= hi) & (ek >= lo)
            & df.Side.isin([5, 6]).to_numpy()).astype(bool)


def _cluster_one(pos_local: np.ndarray, t: np.ndarray, rlocal: float,
                 twindow: float) -> np.ndarray:
    """Connected components of (spatial ball ∩ time ball), as the builder does.

    Brute force over the pair matrix: half-detector events hold ~10 hits, so
    the O(n^2) form is both faster and easier to trust than a tree here.
    """
    n = pos_local.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    d2 = ((pos_local[:, None, :] - pos_local[None, :, :]) ** 2).sum(-1)
    adj = (d2 <= rlocal * rlocal) & (np.abs(t[:, None] - t[None, :]) <= twindow)

    # Connected components by repeated label propagation (min over neighbours),
    # the same fixed point the reference reaches with min-aggregation passing.
    lab = np.arange(n)
    while True:
        new = np.where(adj, lab[None, :], n).min(axis=1)
        new = np.minimum(new, lab)
        if np.array_equal(new, lab):
            break
        lab = new
    return np.unique(lab, return_inverse=True)[1]


def scan(df: pd.DataFrame, twindows: list[float], rlocal: float,
         nsigma: float | None = None) -> list[dict]:
    """Per-window cluster-quality rates over all (event, half) graphs."""
    df = df.copy()
    df["_sig"] = (prompt_label(df, nsigma) if nsigma is not None
                  else df["n0_label"].to_numpy(np.int64).astype(bool))
    groups = []
    for istop, half in ((True, df[df.fY > 0]), (False, df[df.fY < 0])):
        bad = half[half.PDG == 0].Row.unique()
        half = half[~half.Row.isin(bad)]
        for row, g in half.groupby("Row", sort=False):
            groups.append(g)
    print(f"  {len(groups)} half-event graphs, "
          f"{sum(len(g) for g in groups)} hits", flush=True)

    prepared = [(
        g[["LayerId", "RowId", "ColumnId"]].to_numpy(np.float64),
        g["fTime"].to_numpy(np.float64),
        g["Id"].to_numpy(np.int64),
        g["_sig"].to_numpy(bool),
        g["Ekin"].to_numpy(np.float64),
    ) for g in groups]

    out = []
    for tw in twindows:
        n_cl = n_cl_iso = 0
        n_sig = n_sig_iso = n_sig_merge = 0
        purity_num = purity_den = 0.0
        n_neutron = n_neutron_split = 0
        e_cluster: list[float] = []
        e_truth: list[float] = []
        # Dombay slide 9 counts three multiplicities, all on PROMPT hits:
        #   clusters per prompt n0        -> splitting
        #   prompt n0 per signal cluster  -> isolation
        #   background hits per signal cluster
        # A neutron deposits many hits but usually only one *prompt* hit, so
        # counting all its hits (as an earlier version here did) reports a
        # split rate an order of magnitude too high.
        mult_cl_per_n: dict[int, int] = {}
        mult_n_per_cl: dict[int, int] = {}
        mult_bkg_per_cl: dict[int, int] = {}

        for pos, t, ids, sig, ekin in prepared:
            lab = _cluster_one(pos, t, rlocal, tw)
            k = lab.max() + 1
            n_cl += k

            for c in range(k):
                m = lab == c
                cids = ids[m]
                uniq, cnt = np.unique(cids, return_counts=True)
                isolated = uniq.size == 1
                n_cl_iso += isolated
                if sig[m].any():
                    n_sig += 1
                    n_sig_iso += isolated
                    # prompt neutrons sharing this cluster
                    n_prompt = int(np.unique(ids[m & sig]).size)
                    mult_n_per_cl[n_prompt] = mult_n_per_cl.get(n_prompt, 0) + 1
                    if n_prompt > 1:
                        n_sig_merge += 1
                    # hits in this cluster that are not prompt-neutron hits
                    n_bkg = int((~sig[m]).sum())
                    mult_bkg_per_cl[n_bkg] = mult_bkg_per_cl.get(n_bkg, 0) + 1
                    purity_num += cnt.max()
                    purity_den += cids.size
                    e_cluster.append(float(ekin[m & sig].max()))

            # clusters holding the PROMPT hits of one neutron
            for nid in np.unique(ids[sig]):
                n_neutron += 1
                own_prompt = (ids == nid) & sig
                k_cl = int(np.unique(lab[own_prompt]).size)
                mult_cl_per_n[k_cl] = mult_cl_per_n.get(k_cl, 0) + 1
                if k_cl > 1:
                    n_neutron_split += 1
                e_truth.append(float(ekin[ids == nid][0]))

        ec = np.asarray(e_cluster)
        et = np.asarray(e_truth)

        rec = {
            "twindow": tw,
            "nsigma": float(nsigma) if nsigma is not None else None,
            "n_clusters": int(n_cl),
            "clusters_per_graph": float(n_cl / len(prepared)),
            "isolation_all": float(n_cl_iso / max(n_cl, 1)),
            "n_signal_clusters": int(n_sig),
            "isolation_signal": float(n_sig_iso / max(n_sig, 1)),
            "merge_rate_signal": float(n_sig_merge / max(n_sig, 1)),
            "purity_signal": float(purity_num / max(purity_den, 1.0)),
            "n_signal_neutrons": int(n_neutron),
            "split_rate_signal": float(n_neutron_split / max(n_neutron, 1)),
        }
        # Kept for the record: this rises monotonically with the window and so
        # selects nothing. See the module docstring.
        rec["usable_signal"] = float((1.0 - rec["split_rate_signal"])
                                     * (1.0 - rec["merge_rate_signal"]))

        # Count closure: signal clusters against true neutrons. Splits push it
        # up, merges push it down, so it crosses zero.
        rec["count_bias"] = float(ec.size / max(et.size, 1) - 1.0)

        def _norm(d):
            tot = sum(d.values()) or 1
            return {str(k): v / tot for k, v in sorted(d.items())}

        rec["mult_clusters_per_prompt_n"] = _norm(mult_cl_per_n)
        rec["mult_prompt_n_per_signal_cluster"] = _norm(mult_n_per_cl)
        rec["mult_background_per_signal_cluster"] = _norm(mult_bkg_per_cl)
        rec["isolation_dombay"] = float(mult_n_per_cl.get(1, 0)
                                        / max(sum(mult_n_per_cl.values()), 1))
        rec["unsplit_dombay"] = float(mult_cl_per_n.get(1, 0)
                                      / max(sum(mult_cl_per_n.values()), 1))
        rec["mean_background_per_signal_cluster"] = float(
            sum(k * v for k, v in mult_bkg_per_cl.items())
            / max(sum(mult_bkg_per_cl.values()), 1))

        # Same closure inside the two hardness bands, and on the ratio itself.
        rec["bands"] = {}
        for tag, (rthr, delta) in BANDS.items():
            lo_c = int((ec < rthr - delta).sum())
            hi_c = int((ec >= rthr + delta).sum())
            lo_t = int((et < rthr - delta).sum())
            hi_t = int((et >= rthr + delta).sum())
            r_c = hi_c / lo_c if lo_c else float("nan")
            r_t = hi_t / lo_t if lo_t else float("nan")
            rec["bands"][tag] = {
                "rthr": rthr, "delta": delta,
                "n_low_cluster": lo_c, "n_high_cluster": hi_c,
                "n_low_truth": lo_t, "n_high_truth": hi_t,
                "bias_low": float(lo_c / lo_t - 1.0) if lo_t else float("nan"),
                "bias_high": float(hi_c / hi_t - 1.0) if hi_t else float("nan"),
                "R_cluster": float(r_c), "R_truth": float(r_t),
                "R_bias": float(r_c / r_t - 1.0) if r_t else float("nan"),
            }
        out.append(rec)
        ref = rec["bands"][REF_BAND]
        print(f"  tw={tw:4.2f}  isolation={100*rec['isolation_dombay']:6.2f}%  "
              f"unsplit={100*rec['unsplit_dombay']:6.2f}%  "
              f"<bkg/cl>={rec['mean_background_per_signal_cluster']:5.2f}  "
              f"count_bias={100 * rec['count_bias']:+7.2f}%  "
              f"R_bias={100 * ref['R_bias']:+6.2f}%", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hits", required=True, help="hits parquet cache")
    p.add_argument("--every", type=int, default=20,
                   help="keep every Nth event (spreads over generation jobs)")
    p.add_argument("--max-events", type=int, default=60000)
    p.add_argument("--rlocal", type=float, default=RLOCAL_DEFAULT)
    p.add_argument("--twindows", nargs="*", type=float,
                   default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--system", default=r"Xe+CsI, $\sqrt{s_{NN}}$ = 2.87 GeV",
                   help="system and energy for the in-frame label; the two "
                        "samples differ and a figure that names the wrong one "
                        "is not interpretable")
    p.add_argument("--sample-note", default=r"MC truth, $U_{\rm sym}$ = 18 MeV",
                   help="second label line, e.g. which sample this is")
    p.add_argument("--prompt-sigmas", nargs="*", type=float, default=None,
                   help="Widths of the prompt-hit time match to scan, in units "
                        "of the 0.15 ns time resolution. The production fixes "
                        "this at 2 with no recorded reason, and it is not "
                        "independent of the coincidence window: both control "
                        "how much of a neutron's deposit counts as signal. "
                        "Omit to use the label already stored in the cache.")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)

    print(f"reading {a.hits}", flush=True)
    df = pd.read_parquet(a.hits, columns=COLS)
    print(f"  {len(df)} hits, {df.Row.nunique()} events", flush=True)

    keep = df.Row % a.every == 0
    df = df[keep]
    rows = np.sort(df.Row.unique())
    if rows.size > a.max_events:
        rows = rows[: a.max_events]
        df = df[df.Row.isin(rows)]
    print(f"  sampled {df.Row.nunique()} events, {len(df)} hits", flush=True)

    if a.prompt_sigmas:
        recs = []
        for ns in a.prompt_sigmas:
            print(f"  prompt match = {ns:g} sigma", flush=True)
            recs += scan(df, a.twindows, a.rlocal, nsigma=ns)
    else:
        recs = scan(df, a.twindows, a.rlocal)

    meta = {
        "hits": a.hits,
        "rlocal": a.rlocal,
        "events": int(df.Row.nunique()),
        "hits_used": int(len(df)),
        "every": a.every,
        "scan": recs,
        "system": a.system,
        "note": a.sample_note,
    }
    jpath = os.path.join(a.out, "time_window_scan.json")
    with open(jpath, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {jpath}")

    _plot(recs, a.rlocal, meta, os.path.join(a.out, "time_window_scan.png"))
    return 0


def _plot(recs: list[dict], rlocal: float, meta: dict, path: str) -> None:
    """Two figures: the slide-9 multiplicities, and what the window costs R.

    The first reproduces the three distributions the reference report uses to
    justify the clustering (its slide 9) so the two can be compared directly.
    The second is the part that matters for this analysis and that the
    reference does not quote: the bias the clustering puts on the hardness
    ratio.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_b_full import apply_style, hep_label, no_grid, FULL
    apply_style()

    tw = np.array([r["twindow"] for r in recs])
    _plot_multiplicities(recs, tw, rlocal, meta, path, plt,
                         hep_label, no_grid, FULL)
    _plot_bias(recs, tw, rlocal, meta, path, plt, hep_label, no_grid, FULL)


def _bar_dist(ax, rec, key, colour, label):
    """One multiplicity distribution as a step histogram over integer k."""
    d = rec.get(key, {})
    if not d:
        return
    ks = np.array(sorted(int(k) for k in d))
    vs = np.array([d[str(k)] for k in ks], dtype=float)
    ax.step(np.append(ks, ks[-1] + 1) - 0.5, np.append(vs, vs[-1]),
            where="post", color=colour, lw=1.2, label=label)


def _plot_multiplicities(recs, tw, rlocal, meta, path, plt, hep_label,
                         no_grid, FULL):
    """Slide-9 distributions at the reference window and at the alternatives."""
    shown = [r for r in recs if r["twindow"] in (0.5, 1.0, 1.5)]
    if not shown:
        shown = recs[:3]
    cols = {0.5: "#1f77b4", 1.0: "#2b2b2b", 1.5: "#B22222"}

    fig, ax = plt.subplots(1, 3, figsize=(FULL, FULL / 2.9))
    specs = [
        ("mult_clusters_per_prompt_n", "clusters per prompt $n^0$",
         "splitting"),
        ("mult_prompt_n_per_signal_cluster", "prompt $n^0$ per signal cluster",
         "isolation"),
        ("mult_background_per_signal_cluster",
         "background hits per signal cluster", "contamination"),
    ]
    for a, (key, xlab, tag) in zip(ax, specs):
        for r in shown:
            c = cols.get(r["twindow"], "#777777")
            _bar_dist(a, r, key, c,
                      rf"$t_{{\rm win}}$ = {r['twindow']:g} ns")
        a.set_xlabel(xlab)
        a.set_yscale("log")
        a.text(0.97, 0.03, tag, transform=a.transAxes, ha="right",
               va="bottom", fontsize=6.5, style="italic")
    ax[0].set_ylabel("fraction")
    ax[0].set_xlim(0.5, 4.5)
    ax[1].set_xlim(0.5, 4.5)
    ax[2].set_xlim(-0.5, 15.5)
    ax[2].legend(loc="upper right", fontsize=6.2)
    no_grid(*ax)
    hep_label(ax[0],
              system=meta.get("system", r"Xe+CsI, $\sqrt{s_{NN}}$ = 2.87 GeV"),
              extra=(f"{meta.get('note', 'MC truth')}\n"
                     rf"$r_{{\rm local}}$ = {rlocal} cells"),
              loc="lower left")
    fig.tight_layout()
    p2 = path.replace(".png", "_multiplicities.png")
    fig.savefig(p2, dpi=200)
    fig.savefig(p2.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {p2}")


def _plot_bias(recs, tw, rlocal, meta, path, plt, hep_label, no_grid, FULL):
    iso = np.array([100 * r["isolation_dombay"] for r in recs])
    uns = np.array([100 * r["unsplit_dombay"] for r in recs])
    bkg = np.array([r["mean_background_per_signal_cluster"] for r in recs])
    cbias = np.array([100 * r["count_bias"] for r in recs])

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FULL, FULL / 2.9))

    a1.plot(tw, iso, "o-", color="#1f77b4", ms=3.5, mfc="none",
            label="isolation: one prompt $n^0$ per cluster")
    a1.plot(tw, uns, "s--", color="#2b2b2b", ms=3.5, mfc="none",
            label="unsplit: one cluster per prompt $n^0$")
    a1.axhline(97, color="#B22222", ls=":", lw=1.0)
    a1.annotate("reference claim 97 %", xy=(tw.max(), 97), xytext=(-3, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6.0, color="#B22222")
    a1.set_ylabel("fraction  [%]")
    a1.legend(loc="lower left", fontsize=6.0)

    a2.plot(tw, bkg, "o-", color="#B22222", ms=3.5, mfc="none")
    a2.set_ylabel("background hits per signal cluster")

    a3.axhline(0, color="#888888", lw=0.8, ls="--")
    styles = {REF_BAND: ("o-", "#2b2b2b", 1.4)}
    for tag in recs[0]["bands"]:
        st, col, lw = styles.get(tag, ("^--", None, 1.0))
        lab = (rf"$E_{{\rm thr}}$={recs[0]['bands'][tag]['rthr']:.1f}, "
               rf"$\delta$={recs[0]['bands'][tag]['delta']:.1f}")
        a3.plot(tw, [100 * r["bands"][tag]["R_bias"] for r in recs],
                st, color=col, lw=lw, ms=3.5, mfc="none", label=lab)
    a3.set_ylabel(r"bias on $R$  [%]")
    a3.legend(loc="upper left", fontsize=6.0)

    for ax in (a1, a2, a3):
        ax.set_xlabel(r"$t_{\rm win}$  [ns]")
        for mark, col in ((1.0, "tab:orange"), (1.5, "tab:brown")):
            if mark in set(tw.tolist()):
                ax.axvline(mark, color=col, ls=":", lw=1.0, alpha=0.8)
        _pad_ylim(ax)
    # the two settings go in the legend rather than as rotated text crossing
    # every panel
    from matplotlib.lines import Line2D
    h, l = a1.get_legend_handles_labels()
    h += [Line2D([], [], color="tab:orange", ls=":", lw=1.0),
          Line2D([], [], color="tab:brown", ls=":", lw=1.0)]
    l += ["reference, 1.0 ns", "this pipeline, 1.5 ns"]
    a1.legend(h, l, loc="lower left", fontsize=6.0)
    no_grid(a1, a2, a3)
    # panel 1 is full: the two curves span the top and the legend the bottom
    hep_label(a2,
              system=meta.get("system", r"Xe+CsI, $\sqrt{s_{NN}}$ = 2.87 GeV"),
              extra=meta.get("note", "MC truth"), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {path}")


def _zero_crossing(x: np.ndarray, y: np.ndarray) -> float | None:
    """Linear interpolation of the first sign change in y(x), if any."""
    sgn = np.sign(y)
    for i in range(len(y) - 1):
        if sgn[i] != 0 and sgn[i + 1] != 0 and sgn[i] != sgn[i + 1]:
            return float(x[i] + (x[i + 1] - x[i]) * (-y[i]) / (y[i + 1] - y[i]))
    return None


def _pad_ylim(ax) -> None:
    """Appendix A: limits must contain every drawn point, with margin."""
    vals = []
    for ln in ax.get_lines():
        if ln.get_transform() is not ax.transData:
            continue
        yd = np.asarray(ln.get_ydata(), dtype=float)
        vals.extend(yd[np.isfinite(yd)].tolist())
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)


if __name__ == "__main__":
    sys.exit(main())
