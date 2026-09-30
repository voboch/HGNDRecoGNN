"""Event-level multi-neutron reconstruction efficiency.

The usual inclusive efficiency hides the question that matters for events with
more than one neutron: after reconstructing the leading neutron, how often are
the second, third, ... true neutrons retained?  This script evaluates one common
classifier threshold on every sample and reports

  * P(the k-th hardest true neutron is selected | N_true >= k),
  * P(at least k true-neutron clusters are selected | N_true >= k), and
  * the full selected-multiplicity response conditional on N_true.

``cl_label == 1`` defines a reconstructable truth-matched neutron cluster.  The
rank is by descending ``e_true`` and is therefore deterministic.  Statistical
intervals are Wilson binomial intervals; they describe the evaluated sample and
do not include detector or model systematics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


SAMPLES = ("zeroSpot", "defaultSpot", "bigSpot")
U_MEV = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
COLORS = {"zeroSpot": "#2166AC", "defaultSpot": "#222222", "bigSpot": "#B2182B"}
MARKERS = {"zeroSpot": "o", "defaultSpot": "s", "bigSpot": "^"}
MIN_PLOT_EVENTS = 20


def hep_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11.5,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def wilson(k: int, n: int, z: float = 1.0) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, ctr - half), min(1.0, ctr + half)


def load_threshold(pred_dir: Path, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    p = pred_dir / "performance.json"
    if not p.exists():
        raise FileNotFoundError("--threshold not supplied and performance.json is absent")
    d = json.loads(p.read_text())
    for key in ("working_threshold", "threshold"):
        if key in d:
            return float(d[key])
    raise KeyError(f"no working threshold recorded in {p}")


def analyse_sample(path: Path, threshold: float, max_rank: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    d = pd.read_pickle(path)
    need = {"Row", "cl_label", "cl_score", "e_true"}
    missing = need - set(d.columns)
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")

    d = d.loc[:, list(need)].copy()
    d["is_true"] = d.cl_label.to_numpy() == 1
    d["selected"] = d.cl_score.to_numpy() > threshold
    d["selected_true"] = d.is_true & d.selected

    per_event = d.groupby("Row", sort=False).agg(
        n_true=("is_true", "sum"),
        n_selected_true=("selected_true", "sum"),
        n_selected=("selected", "sum"),
    ).astype(int).reset_index()

    sig = d[d.is_true].sort_values(["Row", "e_true"], ascending=[True, False]).copy()
    sig["rank"] = sig.groupby("Row", sort=False).cumcount() + 1

    ranks = []
    for rank in range(1, max_rank + 1):
        eligible = per_event[per_event.n_true >= rank]
        ranked = sig[sig["rank"] == rank]
        n = int(len(eligible))
        n_rank = int(ranked.selected.sum())
        n_cum = int((eligible.n_selected_true >= rank).sum())
        n_raw = int((eligible.n_selected >= rank).sum())
        pr, lr, hr = wilson(n_rank, n)
        pc, lc, hc = wilson(n_cum, n)
        pp, lp, hp = wilson(n_raw, n)
        ranks.append({
            "rank": rank, "eligible_events": n,
            "rank_selected": n_rank, "rank_efficiency": pr,
            "rank_eff_lo": lr, "rank_eff_hi": hr,
            "at_least_k_true_selected": n_cum, "cumulative_efficiency": pc,
            "cumulative_eff_lo": lc, "cumulative_eff_hi": hc,
            "at_least_k_selected_raw": n_raw, "raw_cumulative_probability": pp,
            "raw_cumulative_lo": lp, "raw_cumulative_hi": hp,
        })
    rank_df = pd.DataFrame(ranks)

    rows = []
    max_true = min(max_rank, int(per_event.n_true.max()))
    for nt in range(1, max_true + 1):
        q = per_event[per_event.n_true == nt]
        n = int(len(q))
        all_rec = int((q.n_selected_true >= nt).sum())
        any_rec = int((q.n_selected_true >= 1).sum())
        p_all, lo_all, hi_all = wilson(all_rec, n)
        p_any, lo_any, hi_any = wilson(any_rec, n)
        rows.append({
            "n_true": nt, "events": n,
            "mean_selected_true": float(q.n_selected_true.mean()) if n else np.nan,
            "mean_selected_raw": float(q.n_selected.mean()) if n else np.nan,
            "mean_fraction_selected": float((q.n_selected_true / nt).mean()) if n else np.nan,
            "p_any": p_any, "p_any_lo": lo_any, "p_any_hi": hi_any,
            "p_all": p_all, "p_all_lo": lo_all, "p_all_hi": hi_all,
        })
    by_mult = pd.DataFrame(rows)

    cap = max_rank
    truth = np.minimum(per_event.n_true.to_numpy(), cap)
    reco = np.minimum(per_event.n_selected_true.to_numpy(), cap)
    matrix = np.zeros((cap + 1, cap + 1), dtype=int)
    np.add.at(matrix, (truth, reco), 1)
    row_sum = matrix.sum(axis=1, keepdims=True)
    response = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum > 0)

    summary = {
        "prediction_file": str(path),
        "threshold": threshold,
        "n_cluster_rows": int(len(d)),
        "n_events_with_clusters": int(len(per_event)),
        "n_events_with_true_neutron_cluster": int((per_event.n_true > 0).sum()),
        "mean_n_true_given_positive": float(per_event.loc[per_event.n_true > 0, "n_true"].mean()),
        "mean_n_selected_true_given_positive": float(per_event.loc[per_event.n_true > 0, "n_selected_true"].mean()),
        "response_counts": matrix.tolist(),
        "response_probability": response.tolist(),
        "overflow_bin": cap,
    }
    return summary, rank_df, by_mult


def plot_rank(all_rank: pd.DataFrame, all_mult: pd.DataFrame, outdir: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for tag in SAMPLES:
        # Keep all ranks in the CSV/JSON, but do not turn one-event tails into
        # apparent physics outliers.  The fifth-neutron point exists only for
        # one 90 MeV event and has no comparative meaning.
        q = all_rank[(all_rank.dataset == tag)
                     & (all_rank.eligible_events >= MIN_PLOT_EVENTS)]
        c, mk = COLORS[tag], MARKERS[tag]
        y = q.rank_efficiency.to_numpy()
        ye = np.clip(np.vstack([y - q.rank_eff_lo, q.rank_eff_hi - y]), 0, None)
        ax0.errorbar(q["rank"], y, yerr=ye, fmt=mk + "-", color=c,
                     capsize=2.5, label=rf"$U_{{\rm sym}}={U_MEV[tag]}$ MeV")
        yc = q.cumulative_efficiency.to_numpy()
        yce = np.clip(
            np.vstack([yc - q.cumulative_eff_lo, q.cumulative_eff_hi - yc]),
            0, None,
        )
        ax0.errorbar(q["rank"] + 0.05, yc, yerr=yce, fmt=mk + "--", color=c,
                     mfc="white", capsize=2.5, alpha=0.8)

        m = all_mult[(all_mult.dataset == tag)
                     & (all_mult.events >= MIN_PLOT_EVENTS)]
        ax1.plot(m.n_true, m.mean_fraction_selected, mk + "-", color=c,
                 label=rf"$U_{{\rm sym}}={U_MEV[tag]}$ MeV")
        ax1.plot(m.n_true, m.p_all, mk + "--", color=c, mfc="white", alpha=0.8)

    ax0.set(xlabel=r"true-neutron rank $k$ (ordered in $E_{\rm kin}$)",
            ylabel="conditional probability", ylim=(0, 1.03))
    ax0.text(0.03, 0.05, "solid: kth neutron selected\ndashed: at least k true clusters selected",
             transform=ax0.transAxes, fontsize=9, va="bottom")
    ax1.set(xlabel=r"true reconstructable multiplicity $N_n^{\rm true}$",
            ylabel="event-level efficiency", ylim=(0, 1.03))
    ax1.text(0.03, 0.05, "solid: mean selected fraction\ndashed: all true clusters selected",
             transform=ax1.transAxes, fontsize=9, va="bottom")
    ax0.text(0.98, 0.05, rf"points require $N_{{\rm event}}\geq{MIN_PLOT_EVENTS}$",
             transform=ax0.transAxes, fontsize=8, va="bottom", ha="right", color="0.35")
    for ax in (ax0, ax1):
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8.5, loc="best")
    fig.suptitle(r"HGND multi-neutron reconstruction, common classifier working point", y=1.01)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"conditional_neutron_efficiency_v2.{ext}")
    plt.close(fig)


def plot_response(summaries: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    im = None
    for ax, tag in zip(axes, SAMPLES):
        p = np.asarray(summaries[tag]["response_probability"], dtype=float)
        show = np.ma.masked_where(p <= 0, p)
        im = ax.imshow(show, origin="lower", cmap="viridis", norm=LogNorm(vmin=1e-3, vmax=1),
                       aspect="equal")
        ax.plot([-0.5, p.shape[0] - 0.5], [-0.5, p.shape[0] - 0.5], "w--", lw=1)
        ax.set_title(rf"$U_{{\rm sym}}={U_MEV[tag]}$ MeV")
        ax.set_xlabel(r"selected truth-matched multiplicity $N_n^{\rm sel}$")
    axes[0].set_ylabel(r"true reconstructable multiplicity $N_n^{\rm true}$")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                     label=r"$P(N_n^{\rm sel}\mid N_n^{\rm true})$")
    fig.suptitle("HGND multiplicity response (overflow included in final bin)")
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.16, top=0.82, wspace=0.10)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"multiplicity_response.{ext}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--max-rank", type=int, default=6)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    threshold = load_threshold(args.pred_dir, args.threshold)
    hep_style()

    summaries: dict[str, dict] = {}
    ranks, mults = [], []
    for tag in SAMPLES:
        path = args.pred_dir / tag / "pred_clusters_smash.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        summary, rank_df, mult_df = analyse_sample(path, threshold, args.max_rank)
        summary["U_MeV"] = U_MEV[tag]
        summaries[tag] = summary
        ranks.append(rank_df.assign(dataset=tag, U_MeV=U_MEV[tag]))
        mults.append(mult_df.assign(dataset=tag, U_MeV=U_MEV[tag]))

    all_rank = pd.concat(ranks, ignore_index=True)
    all_mult = pd.concat(mults, ignore_index=True)
    all_rank.to_csv(args.out_dir / "conditional_rank_efficiency.csv", index=False)
    all_mult.to_csv(args.out_dir / "efficiency_by_true_multiplicity.csv", index=False)
    payload = {
        "definition": {
            "truth": "cl_label == 1 (truth-matched reconstructable neutron cluster)",
            "selected": f"cl_score > {threshold:.6g}",
            "rank": "descending e_true within event",
            "interval": "68% Wilson binomial; statistical only",
            "event_scope": "events represented by at least one cluster row",
        },
        "samples": summaries,
    }
    (args.out_dir / "multiplicity_efficiency.json").write_text(json.dumps(payload, indent=1))
    plot_rank(all_rank, all_mult, args.out_dir)
    plot_response(summaries, args.out_dir)
    print(f"threshold={threshold:.6f}")
    for tag in SAMPLES:
        q = all_rank[all_rank.dataset == tag]
        vals = " ".join(f"k={int(r['rank'])}:{r.rank_efficiency:.3f}" for _, r in q.iterrows())
        print(f"{tag:12s} {vals}")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
