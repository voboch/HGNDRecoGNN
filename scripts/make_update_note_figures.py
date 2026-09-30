"""HEP-style figures for the HGND n/p and spectral-hardness update note."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAMPLES = ("zeroSpot", "defaultSpot", "bigSpot")
U_MEV = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
COLORS = {"zeroSpot": "#2166AC", "defaultSpot": "#222222", "bigSpot": "#B2182B"}
MARKERS = {"zeroSpot": "o", "defaultSpot": "s", "bigSpot": "^"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 11.5,
        "axes.linewidth": 1.15,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })


def stamp(ax, subtitle: str) -> None:
    ax.text(0.03, 0.97, "BM@N HGND  Simulation", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=10)
    ax.text(0.03, 0.89, subtitle, transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5)


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png")
    plt.close(fig)


def plot_global_np(csv_path: Path, out: Path) -> None:
    d = pd.read_csv(csv_path)
    panels = (("abs_y_lt_0p5", r"$|y-y_{\rm cm}|<0.5$"),
              ("all_y", r"full phase space (all rapidities)"))
    fig = plt.figure(figsize=(10.8, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.05], hspace=0.04, wspace=0.20)
    for col, (panel, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ar = fig.add_subplot(gs[1, col], sharex=ax)
        if panel == "all_y":
            q = d[(d.figure == "ekin") & d.valid].copy()
            q = q.groupby(
                ["dataset", "spot_mev", "bin_lo", "bin_hi", "bin_center"],
                as_index=False,
            )[["n", "p"]].sum()
            q["ratio"] = q.n / q.p
            q["stat_error"] = q.ratio * np.sqrt(1.0 / q.n + 1.0 / q.p)
        else:
            q = d[(d.figure == "ekin") & (d.panel == panel) & d.valid].copy()
        q = q[(q.bin_center >= 0.3) & (q.bin_center <= 3.0)]
        base = q[q.dataset == "zeroSpot"].set_index("bin_center")
        for tag in SAMPLES:
            z = q[q.dataset == tag]
            ax.errorbar(z.bin_center, z.ratio, yerr=z.stat_error,
                        fmt=MARKERS[tag] + "-", ms=4.2, lw=1.0,
                        color=COLORS[tag], capsize=1.8,
                        label=rf"$U_{{\rm sym}}={U_MEV[tag]}$ MeV")
            common = z.set_index("bin_center").join(
                base[["ratio", "stat_error"]], rsuffix="_0", how="inner")
            ratio = common.ratio / common.ratio_0
            if tag == "zeroSpot":
                err = np.zeros_like(ratio)
            else:
                err = ratio * np.sqrt((common.stat_error / common.ratio) ** 2
                                      + (common.stat_error_0 / common.ratio_0) ** 2)
            ar.errorbar(common.index, ratio, yerr=err,
                        fmt=MARKERS[tag] + "-", ms=3.8, lw=1.0,
                        color=COLORS[tag], capsize=1.5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(r"primary-nucleon yield ratio $n/p$")
        ax.grid(alpha=0.16)
        ax.legend(fontsize=8.5, loc="best")
        ax.tick_params(labelbottom=False)
        ar.axhline(1, color="0.45", lw=1)
        ar.set(xlabel=r"$E_{\rm kin}$ [GeV]",
               ylabel=r"ratio to $U_{\rm sym}=0$", ylim=(0.94, 1.10))
        ar.grid(alpha=0.16)
        stamp(ax, "Xe+Cs, 2.5 AGeV, full production")
    fig.suptitle("Global n/p feasibility: strong differential structure, weak integrated leverage",
                 fontsize=13, y=0.995)
    save(fig, out, "fig01_np_global_ekin")


def _acceptance_job_spectra(npz_path: Path, coarse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    efine = z["_e_edges"]
    fine_lo, fine_hi = efine[:-1], efine[1:]
    keys = [k for k in z.files if k.endswith("|acc")]
    out = np.zeros((len(keys), 2, len(coarse) - 1), dtype=float)
    for jf, key in enumerate(keys):
        a = z[key]
        for b in range(len(coarse) - 1):
            # Only merge complete native bins.  Assigning bins by their centre
            # created a false empty interval at 2.2--2.6 GeV because the native
            # logarithmic binning has no centre in that range.
            take = ((fine_lo >= coarse[b] - 1e-8)
                    & (fine_hi <= coarse[b + 1] + 1e-8))
            if not np.any(take):
                raise ValueError(
                    f"coarse bin {coarse[b]:.6g}--{coarse[b + 1]:.6g} "
                    "contains no complete native bins"
                )
            out[jf, :, b] = a[:, take].sum(axis=1)
    return out, np.asarray(keys)


def _ratio_boot(a: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = a.sum(axis=0)
    central = np.divide(total[0], total[1], out=np.full(total.shape[1], np.nan), where=total[1] > 0)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(a), len(a))
        s = a[pick].sum(axis=0)
        boots.append(np.divide(s[0], s[1], out=np.full(s.shape[1], np.nan), where=s[1] > 0))
    boots = np.asarray(boots)
    finite = np.isfinite(boots).sum(axis=0)
    err = np.array([
        np.nanstd(boots[:, i], ddof=1) if finite[i] > 1 else np.nan
        for i in range(boots.shape[1])
    ])
    return central, err, boots


def plot_acceptance_np(data_dir: Path, out: Path, n_boot: int, metrics: dict) -> None:
    # These are exact edges of the stored logarithmic histogram.  Rebinning on
    # native boundaries makes every plotted interval contiguous and conserves
    # all entries without interpolation.
    coarse = np.array([
        0.0, 0.291172894, 0.465505191, 0.744214476, 1.18979379,
        1.50438325, 1.90215228, 2.40509411, 3.04101714,
        3.84508248, 4.86174810,
    ])
    x = 0.5 * (coarse[:-1] + coarse[1:])
    xe = 0.5 * (coarse[1:] - coarse[:-1])
    rng = np.random.default_rng(20260928)
    vals, errs, boots = {}, {}, {}
    for tag in SAMPLES:
        a, keys = _acceptance_job_spectra(data_dir / f"{tag}_filehist.npz", coarse)
        vals[tag], errs[tag], boots[tag] = _ratio_boot(a, n_boot, rng)
        metrics.setdefault("acceptance_spectra", {})[tag] = {
            "n_job_files": int(len(keys)), "n": a[:, 0].sum(axis=0).tolist(),
            "p": a[:, 1].sum(axis=0).tolist(), "np": vals[tag].tolist(),
            "np_job_bootstrap_error": errs[tag].tolist(),
        }

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.15], hspace=0.05)
    ax, ar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    keep = x <= 5.0
    for tag in SAMPLES:
        ax.errorbar(x[keep], vals[tag][keep], xerr=xe[keep], yerr=errs[tag][keep],
                    fmt=MARKERS[tag] + "-", ms=4, lw=1, color=COLORS[tag], capsize=2,
                    label=rf"$U_{{\rm sym}}={U_MEV[tag]}$ MeV")
    dr = np.divide(vals["bigSpot"], vals["zeroSpot"],
                   out=np.full_like(vals["bigSpot"], np.nan),
                   where=vals["zeroSpot"] > 0)
    drb = np.divide(boots["bigSpot"], boots["zeroSpot"],
                    out=np.full_like(boots["bigSpot"], np.nan),
                    where=boots["zeroSpot"] > 0)
    finite = np.isfinite(drb).sum(axis=0)
    dre = np.array([
        np.nanstd(drb[:, i], ddof=1) if finite[i] > 1 else np.nan
        for i in range(drb.shape[1])
    ])
    ar.errorbar(x[keep], dr[keep], xerr=xe[keep], yerr=dre[keep], fmt="o-",
                ms=4, color="#7F3C8D", capsize=2)
    ar.axhline(1, color="0.4", lw=1)
    ax.set_ylabel(r"primary-nucleon yield ratio $n/p$")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=8.5)
    ax.tick_params(labelbottom=False)
    ar.set(xlabel=r"$E_{\rm kin}$ [GeV]", ylabel=r"$(n/p)_{90}/(n/p)_0$")
    ar.grid(alpha=0.16)
    stamp(ax, "exact HGND front-face acceptance; job-file bootstrap")
    fig.suptitle("Differential n/p in the HGND acceptance", fontsize=13, y=0.995)
    # Version the corrected native-boundary rendering so Markdown and notebook
    # front ends cannot reuse a cached image under the former filename.
    save(fig, out, "fig02_np_hgnd_acceptance_ekin_v2")


def _hybrid_response(dn: float, en: float, dp: float, ep: float) -> tuple[float, float]:
    """Return (n response)/(p response)-1 with independent-error propagation."""
    a, b = 1.0 + dn, 1.0 + dp
    value = a / b - 1.0
    error = (a / b) * np.sqrt((en / a) ** 2 + (ep / b) ** 2)
    return float(value), float(error)


def _mean_error(values: pd.Series) -> tuple[float, float]:
    values = values.astype(float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def _ratio_error(numerator: pd.Series, denominator: pd.Series) -> tuple[float, float]:
    """Ratio of event means with an event-level delta-method uncertainty."""
    high = numerator.astype(float)
    low = denominator.astype(float)
    ratio = float(high.mean() / low.mean())
    influence = high - ratio * low
    error = float(influence.std(ddof=1) / np.sqrt(len(influence)) / low.mean())
    return ratio, error


def _relative_response(
    zero: tuple[float, float], big: tuple[float, float]
) -> tuple[float, float]:
    value = big[0] / zero[0] - 1.0
    error = (1.0 + value) * np.sqrt(
        (big[1] / big[0]) ** 2 + (zero[1] / zero[0]) ** 2
    )
    return float(value), float(error)


def summarise_reconstructed_observables(
    pred_dir: Path, threshold: float = 0.45
) -> dict:
    """Build event-clustered observables from one frozen reconstruction model."""
    per_sample: dict[str, dict] = {}
    for tag in SAMPLES:
        path = pred_dir / tag / "pred_clusters_smash.pkl"
        d = pd.read_pickle(path)
        required = {"Row", "cl_score", "e_pred", "cl_label"}
        if not required.issubset(d.columns):
            raise ValueError(f"{path} lacks columns {sorted(required - set(d.columns))}")
        rows = pd.Index(np.sort(d["Row"].unique()), name="Row")
        selected = d[d["cl_score"] > threshold].copy()

        def counts(mask: pd.Series | np.ndarray) -> pd.Series:
            c = selected.loc[mask].groupby("Row").size()
            return c.reindex(rows, fill_value=0).astype(float)

        def energy_sum(mask: pd.Series | np.ndarray) -> pd.Series:
            e = selected.loc[mask].groupby("Row")["e_pred"].sum()
            return e.reindex(rows, fill_value=0).astype(float)

        candidate = counts(np.ones(len(selected), dtype=bool))
        matched = counts(selected["cl_label"].eq(1))
        differential = counts(
            selected["e_pred"].ge(2.6) & selected["e_pred"].lt(3.0)
        ) / 0.4
        high = counts(selected["e_pred"].ge(2.6))
        low = counts(selected["e_pred"].ge(1.8) & selected["e_pred"].lt(2.4))
        band_1_2 = selected["e_pred"].ge(1.0) & selected["e_pred"].lt(2.0)
        band_count = counts(band_1_2)
        band_energy = energy_sum(band_1_2)
        equal_low = counts(selected["e_pred"].ge(1.0) & selected["e_pred"].lt(1.5))
        equal_high = counts(selected["e_pred"].ge(1.5) & selected["e_pred"].lt(2.0))
        tail_low = counts(selected["e_pred"].ge(1.0) & selected["e_pred"].lt(1.8))
        tail_high = counts(selected["e_pred"].ge(1.8) & selected["e_pred"].lt(2.0))
        per_sample[tag] = {
            "prediction_file": str(path),
            "n_cluster_rows": int(len(d)),
            "n_events": int(len(rows)),
            "n_selected": int(len(selected)),
            "selected_purity": float(selected["cl_label"].mean()),
            "candidate_yield": _mean_error(candidate),
            "truth_matched_yield": _mean_error(matched),
            "differential_yield_2p6_3p0": _mean_error(differential),
            "safe_hardness": _ratio_error(high, low),
            "band_hardness_1p0_1p5_vs_1p5_2p0": _ratio_error(equal_high, equal_low),
            "band_hardness_1p0_1p8_vs_1p8_2p0": _ratio_error(tail_high, tail_low),
            "band_mean_energy_1p0_2p0": _ratio_error(band_energy, band_count),
            "safe_window_counts": {
                "low_1p8_2p4": int(low.sum()),
                "high_ge_2p6": int(high.sum()),
            },
            "band_window_counts": {
                "low_1p0_1p5": int(equal_low.sum()),
                "high_1p5_2p0": int(equal_high.sum()),
                "low_1p0_1p8": int(tail_low.sum()),
                "high_1p8_2p0": int(tail_high.sum()),
                "all_1p0_2p0": int(band_count.sum()),
            },
        }

    responses = {}
    for key in (
        "candidate_yield", "truth_matched_yield",
        "differential_yield_2p6_3p0", "safe_hardness",
        "band_hardness_1p0_1p5_vs_1p5_2p0",
        "band_hardness_1p0_1p8_vs_1p8_2p0",
        "band_mean_energy_1p0_2p0",
    ):
        responses[key] = _relative_response(
            per_sample["zeroSpot"][key], per_sample["bigSpot"][key]
        )
    return {
        "definition": {
            "model": "single frozen HGND GNN evaluated on all three SMASH samples",
            "score_selection": f"cl_score > {threshold}",
            "candidate_yield": "all selected clusters per represented event; includes fakes",
            "truth_matched_yield": "selected cl_label == 1 per represented event; closure diagnostic",
            "differential_yield_2p6_3p0": "selected candidates per event per GeV for 2.6 <= e_pred < 3.0 GeV",
            "safe_hardness": "N(e_pred >= 2.6 GeV) / N(1.8 <= e_pred < 2.4 GeV)",
            "band_hardness_1p0_1p5_vs_1p5_2p0": "N(1.5 <= e_pred < 2.0 GeV) / N(1.0 <= e_pred < 1.5 GeV)",
            "band_hardness_1p0_1p8_vs_1p8_2p0": "N(1.8 <= e_pred < 2.0 GeV) / N(1.0 <= e_pred < 1.8 GeV)",
            "band_mean_energy_1p0_2p0": "mean e_pred of selected candidates with 1.0 <= e_pred < 2.0 GeV",
            "uncertainty": "event-level standard error; independent-sample propagation for 90/0",
        },
        "samples": per_sample,
        "responses": {
            key: {"relative_change": value, "error": error}
            for key, (value, error) in responses.items()
        },
    }


def plot_separation_power(
    acc_report: dict,
    acceptance_summary: dict,
    reco_summary: dict,
    proton_window_report: dict,
    out: Path,
    metrics: dict,
) -> None:
    rows: list[dict[str, object]] = []

    def add(label: str, response: float, error: float, level: str, note: str) -> None:
        rows.append({
            "observable": label,
            "response_percent": 100.0 * response,
            "error_percent": 100.0 * error,
            "separation_sigma": abs(response) / error if error > 0 else np.nan,
            "level": level,
            "note": note,
        })

    integrated = acc_report["integrated"]
    angular_yield = acceptance_summary["hgnd_band"]["zero_to_big_neutron_yield"]
    angular_response = float(angular_yield["relative_change"])
    angular_significance = float(angular_yield["significance"])
    add(
        r"truth differential $n$ yield",
        angular_response,
        abs(angular_response) / angular_significance,
        "truth",
        r"per-event yield in $8.9^\circ<\theta<13.1^\circ$",
    )
    for key, label in (
        ("np_acc", r"truth yield $n/p$"),
        ("R_n_acc", r"truth hardness $R_n$"),
        ("R_p_acc", r"truth hardness $R_p$"),
        ("Rn_over_Rp_acc", r"truth double ratio $R_n/R_p$"),
    ):
        z = integrated[key]
        add(label, z["rel_90_over_0"], z["rel_err"], "truth", "exact HGND acceptance")

    for key, label, note in (
        ("candidate_yield", r"reco candidate yield $\widehat Y_n$",
         "all selected clusters; includes fakes"),
        ("truth_matched_yield", r"reco matched yield $\widehat Y_n^{\rm match}$",
         "truth-matched closure diagnostic"),
        ("differential_yield_2p6_3p0", r"reco $d\widehat N_n/dE$",
         r"selected candidates; $2.6\leq E_{\rm reco}<3.0$ GeV"),
        ("safe_hardness", r"reco safe-window $\widehat R_n$",
         r"$E_{\rm reco}\in[1.8,2.4)$ vs. $E_{\rm reco}\geq2.6$ GeV"),
        ("band_hardness_1p0_1p5_vs_1p5_2p0", r"reco $R_n^{1{-}2}$ (equal bins)",
         r"$[1.5,2.0)/[1.0,1.5)$ GeV; not 0/18/90 ordered; exploratory"),
        ("band_hardness_1p0_1p8_vs_1p8_2p0", r"reco $R_n^{1{-}2}$ (upper tail)",
         r"$[1.8,2.0)/[1.0,1.8)$ GeV; 0/18/90 ordered; exploratory"),
        ("band_mean_energy_1p0_2p0", r"reco $\langle E\rangle_{1{-}2}$",
         r"mean selected-candidate energy in $[1.0,2.0)$ GeV; not ordered; exploratory"),
    ):
        z = reco_summary["responses"][key]
        add(label, z["relative_change"], z["error"], "reconstructed", note)

    reco_yield = reco_summary["responses"]["candidate_yield"]
    proton_yield = acceptance_summary["hgnd_band"]["zero_to_big_proton_yield"]
    dp_yield = float(proton_yield["relative_change"])
    ep_yield = abs(dp_yield) / float(proton_yield["significance"])
    value, error = _hybrid_response(
        reco_yield["relative_change"], reco_yield["error"], dp_yield, ep_yield
    )
    add(
        r"hybrid yield $\widehat Y_n/Y_p^{\rm MC}$", value, error, "hybrid",
        "reco candidates and MC-truth proton yield; independent-error estimate",
    )

    reco_hardness = reco_summary["responses"]["safe_hardness"]
    dp = integrated["R_p_acc"]["rel_90_over_0"]
    ep = integrated["R_p_acc"]["rel_err"]
    value, error = _hybrid_response(
        reco_hardness["relative_change"], reco_hardness["error"], dp, ep
    )
    add(
        r"hybrid $\widehat R_n/R_p^{\rm MC}$", value, error, "hybrid",
        "safe-window reco neutron and MC-truth proton; independent-error estimate",
    )

    proton_shapes = proton_window_report["b_reweighted"]["observables"]
    for neutron_key, proton_key, label, note in (
        (
            "band_hardness_1p0_1p5_vs_1p5_2p0",
            "R_p_1p0_1p5_vs_1p5_2p0",
            r"hybrid $R_{n,{\rm eq}}^{1{-}2}/R_{p,{\rm eq}}^{1{-}2,{\rm MC}}$",
            "equal-bin reco-neutron/MC-proton shape ratio",
        ),
        (
            "band_hardness_1p0_1p8_vs_1p8_2p0",
            "R_p_1p0_1p8_vs_1p8_2p0",
            r"hybrid $R_{n,{\rm tail}}^{1{-}2}/R_{p,{\rm tail}}^{1{-}2,{\rm MC}}$",
            "upper-tail reco-neutron/MC-proton shape ratio",
        ),
        (
            "band_mean_energy_1p0_2p0",
            "mean_E_p_1p0_2p0",
            r"hybrid $\langle E_n\rangle_{1{-}2}/\langle E_p\rangle_{1{-}2}^{\rm MC}$",
            "reco-neutron/MC-proton in-band mean-energy ratio",
        ),
    ):
        n = reco_summary["responses"][neutron_key]
        p = proton_shapes[proton_key]
        value, error = _hybrid_response(
            n["relative_change"], n["error"],
            p["relative_change_90_over_0"], p["relative_error"],
        )
        add(label, value, error, "hybrid",
            note + "; independent-error estimate; exploratory")

    table = pd.DataFrame(rows)
    table.to_csv(out / "observable_separation_power.csv", index=False)
    metrics["observable_separation_power"] = table.to_dict(orient="records")

    colors = {
        "truth": "#2166AC",
        "reconstructed": "#D95F02",
        "hybrid": "#7F3C8D",
    }
    fig, ax = plt.subplots(figsize=(10.4, 10.2))
    y = np.arange(len(table))[::-1]
    for i, row in table.iterrows():
        yi = y[i]
        ax.errorbar(
            row.response_percent, yi, xerr=row.error_percent,
            fmt="o", ms=5.5, color=colors[row.level], capsize=3,
        )
    ax.axvline(0, color="0.4", lw=1)
    ax.set_yticks(y, table.observable)
    ax.set_xlabel(r"relative response $90/0-1$ [\%]")
    ax.set_xlim(-8.2, 13.8)
    ax.grid(axis="x", alpha=0.18)
    # Keep significance labels in a fixed, dedicated column.  Positioning them
    # beside the points made the large truth effects collide with the frame and
    # the low-significance reconstruction rows collide with the legend.
    for i, row in table.iterrows():
        ax.text(
            0.985, y[i], f"{row.separation_sigma:.1f}$\\sigma$",
            transform=ax.get_yaxis_transform(), va="center", ha="right",
            fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
        )
    ax.text(
        0.985, 1.015, r"$|\Delta|/\sigma$", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5, color="0.25",
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=k)
        for k, c in colors.items()
    ]
    ax.legend(
        handles=handles, frameon=False, loc="upper center",
        bbox_to_anchor=(0.55, 1.12), ncol=3,
    )
    stamp(ax, "one frozen GNN and q>0.45 for every reconstructed entry")
    fig.suptitle("Truth and reconstructed observable separation power", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig06_observable_separation_power_v4")


def plot_reconstruction_performance(performance: dict, out: Path) -> None:
    """Paper-style SMASH performance summary for the single current model."""
    scan = pd.DataFrame(performance["threshold_scan"])
    energy = pd.DataFrame(performance["energy_performance"])
    threshold = float(performance["working_threshold"])
    wp = scan.iloc[(scan.threshold - threshold).abs().argmin()]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1))
    ax = axes[0]
    ax.plot(scan.efficiency_count, scan.purity_count, "o-", ms=3.5,
            color="#2166AC", label="cluster count")
    ax.plot(scan.efficiency_energy, scan.purity_energy, "s--", ms=3.2,
            color="#B2182B", label="energy weighted")
    ax.plot(wp.efficiency_count, wp.purity_count, marker="*", ms=15,
            color="#222222", label=rf"working point $q>{threshold:.2f}$")
    ax.set(xlabel="signal efficiency", ylabel="purity", xlim=(0.15, 1.0), ylim=(0.4, 1.01))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhspan(-0.10, 0.10, color="0.85", zorder=0, label=r"$\pm10\%$")
    ax.axhline(0, color="0.35", lw=1)
    ax.plot(energy.e_mid, 100 * energy.linearity, "o-", color="#2166AC")
    ax.set(xlabel=r"true neutron $E_{\rm kin}$ [GeV]", ylabel="median energy bias [%]")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.axhspan(0, 10, color="0.85", zorder=0, label=r"$10\%$ target")
    ax.plot(energy.e_mid, 100 * energy.resolution, "o-", color="#B2182B")
    ax.set(xlabel=r"true neutron $E_{\rm kin}$ [GeV]", ylabel="robust energy resolution [%]")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.16)
    stamp(axes[0], rf"SMASH; one model; ROC AUC={performance['roc_auc']:.3f}")
    fig.suptitle("HGND neutron reconstruction performance on SMASH", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, out, "fig08_smash_reconstruction_performance")


def plot_differential_yield_np(acceptance_summary: dict, out: Path) -> None:
    """Compare angular neutron-yield and n/p responses in the HGND region."""
    datasets = acceptance_summary["datasets"]
    zero = datasets["zeroSpot"]["theta"]
    lo = np.array([row["theta_lo"] for row in zero], dtype=float)
    hi = np.array([row["theta_hi"] for row in zero], dtype=float)
    x = 0.5 * (lo + hi)
    xe = np.vstack((x - lo, hi - x))
    keep = hi <= 40

    fig, (ay, ar) = plt.subplots(2, 1, figsize=(7.6, 6.3), sharex=True)
    for tag in ("defaultSpot", "bigSpot"):
        rows = datasets[tag]["theta"]
        n0 = np.array([row["n_per_event"] for row in zero], dtype=float)
        sn0 = np.array([row["n_per_event_error"] for row in zero], dtype=float)
        n = np.array([row["n_per_event"] for row in rows], dtype=float)
        sn = np.array([row["n_per_event_error"] for row in rows], dtype=float)
        yn = 100.0 * (n / n0 - 1.0)
        en = 100.0 * (n / n0) * np.sqrt((sn / n) ** 2 + (sn0 / n0) ** 2)

        r0 = np.array([row["ratio"] for row in zero], dtype=float)
        sr0 = np.array([row["ratio_error"] for row in zero], dtype=float)
        r = np.array([row["ratio"] for row in rows], dtype=float)
        sr = np.array([row["ratio_error"] for row in rows], dtype=float)
        yr = 100.0 * (r / r0 - 1.0)
        er = 100.0 * (r / r0) * np.sqrt((sr / r) ** 2 + (sr0 / r0) ** 2)

        label = rf"$U_{{\rm sym}}={U_MEV[tag]}$ MeV"
        kw = dict(
            xerr=xe[:, keep], fmt=MARKERS[tag] + "-", ms=4.2, lw=1.0,
            color=COLORS[tag], capsize=1.8, label=label,
        )
        ay.errorbar(x[keep], yn[keep], yerr=en[keep], **kw)
        ar.errorbar(x[keep], yr[keep], yerr=er[keep], **kw)

    for ax in (ay, ar):
        ax.axhline(0, color="0.4", lw=1)
        ax.axvspan(8.9, 13.1, color="#F2C14E", alpha=0.24)
        ax.grid(alpha=0.16)
    ay.set_ylabel(r"neutron yield response to $U_{\rm sym}=0$ [\%]")
    ar.set_ylabel(r"$n/p$ response to $U_{\rm sym}=0$ [\%]")
    ar.set_xlabel(r"polar angle $\theta_{\rm lab}$ [deg]")
    ar.set_xlim(0, 40)
    ay.legend(fontsize=8.5, ncol=2)
    stamp(ay, "full production; event-clustered errors; HGND angular band shaded")
    fig.suptitle("Differential neutron yield and n/p response", fontsize=13, y=0.995)
    fig.tight_layout()
    save(fig, out, "fig07_differential_neutron_yield_np_theta")


def _integrated(report: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    z = report["integrated"][key]
    return (np.array([z["central"][tag] for tag in SAMPLES]),
            np.array([z["err"][tag] for tag in SAMPLES]))


def plot_hardness(global_report: dict, acc_report: dict, out: Path) -> None:
    u = np.array([U_MEV[t] for t in SAMPLES])
    g, ge = _integrated(global_report, "R_n_4pi")
    a, ae = _integrated(acc_report, "R_n_acc")
    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.15], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ar = fig.add_subplot(gs[1], sharex=ax)
    ax.errorbar(u, g, yerr=ge, fmt="o-", color="#4D4D4D", capsize=3,
                label=r"global $4\pi$")
    ax.errorbar(u, a, yerr=ae, fmt="s-", color="#D95F02", capsize=3,
                label="HGND front face")
    for y, e, c, mk in ((g, ge, "#4D4D4D", "o"), (a, ae, "#D95F02", "s")):
        r = y / y[0]
        re = r * np.sqrt((e / y) ** 2 + (e[0] / y[0]) ** 2)
        ar.errorbar(u, r, yerr=re, fmt=mk + "-", color=c, capsize=3)
    ax.set_ylabel(r"neutron hardness $R_n=N(E>2)/N(E<1)$")
    ax.grid(alpha=0.16)
    ax.legend()
    ax.tick_params(labelbottom=False)
    ar.axhline(1, color="0.4", lw=1)
    ar.set(xlabel=r"$U_{\rm sym}$ [MeV]", ylabel=r"$R_n/R_n(0)$")
    ar.grid(alpha=0.16)
    stamp(ax, "b-reweighted full production; job-file bootstrap")
    fig.suptitle("Spectral hardness localises the symmetry-potential response",
                 fontsize=13, y=0.995)
    save(fig, out, "fig03_neutron_hardness_global_vs_hgnd")


def plot_observable_summary(global_report: dict, acc_report: dict, out: Path) -> None:
    u = np.array([U_MEV[t] for t in SAMPLES])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    series = [
        (global_report, "np_4pi", r"$n/p$, global $4\pi$", "#4D4D4D", "o"),
        (acc_report, "np_acc", r"$n/p$, HGND front face", "#D95F02", "s"),
    ]
    for rep, key, lab, c, mk in series:
        y, e = _integrated(rep, key)
        r = y / y[0]
        re = r * np.sqrt((e / y) ** 2 + (e[0] / y[0]) ** 2)
        ax0.errorbar(u, r, yerr=re, fmt=mk + "-", color=c, capsize=3, label=lab)
    for key, lab, c, mk in (
        ("R_n_acc", r"$R_n$", "#2166AC", "o"),
        ("R_p_acc", r"$R_p$", "#B2182B", "s"),
        ("Rn_over_Rp_acc", r"$R_n/R_p$", "#7F3C8D", "^"),
    ):
        y, e = _integrated(acc_report, key)
        r = y / y[0]
        re = r * np.sqrt((e / y) ** 2 + (e[0] / y[0]) ** 2)
        ax1.errorbar(u, r, yerr=re, fmt=mk + "-", color=c, capsize=3, label=lab)
    for ax in (ax0, ax1):
        ax.axhline(1, color="0.45", lw=1)
        ax.set_xlabel(r"$U_{\rm sym}$ [MeV]")
        ax.set_ylabel("ratio to 0 MeV")
        ax.grid(alpha=0.16)
        ax.legend(fontsize=8.5)
    ax0.set_title("Yield ratio: small or null response")
    ax1.set_title("Spectral shape: monotonic isovector response")
    stamp(ax0, "full production")
    stamp(ax1, "exact HGND front-face acceptance")
    fig.suptitle("Observable choice, not event count, determines feasibility",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, out, "fig04_observable_feasibility_summary")


def plot_centrality(acc_report: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for key, lab, c, mk in (
        ("R_n_acc", r"$R_n$", "#2166AC", "o"),
        ("R_p_acc", r"$R_p$", "#B2182B", "s"),
        ("Rn_over_Rp_acc", r"$R_n/R_p$", "#7F3C8D", "^"),
        ("np_acc", r"$n/p$ yield", "#4D4D4D", "D"),
    ):
        rows = acc_report["classes"][key]
        x = np.array([(r["b_lo"] + r["b_hi"]) / 2 for r in rows])
        y = 100 * np.array([r["rel_90_over_0"] for r in rows])
        e = 100 * np.array([r["rel_err"] for r in rows])
        ax.errorbar(x, y, yerr=e, fmt=mk + "-", color=c, ms=4, capsize=2, label=lab)
    ax.axhline(0, color="0.4", lw=1)
    ax.set(xlabel=r"impact parameter class centre $b$ [fm]",
           ylabel=r"relative response $(90/0-1)$ [%]")
    ax.grid(alpha=0.16)
    ax.legend(ncol=2, fontsize=8.5)
    stamp(ax, "HGND front face; b reweighted within job bootstrap")
    fig.suptitle("Centrality dependence of the truth-level response", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig05_hgnd_response_vs_centrality")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-csv", type=Path,
                    default=Path("results/np_ratio_smash_check_full/np_ratio_binned.csv"))
    ap.add_argument("--acceptance-data", type=Path,
                    default=Path("results/update_note/data"))
    ap.add_argument("--global-report", type=Path,
                    default=Path("results/b_full_analysis/b_reweight.json"))
    ap.add_argument("--acceptance-report", type=Path,
                    default=Path("results/acceptance_full/b_reweight.json"))
    ap.add_argument("--acceptance-summary", type=Path,
                    default=Path("results/np_ratio_acceptance_full/acceptance_full.json"))
    ap.add_argument("--reco-pred-dir", type=Path,
                    default=Path("results/update_note/model_v3"))
    ap.add_argument("--performance-report", type=Path,
                    default=Path("results/retrain_eval/performance_v3_epoch3.json"))
    ap.add_argument("--proton-window-report", type=Path,
                    default=Path("results/update_note/proton_window_observables.json"))
    ap.add_argument("--score-threshold", type=float, default=0.45)
    ap.add_argument("--out-dir", type=Path, default=Path("results/update_note"))
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    metrics: dict = {"n_boot": args.n_boot}
    g = json.loads(args.global_report.read_text())
    a = json.loads(args.acceptance_report.read_text())
    acceptance_summary = json.loads(args.acceptance_summary.read_text())
    performance = json.loads(args.performance_report.read_text())
    proton_window_report = json.loads(args.proton_window_report.read_text())
    reco_summary = summarise_reconstructed_observables(
        args.reco_pred_dir, args.score_threshold
    )
    (args.out_dir / "smash_reconstructed_observables.json").write_text(
        json.dumps(reco_summary, indent=2)
    )
    plot_global_np(args.global_csv, args.out_dir)
    plot_acceptance_np(args.acceptance_data, args.out_dir, args.n_boot, metrics)
    plot_hardness(g, a, args.out_dir)
    plot_observable_summary(g, a, args.out_dir)
    plot_centrality(a, args.out_dir)
    plot_separation_power(
        a, acceptance_summary, reco_summary, proton_window_report,
        args.out_dir, metrics,
    )
    plot_differential_yield_np(acceptance_summary, args.out_dir)
    plot_reconstruction_performance(performance, args.out_dir)
    (args.out_dir / "update_note_plot_data.json").write_text(json.dumps(metrics, indent=1))
    print(f"wrote figures and data to {args.out_dir}")


if __name__ == "__main__":
    main()
