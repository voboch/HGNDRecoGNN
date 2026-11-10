"""Plot the selection-only objective and its band-count tradeoff."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from plot_b_full import apply_style, mark_provenance  # noqa: E402

result = json.loads((HERE / "choice.json").read_text())
rows = result["scan"]
target = np.array([r["purity_target_percent"] for r in rows])
base = np.flatnonzero(target == 70).item()
best = np.flatnonzero(target == result["chosen_target_percent"]).item()
value = [r.get("contrast", {}).get("value", np.nan) for r in rows]
sd = [r.get("contrast", {}).get("file_sd", np.nan) for r in rows]
totals = np.array([r["counts_potential_repeat_hi_lo_total_signal"] for r in rows])
# Normalize each repeat to its own baseline before the equal-repeat average.
endpoints = totals[:, [0, 2], :, :2].sum(axis=1)
retention = 100 * (endpoints / endpoints[base]).mean(axis=1)

apply_style()
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)
fig, (ax, bands) = plt.subplots(
    2, 1, sharex=True, figsize=(9.2, 5.6), gridspec_kw={"height_ratios": [1.7, 1]}
)
fig.subplots_adjust(left=0.14, right=0.975, top=0.80, bottom=0.15, hspace=0.04)
fig.text(
    0.14, 0.955, "BM@N HGND simulation · SMASH Xe+Cs, 2.87 GeV/nucleon", fontsize=12
)
fig.text(
    0.14,
    0.909,
    "286 even files · calibrated SMASH-trained estimates · fixed energy bands",
    fontsize=11,
)
fig.text(
    0.14,
    0.862,
    "Filled circle: eligible   Diamond: selected   Open circle: exceeds precision bound",
    fontsize=10,
)
for i, row in enumerate(rows):
    color = "#135887" if row["eligible"] else "0.45"
    ax.errorbar(
        target[i],
        value[i],
        yerr=sd[i],
        fmt="D" if i == best else "o",
        color=color,
        mfc=color if row["eligible"] else "white",
        capsize=3,
    )
for axis in (ax, bands):
    axis.axvline(70, ls=":", color="0.45")
    axis.axvline(target[best], ls=":", color="#135887")
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.minorticks_on()
ax.axhline(0, color="0.6", ls="--", lw=0.8)
ax.set_ylabel("Reconstructed 90/0\ncontrast (%)")
mark_provenance(ax, "reco", "upper left")
bands.plot(
    target, retention[:, 0], "s-", color="#a64b28", label=r"High band: $E\geq2$ GeV"
)
bands.plot(
    target,
    retention[:, 1],
    "^--",
    color="#292929",
    label=r"Low band: $0.3\leq E<1$ GeV",
)
bands.set_ylabel("Band counts\n(% of 70% policy)")
bands.set_xlabel("Target count purity on validation (%)")
bands.set_xticks(target)
bands.legend(loc="lower left", fontsize=9)
fig.text(
    0.14,
    0.037,
    "Bars: SD over 400 paired file draws. Band counts pool the 0 and 90 MeV samples.",
    fontsize=10,
)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"score_cut_scan.{ext}", dpi=200)
plt.close(fig)
