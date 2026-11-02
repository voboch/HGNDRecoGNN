"""Deterministic kinematic estimates and figures; no MC events or model outputs."""
from pathlib import Path
import hashlib
import json
from math import erfc, sqrt
from statistics import NormalDist
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "HGNDPaper/scripts"))
from HGNDRecoGNN.analysis import neutron_tof_bounds as tof  # noqa: E402
from HGNDRecoGNN.scripts import hgnd_acceptance as geometry  # noqa: E402
from plot_style import apply_style  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    apply_style()
    out = HERE / "first_principles_figures"
    out.mkdir(exist_ok=True)
    # The centre of the outer analysis rectangle is only a reference ray.
    # It may cross the inter-module gap; no active-face claim is made here.
    z = geometry.Z_FRONT / 100
    x = (geometry.X_LO + geometry.X_HI) / 200
    length = float(np.hypot(z, x))
    near = float(np.hypot(z, geometry.X_HI / 100))
    far = float(np.sqrt(z*z + (geometry.X_LO / 100)**2 + (geometry.Y_HI / 100)**2))
    sigma, halfwidth = .15, .3  # Existing single-hit jitter and prompt window, in ns.
    energies = np.array([.1, .2, .3, .5, 1., 2., 3., 4.])
    table = [{"kinetic_GeV": float(e), "tof_ns": float(tof.flight_time_ns(e, length)),
              "timing_sigma_fraction": float(tof.timing_fraction(e, length, sigma)),
              "extra_path_for_0p3ns_cm": float(tof.beta(e) * tof.C_M_PER_NS * halfwidth * 100)}
             for e in energies]
    gates = [{"length_m": float(trial_length), "time_max_ns": float(t),
              "minimum_kinetic_GeV": float(tof.kinetic_from_time_GeV(t, trial_length))}
             for trial_length in (5., 7., length) for t in (35., 40., 50., 60.)]
    gap_table = []
    for lo, hi in ((.8, 1.8), (1., 2.), (1.4, 1.8), (1.6, 1.8), (1.6, 2.)):
        separation = float((tof.flight_time_ns(lo, length) - tof.flight_time_ns(hi, length)) / sigma)
        gap_table.append({"low_stop_GeV": lo, "high_start_GeV": hi,
                          "time_gap_in_sigma": separation,
                          "ideal_gaussian_cross_gap_probability_bound": .5 * erfc(separation / sqrt(2))})
    data = {"scope": "Analytic conditional estimates; no MC truth, reconstructed events or learned models",
            "constants": {"neutron_mass_GeV": tof.NEUTRON_MASS_GEV, "c_m_per_ns": tof.C_M_PER_NS},
            "assumptions": {"reference_length_m": length, "front_ray_range_m": [near, far],
                            "single_hit_sigma_ns": sigma, "prompt_halfwidth_ns": halfwidth,
                            "published_time_gate_ns": 35., "known_start_and_straight_constant_speed": True},
            "table": table, "time_gates": gates, "gap_examples": gap_table,
            "35ns_threshold_front_range_GeV": [float(tof.kinetic_from_time_GeV(35, near)),
                                               float(tof.kinetic_from_time_GeV(35, far))],
            "conditional_prompt_floor_with_35ns_GeV": {
                "nominal_7m": float(tof.kinetic_from_time_GeV(35 + halfwidth, 7.)),
                "local_near": float(tof.kinetic_from_time_GeV(35 + halfwidth, near)),
                "reference": float(tof.kinetic_from_time_GeV(35 + halfwidth, length))},
            "prompt_coverage_without_delay": float(tof.prompt_acceptance(.3, 0)),
            "sources": {
                "HGND_operational_time_window": "https://arxiv.org/html/2412.00455v1#S3.SS1",
                "HGND_detector_design": "https://arxiv.org/abs/2309.09610",
                "neutron_mass": "https://pdg.lbl.gov/2025/AtomicNuclearProperties/neutron.html",
                "pi0_mass": "https://pdg.lbl.gov/2025/listings/rpp2025-list-pi-zero.pdf"},
            "presentation_sha256": {"../HGNDPaper/scripts/plot_style.py": sha(ROOT.parent / "HGNDPaper/scripts/plot_style.py")},
            "input_sha256": {p: sha(ROOT / p) for p in (
                "analysis/neutron_tof_bounds.py", "scripts/hgnd_acceptance.py",
                "scripts/audit_smash_sources.py", "data/smash_transfer.py",
                "reports/energy_label_optimization_20261029/first_principles.py")}}
    # Equal-mass NN->NN pi0 threshold is an onset of an additional channel,
    # not the threshold for elastic recoil or neutron registration.
    pion_mass = .1349768  # PDG2025 pi0 mass; GeV, used only in this illustrative threshold.
    data["equal_mass_NN_pi0_threshold_GeV"] = 2*pion_mass + pion_mass**2 / (2*tof.NEUTRON_MASS_GEV)
    (HERE / "first_principles_numbers.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    def save(fig, name):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        undrawn_labels = set()
        for ax in fig.axes:
            for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    # LogLocator creates extra ticks outside the view; Axis.draw
                    # does not render those labels even if their visible flag is true.
                    if not min(limits) <= tick.get_loc() <= max(limits):
                        undrawn_labels.update((tick.label1, tick.label2))
            for line in ax.lines:
                if line.get_transform() == ax.transData:
                    xx, yy = line.get_data()
                    assert np.all((np.asarray(xx) >= min(ax.get_xlim())) & (np.asarray(xx) <= max(ax.get_xlim())))
                    assert np.all((np.asarray(yy) >= min(ax.get_ylim())) & (np.asarray(yy) <= max(ax.get_ylim())))
        for text in fig.findobj(matplotlib.text.Text):
            if text not in undrawn_labels and text.get_visible() and text.get_text():
                box = text.get_window_extent(renderer)
                if box.width and box.height:
                    assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, text.get_text()
        fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out / f"{name}.png", dpi=180)
        plt.close(fig)

    def figure(title):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        fig.suptitle(title + "\nAnalytic model; no MC or reconstructed events", fontsize=11)
        return fig, axes

    energy = np.geomspace(.05, 4, 600)  # Display spans soft audit and published upper design range.
    fig, axes = figure("Relativistic time of flight: known start, straight constant-speed path")
    ax = axes[0]
    ax.plot(energy, tof.flight_time_ns(energy, length), color=".15", label=f"Reference ray: L={length:.3f} m")
    ax.fill_between(energy, tof.flight_time_ns(energy, near), tof.flight_time_ns(energy, far),
                    color=".65", alpha=.35, label="Outer front-ray range; not an error band")
    ax.axhline(35, color="#B22222", ls="--", label="Published 35 ns time selection")
    ax.axvline(.3, color="#1f77b4", ls=":", label="Historical label floor: 0.3 GeV")
    ax.set(xlabel="Neutron kinetic energy [GeV]", ylabel="Flight time [ns]", xscale="log")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(energy, 100 * tof.timing_fraction(energy, length, sigma), color=".15", label="150 ps: single-hit timing term")
    ax.plot(energy, 100 * tof.timing_fraction(energy, length, sqrt(2)*sigma), color="#1f77b4", ls="--",
            label="212 ps: hypothetical equal start + hit SD")
    ax.axvline(.3, color="#B22222", ls=":")
    ax.set(xlabel="Neutron kinetic energy [GeV]", ylabel="Local timing contribution to sigma(T)/T [%]", xscale="log")
    ax.legend(fontsize=8)
    save(fig, "tof_and_timing")

    fig, axes = figure("A prompt label and an energy-deposition cut constrain different physics")
    # beta<1 gives the natural display scale c*w, about 9 cm; these are
    # hypothetical additional paths, not fitted or measured shower lengths.
    path = np.linspace(0, tof.C_M_PER_NS * halfwidth, 400)
    for shade, (e, style) in zip(np.linspace(.45, .95, 4), ((.1, "-"), (.3, "--"), (1., "-."), (3., ":"))):
        axes[0].plot(path * 100, tof.prompt_acceptance(e, path), color=plt.cm.Blues(shade),
                     ls=style, label=f"T={e:g} GeV")
    axes[0].set(xlabel="Hypothetical extra constant-speed path [cm]", ylabel="Single-hit prompt acceptance", yscale="log")
    axes[0].legend(title="150 ps jitter; +/-0.3 ns label", fontsize=8, title_fontsize=8)
    recoil_energy = np.geomspace(.001, .6, 400)
    axes[1].plot(recoil_energy, tof.elastic_recoil_max_GeV(recoil_energy), color=".15",
                 label="Free elastic n-p, equal masses: max recoil = T")
    axes[1].axhline(.003, color="#B22222", ls="--", label="3 MeV deposited-energy selection")
    axes[1].axvline(.3, color="#1f77b4", ls=":", label="Historical 300 MeV label condition")
    axes[1].set(xlabel="Incident neutron kinetic energy [GeV]", ylabel="Kinematically allowed maximum recoil [GeV]",
                xscale="log", yscale="log")
    axes[1].legend(fontsize=7.5, loc="upper left")
    save(fig, "label_path_and_recoil")

    fig, axes = figure("Time acceptance and ideal migration constrain candidate energy bands")
    time = np.linspace(30, 65, 400)  # Brackets 35 ns and the direct 0.1 GeV ToF; not a new DAQ setting.
    for shade, (trial_length, style) in zip(np.linspace(.45, .95, 3), ((5., ":"), (7., "--"), (length, "-"))):
        axes[0].plot(time, tof.kinetic_from_time_GeV(time, trial_length), color=plt.cm.Blues(shade),
                     ls=style, label=f"L={trial_length:.3f} m")
    axes[0].axvline(35, color="#B22222", ls="--")
    axes[0].axhline(.3, color="#1f77b4", ls=":", label="0.3 GeV reference")
    axes[0].set(xlabel="Hypothetical maximum direct ToF [ns]", ylabel="Implied minimum incident energy [GeV]")
    axes[0].legend(fontsize=8)
    high = np.linspace(1.7, 2.4, 400)  # Covers the proposed H grid above these low-stop examples.
    for shade, (low, style) in zip(np.linspace(.45, .95, 3), ((1., "-"), (1.4, "--"), (1.6, ":"))):
        z_gap = (tof.flight_time_ns(low, length) - tof.flight_time_ns(high, length)) / sigma
        axes[1].plot(high, z_gap, color=plt.cm.Blues(shade), ls=style, label=f"Low-band upper edge={low:g} GeV")
    axes[1].axhline(NormalDist().inv_cdf(.99), color="#B22222", ls="-.",
                    label="Illustrative 1% one-sided Gaussian tail")
    axes[1].set(xlabel="High-band lower edge [GeV]", ylabel="Ideal cross-gap time separation / 150 ps")
    axes[1].legend(fontsize=8)
    save(fig, "time_window_and_gap")
    print(json.dumps({"reference_length_m": length, "figures": 3,
                      "T_for_35ns_GeV": float(tof.kinetic_from_time_GeV(35, length)),
                      "empirical_inputs_opened": False}))


if __name__ == "__main__":
    main()
