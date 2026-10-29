"""Background coefficients for a per-band correction of the reconstructed R_n contrast (validation only).

Protocol: reports/rn_background_correction_protocol_20261029.md (revised after its preflight
review). No training, inference or report-set number: the test files of the odd split parts
are not read. Inputs are the validation predictions of the nine SMASH training runs at their
frozen 0.70 cuts and the frozen DCM model's own validation predictions.

The hard-cluster model corrects the low band as S_L = N_L - c N_H, where c = B_L / N_H is the
low-band background per selected high-band cluster, fitted pooled over the potentials (the
high-band factor cancels in the contrast K). Two questions are measured:

* potential dependence: c(90 MeV) / c(0 MeV) on SMASH validation, translated into the bias of
  the corrected K that a pooled c would leave, about -100 (ratio - 1)(1 - pi_L)/pi_L points;
* generator dependence (A1 only): r = c_DCM / c_SMASH for the same model, cut and calibration,
  its decomposition into background-per-event and hard-clusters-per-event ratios, and the
  fraction of the correction a swap would remove, (1 - r) / (1 - r (1 - pi_L)).

Labels and B_L use MC truth; N_H, N_L and the bands use reconstructed quantities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.scripts import reco_rn_heldout as held  # noqa: E402
from HGNDRecoGNN.scripts import reco_rn_retrospective as retro  # noqa: E402

TAGS, SEEDS = retro.TAGS, retro.SEEDS
ARMS = ("A1", "B1")
BAND = held.BANDS["primary"]
BASE = ROOT / "results/smash_train_20261028"
TRAININGS = BASE / "train_4466016"
TASK = {(f, s): 3 * f + i for f in range(3) for i, s in enumerate(SEEDS)}
NPZ = {"A1": "predictions_A.npz", "B1": "predictions_B.npz"}
# Count columns: selected clusters by reconstructed-energy band, all (N) and MC-positive (S).
NH, NL, SH, SL = 0, 1, 2, 3
# Reproduction guards: the A1 generator ratios were seen while drafting the protocol
# (κ_L 0.49-0.55, κ_H 0.84-0.86 per run); a value outside these bands means the inputs changed.
SEEN = {"kappa_L": (0.45, 0.60), "kappa_H": (0.80, 0.92)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def band_masks(e_pred):
    return e_pred >= BAND["high"], (e_pred >= BAND["low"][0]) & (e_pred < BAND["low"][1])


def counts_of(score, e_pred, label, cut):
    """[N_H, N_L, S_H, S_L] of selected clusters; label is the native MC-truth label."""
    sel = score > cut
    hi, lo = band_masks(e_pred)
    y = label == 1
    return np.array([(sel & hi).sum(), (sel & lo).sum(), (sel & hi & y).sum(), (sel & lo & y).sum()], float)


def coefficients(c, n_ev=None):
    """Coefficients from summed counts c = [N_H, N_L, S_H, S_L] (any leading shape) and events."""
    b_h, b_l = c[..., NH] - c[..., SH], c[..., NL] - c[..., SL]
    out = {"c": b_l / c[..., NH], "kappa_H": b_h / c[..., SH], "kappa_L": b_l / c[..., SH],
           "pi_H": c[..., SH] / c[..., NH], "pi_L": c[..., SL] / c[..., NL]}
    if n_ev is not None:
        out |= {"beta_H": b_h / n_ev, "beta_L": b_l / n_ev, "N_H_per_event": c[..., NH] / n_ev}
    return out


def corrected(c, n_ev, coef, model):
    """Corrected (high, low) counts. M_hard uses only c: the high-band factor cancels in K."""
    if model == "C4":
        return c[..., NH], c[..., NL]
    if model == "C3":
        return c[..., SH], c[..., SL]
    if model == "M_event":
        return c[..., NH] - coef["beta_H"] * n_ev, c[..., NL] - coef["beta_L"] * n_ev
    if model == "M_hard":
        return c[..., NH], c[..., NL] - coef["c"] * c[..., NH]
    raise ValueError(model)


def contrast(high, low):
    """100[R(90)/R(0) - 1] from pooled (high, low) per tag."""
    return 100 * ((high["bigSpot"] / low["bigSpot"]) / (high["zeroSpot"] / low["zeroSpot"]) - 1)


def k_bias(ratio, pi_l):
    """First-order bias of the M_hard contrast (points) when c differs by `ratio` between 90 and 0 MeV."""
    return -100 * (ratio - 1) * (1 - pi_l) / pi_l


def removed_fraction(r, pi_l):
    """Fraction of the M_hard correction (K_hat - C4) lost when c is scaled by r, to first order."""
    return (1 - r) / (1 - r * (1 - pi_l))


# ── inputs ───────────────────────────────────────────────────────────────

def parts(split_path):
    manifest = json.loads(Path(split_path).read_text())
    return {f"{g['tag']}/{c}": g["part"] for g in manifest["groups"] for c in g["contents"]}


def run_record(fold, seed):
    t = TASK[(fold, seed)]
    return TRAININGS / f"extracted/task_{t}/train", json.loads((TRAININGS / f"task_{t}/complete.json").read_text())


def even_part_events(split_parts):
    """Events with at least one candidate cluster per truth-bound even-part file (published per-file counts).

    This is the event definition the DCM validation shares (its predictions list only events with
    clusters); the processed span would also count empty events. Only the files the fits use are
    taken; the odd-part (report-set) files are never opened.
    """
    published = json.loads((BASE / "heldout_rn/per_file_counts.json").read_text())["primary"]["B1"]
    return {r["content"]: int(r["events_with_candidates"]) for tag in TAGS for r in published[tag]
            if split_parts[r["content"]] % 2 == 0}


def validation_counts(fold, seed, arm, members):
    """{content: [N_H, N_L, S_H, S_L]} on the run's truth-bound validation files at its frozen cut."""
    from HGNDRecoGNN.scripts.smash_train import apply_calibration
    directory, complete = run_record(fold, seed)
    d = np.load(directory / "validation" / NPZ[arm], allow_pickle=False)
    e = apply_calibration(complete["calibrations"][arm], d["e_pred"])
    keep = np.isin(d["content"], members)
    frame = pd.DataFrame({"content": d["content"][keep], "score": d["cl_score"][keep], "e": e[keep],
                          "y": d["cl_label"][keep]})
    out = {c: counts_of(g.score.to_numpy(), g.e.to_numpy(), g.y.to_numpy(), complete["cuts"][arm])
           for c, g in frame.groupby("content", sort=True)}
    if sorted(out) != sorted(members):
        raise ValueError(f"Validation files differ from the split part: {arm} fold {fold} seed {seed}")
    return out


def dcm_counts(dcm_dir, fold, seed, calibrated=True):
    """Per-event [N_H, N_L, S_H, S_L] of the frozen A model on DCM validation, with this run's A1 cut."""
    from HGNDRecoGNN.scripts.smash_train import apply_calibration
    _, complete = run_record(fold, seed)
    dcm_complete = json.loads((dcm_dir / f"seed_{seed}/complete.json").read_text())
    if dcm_complete["files"]["checkpoints/model.pt"] != complete["dcm"]["checkpoint_sha256"]:
        raise ValueError(f"DCM validation predictions are not from the A1 checkpoint: seed {seed}")
    f = pd.read_csv(dcm_dir / f"seed_{seed}/predictions.csv")
    e = f.e_pred.to_numpy()
    if calibrated:
        e = apply_calibration(complete["calibrations"]["A1"], e)
    sel = f.cl_score.to_numpy() > complete["cuts"]["A1"]
    hi, lo = band_masks(e)
    y = f.cl_label.to_numpy() == 1
    g = pd.DataFrame({"Row": f.Row, "NH": sel & hi, "NL": sel & lo, "SH": sel & hi & y, "SL": sel & lo & y})
    return g.groupby("Row")[["NH", "NL", "SH", "SL"]].sum().to_numpy(float)


# ── measurement ──────────────────────────────────────────────────────────

def summarize(central, replicas):
    return {"central": float(central), "sd": retro.sd(replicas)}


def potential_dependence(per_tag, ev_tag, rng):
    """Ratios 90/0 MeV of the coefficients, the K bias of a pooled c, and pi_L, with a whole-file bootstrap."""
    by_tag = {t: coefficients(per_tag[t].sum(0), ev_tag[t].sum()) for t in TAGS}
    draws = {t: retro.file_draws(len(per_tag[t]), rng) for t in TAGS}
    rep = {t: coefficients(draws[t] @ per_tag[t], draws[t] @ ev_tag[t]) for t in TAGS}
    pooled = coefficients(sum(v.sum(0) for v in per_tag.values()), sum(v.sum() for v in ev_tag.values()))
    pooled_rep = coefficients(sum(draws[t] @ per_tag[t] for t in TAGS), sum(draws[t] @ ev_tag[t] for t in TAGS))
    ratios = {k: summarize(by_tag["bigSpot"][k] / by_tag["zeroSpot"][k], rep["bigSpot"][k] / rep["zeroSpot"][k])
              for k in ("c", "kappa_L", "beta_L", "beta_H", "pi_L", "N_H_per_event")}
    bias_c = k_bias(by_tag["bigSpot"]["c"] / by_tag["zeroSpot"]["c"], pooled["pi_L"])
    bias_r = k_bias(rep["bigSpot"]["c"] / rep["zeroSpot"]["c"], pooled_rep["pi_L"])
    return {"n_files": {t: len(per_tag[t]) for t in TAGS},
            "pooled": {k: float(v) for k, v in pooled.items()},
            "by_potential": {t: {k: float(v) for k, v in by_tag[t].items()} for t in TAGS},
            "ratio_90_over_0": ratios, "K_bias_of_pooled_c_points": summarize(bias_c, bias_r)}


def generator_dependence(ev_dcm, smash_tags, smash_ev, rng):
    """r = c_DCM / c_SMASH, its decomposition and the removed fraction; DCM event bootstrap x SMASH file bootstrap."""
    dcm, smash = coefficients(ev_dcm.sum(0), len(ev_dcm)), coefficients(
        sum(v.sum(0) for v in smash_tags.values()), sum(v.sum() for v in smash_ev.values()))
    d_draw = retro.file_draws(len(ev_dcm), rng)  # events: DCM validation has no file structure
    dcm_r = coefficients(d_draw @ ev_dcm, d_draw.sum(1).astype(float))
    s_draws = {t: retro.file_draws(len(smash_tags[t]), rng) for t in TAGS}
    smash_r = coefficients(sum(s_draws[t] @ smash_tags[t] for t in TAGS), sum(s_draws[t] @ smash_ev[t] for t in TAGS))
    out = {"dcm_events": int(len(ev_dcm)),
           "dcm": {k: float(v) for k, v in dcm.items()}, "smash": {k: float(v) for k, v in smash.items()}}
    for k in ("c", "kappa_L", "kappa_H", "beta_L", "N_H_per_event"):
        out[f"ratio_{k}"] = summarize(dcm[k] / smash[k], dcm_r[k] / smash_r[k])
    r, r_rep = dcm["c"] / smash["c"], dcm_r["c"] / smash_r["c"]
    out["removed_fraction"] = summarize(removed_fraction(r, smash["pi_L"]), removed_fraction(r_rep, smash_r["pi_L"]))
    out["threshold_r_for_half"] = float(1 / (1 + smash["pi_L"]))
    return out


def run(split_path, dcm_dir, output, make_plot=True):
    started = time.monotonic()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise ValueError(f"Output already exists: {output}")
    split_parts = parts(split_path)
    n_events = even_part_events(split_parts)
    if len(n_events) != 286:
        raise ValueError(f"Expected the 286 truth-bound even-part files, found {len(n_events)}")
    rng = np.random.default_rng(retro.BOOT_SEED)
    per_run, per_seed, generator = {}, {}, {}
    for arm in ARMS:
        for i, seed in enumerate(SEEDS):
            seed_tags, seed_ev = {t: [] for t in TAGS}, {t: [] for t in TAGS}
            for fold in range(3):
                members = sorted(c for c in n_events if split_parts[c] == (2 * fold + 2) % 6)
                vc = validation_counts(fold, seed, arm, members)
                tags = {t: np.array([vc[c] for c in members if c.startswith(t + "/")]) for t in TAGS}
                ev = {t: np.array([n_events[c] for c in members if c.startswith(t + "/")], float) for t in TAGS}
                per_run[f"{arm}/fold_{fold}/seed_{seed}"] = potential_dependence(tags, ev, rng)
                for t in TAGS:
                    seed_tags[t].append(tags[t])
                    seed_ev[t].append(ev[t])
                if arm == "A1":
                    key = f"A1/fold_{fold}/seed_{seed}"
                    generator[key] = {
                        "calibrated": generator_dependence(dcm_counts(dcm_dir, fold, seed), tags, ev, rng),
                        "raw_dcm_energy": generator_dependence(dcm_counts(dcm_dir, fold, seed, False), tags, ev, rng)}
                    for k, (lo, hi) in SEEN.items():
                        v = generator[key]["calibrated"][f"ratio_{k}"]["central"]
                        if not lo <= v <= hi:
                            raise ValueError(f"Generator ratio {k} = {v:.3f} not reproduced: {key}")
            per_seed[f"{arm}/seed_{seed}"] = potential_dependence(
                {t: np.concatenate(v) for t, v in seed_tags.items()}, {t: np.concatenate(v) for t, v in seed_ev.items()}, rng)
    partial.mkdir(parents=True)
    try:
        inputs = partial / "inputs/dcm_validation"
        for seed in SEEDS:
            (inputs / f"seed_{seed}").mkdir(parents=True)
            for name in ("predictions.csv", "complete.json"):
                shutil.copy2(Path(dcm_dir) / f"seed_{seed}/{name}", inputs / f"seed_{seed}/{name}")
        summary = {
            "schema": "rn-background-coefficients-v1",
            "protocol": "reports/rn_background_correction_protocol_20261029.md",
            "split_sha256": sha256(split_path), "band": "primary", "cut": "frozen 0.70 per run",
            "code": {"self_sha256": sha256(Path(__file__))},
            "inputs": {f"dcm_validation/seed_{s}/{n}": sha256(inputs / f"seed_{s}/{n}")
                       for s in SEEDS for n in ("predictions.csv", "complete.json")},
            "dcm_source": "/scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206/"
                          "student_sage_<seed>/{validation/predictions.csv,complete.json}; DCM-QGSM-SMM Xe+CsI "
                          "3.26 AGeV, validation run 988974",
            "scope": "validation only: SMASH, the 286 truth-bound even-part files, each scored by the fold that "
                     "validates it; DCM, the frozen A model's validation. The report-set (odd-part) files are not read. "
                     "B_L and labels are MC truth; bands, N_H and N_L are reconstructed. Per-event rates use events "
                     "with at least one candidate cluster in both samples.",
            "per_run": per_run, "per_seed": per_seed, "generator": generator,
            "settings": {"n_boot": retro.N_BOOT, "boot_seed": retro.BOOT_SEED}}
        if make_plot:
            plot(per_seed, generator, partial)
        summary["seconds"] = time.monotonic() - started
        (partial / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    except Exception:
        shutil.rmtree(partial)
        raise
    partial.rename(output)
    return summary


def plot(per_seed, generator, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from HGNDRecoGNN.scripts.plot_b_full import apply_style, mark_provenance
    apply_style()
    colors = {"A1": "#1f4e79", "B1": "#b5651d"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.84, 2.7))
    for arm in ARMS:
        b = [per_seed[f"{arm}/seed_{s}"]["K_bias_of_pooled_c_points"] for s in SEEDS]
        x = np.arange(3) + (0.1 if arm == "B1" else -0.1)
        left.errorbar(x, [v["central"] for v in b], yerr=[v["sd"] for v in b], fmt="o", ms=3, capsize=1.5,
                      color=colors[arm], label=arm)
    left.axhline(0, color=".7", lw=.5)
    left.set_xticks(range(3))
    left.set_xticklabels([f"seed {s}" for s in SEEDS])
    left.set(ylabel="K bias of a pooled c (points)", title="Potential dependence of c, SMASH validation")
    left.legend(fontsize=6)
    r = np.linspace(0.3, 1.0, 200)
    pi_l = np.mean([g["calibrated"]["smash"]["pi_L"] for g in generator.values()])
    right.plot(r, removed_fraction(r, pi_l), color=".5", lw=.8, label=f"(1−r)/(1−r(1−π_L)), π_L = {pi_l:.2f}")
    for variant, color, face in (("calibrated", colors["A1"], colors["A1"]), ("raw_dcm_energy", ".45", "none")):
        rr = [g[variant]["ratio_c"] for g in generator.values()]
        ff = [g[variant]["removed_fraction"] for g in generator.values()]
        right.errorbar([v["central"] for v in rr], [v["central"] for v in ff], xerr=[v["sd"] for v in rr],
                       yerr=[v["sd"] for v in ff], fmt="o", ms=3, mfc=face, capsize=1, color=color, ls="none",
                       label="A1 runs, " + ("SMASH calibration" if variant == "calibrated" else "raw DCM energy"))
    right.axvline(1 / (1 + pi_l), color=".6", ls=":", lw=.8, label="r = 1/(1+π_L): half removed (reference)")
    right.set(xlabel="r = c_DCM / c_SMASH", ylabel="fraction of the correction removed",
              title="Generator + energy + target swap, A1")
    right.legend(fontsize=5.5)
    for ax in (left, right):
        mark_provenance(ax, "mixed")
    fig.suptitle(f"Validation only; bars: SD of {retro.N_BOOT} resamples (SMASH files, DCM events)", fontsize=6.5)
    (output / "figures").mkdir()
    for ext in ("pdf", "png"):
        fig.savefig(output / "figures" / f"background_coefficients.{ext}", dpi=160,
                    metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, default=BASE / "split/split_manifest.json")
    p.add_argument("--dcm-validation", type=Path, required=True,
                   help="Directory with seed_<s>/{predictions.csv,complete.json} from the frozen DCM campaign")
    p.add_argument("--output", type=Path, default=Path("results/rn_background_correction_20261029"))
    a = p.parse_args()
    run(a.split, a.dcm_validation, ROOT / a.output)
    print(json.dumps({"status": "complete", "output": str(a.output)}))
