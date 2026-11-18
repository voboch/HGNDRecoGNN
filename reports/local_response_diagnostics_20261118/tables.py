"""Generate readable tables from the single diagnostic JSON."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def fmt(value, digits=2):
    return "undefined" if value is None else f"{value:.{digits}f}"


def stat(value):
    sd = value["file_sd"]
    # Retain two significant digits when ordinary rounding would hide a nonzero SD.
    digits = (
        max(2, 1 - int(math.floor(math.log10(sd))))
        if sd is not None and 0 < sd < 0.005
        else 2
    )
    return f"{fmt(value['value'], digits)} ± {fmt(sd, digits)}"


def main():
    r = json.loads((OUT / "results.json").read_bytes())
    pooled = r["scopes"]["pooled"]
    lines = [
        "# Tables from the frozen local diagnostic JSON",
        "",
        "SMASH Xe+Cs, 2.87 GeV/nucleon. 286 previously inspected even files, 1,180,595 processed events; raw energies, "
        "70% native-label validation-purity target; 0.3 < T(reco) < 4 GeV. Mean of three seed estimates. "
        "± denotes sample SD of 400 paired whole-file bootstrap draws at fixed models. Seed SD is reported separately.",
        "",
        "## Geometric MC-neutron multiplicity",
        "",
        "MC includes all true energies. Miss/split denominators are geometric MC neutrons; merge denominator is selected candidates. "
        "Direct associations do not resolve descendants. No physical efficiency or background fraction is inferred.",
        "",
        "| MC multiplicity | Metric (%) | Rules + ToF | DynamicEdgeConv | Paired GNN − rules (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for state, row in pooled["multiplicity"].items():
        for metric in ("miss_percent", "split_percent", "merge_percent"):
            v = row["rates"][metric]
            lines.append(
                f"| {state} | {metric.removesuffix('_percent')} | {stat(v['rules'])} | {stat(v['dynamic_edgeconv'])} | {stat(v['gnn_minus_rules'])} |"
            )
    lines += [
        "",
        "## Seed stability on the common 58,199 primary keys",
        "",
        "Both methods select each key for every seed. True energy is in [0.3,4) GeV. "
        "Seed variation is descriptive for three fixed trained pipelines; it is not a measured training-confidence interval.",
        "",
        "| Energy metric | Rules + ToF, mean ± file SD | GNN, mean ± file SD | GNN seed centers 42 / 123 / 456 | GNN seed SD |",
        "|---|---:|---:|---|---:|",
    ]
    for metric, row in pooled["energy"]["same_across_seeds"]["metrics"].items():
        g = row["dynamic_edgeconv"]
        centers = " / ".join(fmt(x["value"]) for x in g["by_seed"].values())
        lines.append(
            f"| {metric} | {stat(row['rules'])} | {stat(g)} | {centers} | {fmt(g['seed_center_sd'])} |"
        )
    lines += [
        "",
        "## Energy by true transverse momentum",
        "",
        "Common isolated neutrons selected by both procedures within each seed. Counts are per seed; "
        "MC y and pT define all bins. A shared sample across methods does not remove common-selection bias.",
        "",
        "| True pT (GeV/c) | N by seed 42/123/456 | Metric | Rules + ToF | GNN | Paired GNN − rules |",
        "|---|---|---|---:|---:|---:|",
    ]
    for pt, row in enumerate(pooled["energy"]["by_pT"]):
        for metric, v in row["metrics"].items():
            lines.append(
                f"| {'<0.5' if pt == 0 else '>=0.5'} | {row['n_by_seed']} | {metric} | {stat(v['rules'])} | {stat(v['dynamic_edgeconv'])} | {stat(v['gnn_minus_rules'])} |"
            )
    lines += [
        "",
        "## Accepted odd-file Rn results, now separated by seed",
        "",
        "282 previously inspected odd files; frozen 70% benchmark and 75% example policies. "
        "Open hard band T >= 2 GeV. C90/0 = 100[Rn(90)/Rn(0)−1]. "
        "Contrast is in percent; SD and paired MC shift are in percentage points. All other contrasts/windows remain in JSON.",
        "",
        "| Purity target | Seed | Reconstructed contrast ± file SD | Paired shift from geometric MC ± file SD |",
        "|---|---|---:|---:|",
    ]
    for target in ("raw70", "raw75_frozen"):
        v = r["rn_seed_stability"]["cases"][target + "/open_upper"]["90/0"]
        for seed in r["seeds"]:
            lines.append(
                f"| {'70%' if target == 'raw70' else '75% frozen'} | {seed} | {stat(v['gnn']['by_seed'][seed])} | {stat(v['gnn_minus_mc_pp']['by_seed'][seed])} |"
            )
    lines += [
        "",
        "## All joint true-y/pT energy cells",
        "",
        "All registered bins retained. Mean MAE/bias in MeV; paired MAE is GNN minus rules. "
        "Central relative width is in percent and requires at least 25 pairs in every seed. "
        "Undefined means are not replaced by zero. Any undefined required draw suppresses its interval. "
        "The populated y=[1.0,1.1), pT>=0.5 cell contains only 14–15 pairs per seed.",
        "",
        "| MC y | MC pT (GeV/c) | N by seed | Paired MAE ± SD (MeV) | Rules bias (MeV) | GNN bias (MeV) | Rules width (%) | GNN width (%) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(pooled["energy"]["by_y_pT"]):
        for pt, cell in enumerate(row):
            m = cell["metrics"]
            lines.append(
                f"| [{r['y_edges'][i]:g},{r['y_edges'][i + 1]:g}) | {'<0.5' if pt == 0 else '>=0.5'} | {cell['n_by_seed']} | "
                f"{stat(m['mae_MeV']['gnn_minus_rules'])} | {fmt(m['bias_MeV']['rules']['value'])} | {fmt(m['bias_MeV']['dynamic_edgeconv']['value'])} | "
                f"{fmt(m['central_width_percent']['rules']['value'])} | {fmt(m['central_width_percent']['dynamic_edgeconv']['value'])} |"
            )
    digest = hashlib.sha256((OUT / "results.json").read_bytes()).hexdigest()
    lines += ["", f"Source results.json SHA256 `{digest}`.", ""]
    (OUT / "tables.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
