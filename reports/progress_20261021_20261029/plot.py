"""Figures for the dated progress report, from its pinned snapshot only."""

from pathlib import Path
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO.parent / "HGNDPaper"
sys.path.insert(0, str(PAPER / "scripts"))
from plot_style import apply_style

apply_style()
D = json.loads((HERE / "data.json").read_text())
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN = "#24577a", "#b26928", "#427955"
SEEDS = (42, 123, 456)
COLORS = (BLUE, ORANGE, GREEN)


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            if b.width and b.height:
                assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1 + 1 and b.y1 <= fig.bbox.y1 + 1, (name, text.get_text())
    fig.savefig(OUT / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / (name + ".png"), dpi=180)
    plt.close(fig)


def header(fig, title, subtitle):
    fig.text(0.1, 0.985, title, fontsize=10, va="top")
    fig.text(0.1, 0.915, subtitle, fontsize=8.5, va="top")


fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.35))
fig.subplots_adjust(left=.10, right=.98, bottom=.23, top=.79, wspace=.4)
groups = D["dcm_census"]["groups"]
events = np.array([groups[t]["events"] for t in ["988975", "988974", "988976"]])
unique = np.array([groups[t]["unique_contents"] for t in ["988975", "988974", "988976"]])
x = np.arange(3)
axes[0].bar(x-.18, events/1000, .36, color=BLUE, label="Записей событий")
axes[0].bar(x+.18, unique/1000, .36, color="#a7becd", label="Различных содержимых")
axes[0].set_xticks(x, ["train", "validation", "evaluation"], rotation=15)
axes[0].set_ylim(0, 115)
axes[0].set_ylabel("Retained-hit события (тыс.)")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
metrics = list(D["student_leakage"]["metrics"].values())
for i, key in enumerate(["all", "seen_training_candidate", "unseen_training_candidate"]):
    a = np.array([m[key]["roc_auc"] for m in metrics])
    axes[1].scatter(i+np.linspace(-.12, .12, len(a)), a, s=13, color=COLORS[i])
axes[1].set_xticks([0, 1, 2], ["весь test", "seen", "unseen"])
axes[1].set_ylabel("ROC-AUC")
axes[1].set_ylim(.92, 1.005)
header(fig, "DCM: повторение содержимого и студенческий random split", "MC truth census; справа: scores vs MC labels, 18 моделей/seed, без CI")
save(fig, "01_dcm_census")

fig, ax = plt.subplots(figsize=(6.84, 2.7))
fig.subplots_adjust(left=.13, right=.98, bottom=.18, top=.72)
mapping = {"train": 0, "validation": 1, "test": 2}
matrix = np.zeros((3, 6))
for fold in range(3):
    matrix[fold, [(2*fold+2)%6]] = 1
    matrix[fold, [2*fold, 2*fold+1]] = 2
cell_colors=["#d7e3eb", "#f2d7ad", "#6991ab"]
for fold in range(3):
    for part in range(6):
        ax.add_patch(matplotlib.patches.Rectangle((part-.5,fold-.5),1,1,
            facecolor=cell_colors[int(matrix[fold,part])],edgecolor='none'))
        ax.text(part, fold, ["train", "validation", "test"][int(matrix[fold, part])], ha="center", va="center", fontsize=8.5)
ax.set_xlim(-.5,5.5);ax.set_ylim(2.5,-.5)
ax.set_xticks(range(6), [f"P{i}" for i in range(6)])
ax.set_yticks(range(3), [f"fold {i}" for i in range(3)])
ax.tick_params(which="both", length=0)
header(fig, "SMASH: три фолда по группам содержимого", "MC-assisted grouping; файлы каждого test не используются для обучения его модели")
save(fig, "02_crossfit")

# Preserve the already verified vector figures without editing their contents.
for stem, name in [("truth_spectra", "03_truth_spectra"), ("truth_response_v9", "04_truth_contrasts")]:
    for suffix in ["pdf", "png"]:
        if suffix == "pdf":
            expected=D['manuscript']['numerical_checks']['rebuild_output_sha256']['figs/'+stem+'.pdf']
            assert hashlib.sha256((PAPER/'figs'/(stem+'.pdf')).read_bytes()).hexdigest()==expected
        shutil.copyfile(PAPER / "figs" / (stem + "." + suffix), OUT / (name + "." + suffix))

fig, ax = plt.subplots(figsize=(6.84, 3.8))
fig.subplots_adjust(left=.11, right=.98, bottom=.18, top=.77)
retro = D["retrospective"]["result"]["endpoint"]
rungs = ["T0", "T1", "T2", "C1", "C2", "C3", "C4"]
for j, (seed, color) in enumerate(zip(SEEDS, COLORS)):
    vals = [retro[f"{r}/seed_{seed}"] for r in rungs]
    y = np.array([v["central_percent"] for v in vals]); e = np.array([v["bootstrap_sd_percent"] for v in vals])
    ax.errorbar(np.arange(7)+(j-1)*.08, y, yerr=e, marker=["o", "s", "^"][j], ms=4, lw=.7, capsize=2, color=color, label=f"seed {seed}")
    assert ((y-e) >= -1).all() and ((y+e) <= 11).all()
ax.set_ylim(-1, 11); ax.set_xlim(-.4, 6.4)
ax.set_xticks(range(7), rungs)
ax.axvspan(5.55, 6.4, color="#e7edf0", zorder=0)
ax.set_ylabel("90/0 контраст Rₙ (%)")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
header(fig, "26.10: ретроспективная лестница frozen DCM → SMASH", "MC-assisted T0–C3; C4 reconstructed | старый знаменатель E < 1 GeV; 432 файла")
save(fig, "05_retrospective")

arms = D["training"]["classification_energy"]["arms"]
fig, axes = plt.subplots(1, 3, figsize=(6.84, 3.1))
fig.subplots_adjust(left=.10, right=.98, bottom=.21, top=.76, wspace=.55)
for ax, key, label, factor in zip(axes, ["auc", "efficiency", "mae_GeV"], ["ROC-AUC ↑", "Эффективность (%) ↑", "MAE (MeV) ↓"], [1, 100, 1000]):
    all_bounds = []
    for j, (arm, color) in enumerate([("A1", BLUE), ("B1", ORANGE)]):
        v = arms[arm][key]
        values = np.array(v["per_seed"])*factor
        ax.scatter(j+np.array([-.07, 0, .07]), values, color=color, s=20, marker="o" if j == 0 else "s")
        ax.errorbar(j+.17, v["mean"]*factor, yerr=v["seed_sd"]*factor, color=color, fmt="_", ms=7, capsize=2)
        all_bounds.extend([*values, (v["mean"]-v["seed_sd"])*factor, (v["mean"]+v["seed_sd"])*factor])
    lo, hi = min(all_bounds), max(all_bounds); pad=(hi-lo)*.22
    ax.set_ylim(lo-pad, hi+pad); ax.set_xlim(-.35, 1.4)
    ax.set_xticks([0, 1], ["A1\nDCM", "B1\nSMASH"])
    ax.set_ylabel(label)
header(fig, "29.10: SMASH-обучение улучшило cluster metrics", "Reconstructed vs MC truth | 568 test-файлов; 3 seed-точки и mean ± seed SD")
save(fig, "06_training_metrics")

rn = D["training"]["rn_ladder_primary"]
pooled = rn["pooled"]["arms"]
fig, axes = plt.subplots(2, 1, figsize=(6.84, 5.2), gridspec_kw={"height_ratios": [1, 1.2]})
fig.subplots_adjust(left=.23, right=.98, bottom=.11, top=.83, hspace=.48)
ax = axes[0]
t, tsd = pooled["B1"]["T1_percent"], pooled["B1"]["T1_boot_sd"]
ax.axhspan(t-tsd, t+tsd, color=".87", label="T1: MC truth ± file SD")
ax.axhline(t, color=".35", lw=.7)
for arm, color, dx, marker in [("A1", BLUE, -.10, "o"), ("B1", ORANGE, .10, "s")]:
    v=pooled[arm]
    ax.errorbar(np.arange(3)+dx, v["C4_percent"], yerr=v["C4_boot_sd"], color=color, fmt=marker, capsize=2, ms=4, label=f"C4: {arm}")
ax.set_xticks(range(3), [str(s) for s in SEEDS]); ax.set_xlabel("training seed")
ax.set_ylabel("90/0 Rₙ (%)"); ax.set_ylim(-.5, 6.5); ax.set_xlim(-.4, 2.4)
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3, bbox_to_anchor=(-.16, 1.28))
ax=axes[1]
steps=["C1-T2", "C2-C1", "C3-C2", "C4-C3"]
for j,(seed,color) in enumerate(zip(SEEDS,COLORS)):
    y=np.arange(4)+(j-1)*.16
    a=[pooled["B1"]["steps_percent"][s][j] for s in steps]
    e=[pooled["B1"]["steps_boot_sd"][s][j] for s in steps]
    ax.errorbar(a,y,xerr=e,fmt=["o","s","^"][j],color=color,capsize=2,ms=4,label=str(seed))
    assert (np.array(a)-e>-2.8).all() and (np.array(a)+e<1).all()
ax.set_yticks(range(4), ["Native label\nC1 − T2", "Score selection\nC2 − C1", "Energy\nC3 − C2", "Background\nC4 − C3"])
ax.invert_yaxis();ax.axvline(0,color=".4",ls="--",lw=.7)
ax.set_xlim(-2.8,1);ax.set_xlabel("Изменение контраста B1 (процентные пункты)")
ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper left",bbox_to_anchor=(0,1.16))
header(fig, "29.10: физический контраст остаётся ниже MC truth", "Reconstructed C4 и MC-assisted steps; [0.3, 1) / ≥2 GeV; 400 shared file resamples")
save(fig,"07_closure")

fig, axes=plt.subplots(1,2,figsize=(6.84,3.5))
fig.subplots_adjust(left=.11,right=.98,bottom=.20,top=.77,wspace=.38)
for i,(arm,color) in enumerate([("A1",BLUE),("B1",ORANGE)]):
    v=D["training"]["delta_bias_window"]["arms"][arm]["90-0"]
    axes[0].errorbar(i+np.array([-.10,0,.10]),v["delta_bias_MeV"]["per_seed"],yerr=v["boot_sd_MeV"],fmt="o" if i==0 else "s",color=color,capsize=2,ms=4)
axes[0].set_xticks([0,1],["A1: DCM","B1: SMASH"]);axes[0].set_xlim(-.4,1.4)
axes[0].set_ylabel("Δbias (90 − 0), MeV");axes[0].set_ylim(-7,2)
axes[0].axhline(0,color=".4",ls="--",lw=.7)
for j,(seed,color) in enumerate(zip(SEEDS,COLORS)):
    a=[rn[f"fold_{k}"]["arms"]["B1"]["C4_percent"][j] for k in range(3)]
    e=[rn[f"fold_{k}"]["arms"]["B1"]["C4_boot_sd"][j] for k in range(3)]
    axes[1].errorbar(np.arange(3)+(j-1)*.10,a,yerr=e,fmt=["o","s","^"][j],color=color,ms=3,capsize=2,label=str(seed))
    assert (np.array(a)-e>-4).all() and (np.array(a)+e<7).all()
axes[1].set_xticks(range(3),["fold 0","fold 1","fold 2"])
axes[1].set_ylabel("B1: C4 контраст Rₙ (%)");axes[1].set_xlim(-.4,2.4);axes[1].set_ylim(-4,7)
axes[1].axhline(0,color=".4",ls="--",lw=.7);axes[1].legend(frameon=False,fontsize=7.5,ncol=3,loc="upper left")
header(fig,"Остаточные ограничения новой реконструкции", "Reconstructed vs MC truth | слева: Etrue ∈ [1.75, 3) GeV; бары: file-bootstrap SD")
save(fig,"08_bias_folds")

pre=D['purity_preflight']
fig,ax=plt.subplots(figsize=(6.84,3.1))
fig.subplots_adjust(left=.13,right=.98,bottom=.19,top=.77)
ax.axhspan(*pre['preregistered_expected_shift_points'],color='#e3ebf0',label='Прогноз ΔC4: 0.3–1.2 п.п.')
for key,color,marker,label in [('paired_sd_A1_points',BLUE,'o','A1'),('paired_sd_B1_points',ORANGE,'s','B1')]:
    ax.plot(pre['purity_targets'],pre[key],color=color,marker=marker,ms=4,label=label)
ax.set_xlabel('Count-purity target на validation')
ax.set_ylabel('Парная SD изменения C4 (п.п.)')
ax.set_xticks([.65,.70,.75,.80,.85,.90,.95]);ax.set_xlim(.63,.97);ax.set_ylim(-.15,9.8)
ax.legend(frameon=False,fontsize=8,loc='upper left')
header(fig,'29.10: текущий purity-scan получил NO-GO','MC-assisted review на selection S; показаны ошибки, не измеренные ΔC4')
save(fig,'09_purity_preflight')

manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*")) if p.suffix in [".pdf",".png"]}
(HERE/"figure_hashes.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("Exported",len(manifest)//2,"figures with vector PDFs and PNG review copies.")
