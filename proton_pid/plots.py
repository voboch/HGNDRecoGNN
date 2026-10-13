"""Diagnostic plots for calibration support and the slide's two PID cuts."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from plot_b_full import apply_style
from proton_pid.core import bethe_bloch


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(Path(out)/(stem+".png"), dpi=200)
    fig.savefig(Path(out)/(stem+".pdf"))
    plt.close(fig)


def label(ax, text, synthetic=False, x=.02, y=.98, ha="left"):
    status = "Synthetic PID test; not production" if synthetic else "BM@N simulation; work in progress"
    ax.text(x, y, status+"\n"
            + text + "\nno centrality selection", transform=ax.transAxes,
            ha=ha, va="top", fontsize=7)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.grid(False)


def plot_calibration(cal, out):
    apply_style()
    fig, axs = plt.subplots(2, 2, figsize=(6.84, 5.3))
    for name, color, marker in (("proton", "#2b2b2b", "o"), ("pion", "#1f77b4", "s")):
        v = cal["species"][name]
        rows = pd.DataFrame(v["slices"])
        x = rows.p.to_numpy()
        fine = np.linspace(x.min(), x.max(), 200)
        axs[0,0].errorbar(x, rows.dedx_mean, yerr=rows.dedx_mean_se,
                         fmt=marker, ms=3, color=color, label=name)
        axs[0,0].plot(fine, bethe_bloch(fine, v["mass_GeV"], v["bethe_parameters"]), color=color)
        axs[0,1].errorbar(x, rows.m2_mean, yerr=rows.sigma_m2/np.sqrt(rows.n),
                         fmt=marker, ms=3, color=color, label=name)
        axs[0,1].axhline(v["mass_GeV"]**2, color=color, ls="--", lw=.6)
        axs[1,0].errorbar(x, rows.sigma_dedx_relative,
                         yerr=rows.sigma_dedx_relative/np.sqrt(2*(rows.n-1)),
                         fmt=marker, ms=3, color=color, label=name)
        axs[1,1].errorbar(x, rows.sigma_m2,
                         yerr=rows.sigma_m2/np.sqrt(2*(rows.n-1)),
                         fmt=marker, ms=3, color=color, label=name)
    axs[0,0].set_ylabel(r"mean $dE/dx$ [a.u.]")
    axs[0,1].set_ylabel(r"mean $m^2$ [GeV$^2/c^4$]")
    axs[1,0].set_ylabel(r"relative $dE/dx$ width")
    axs[1,1].set_ylabel(r"$\sigma(m^2)$ [GeV$^2/c^4$]")
    for ax in axs.flat:
        ax.set_xlabel(r"rigidity $p/|q|$ [GeV/$c$]")
        ax.legend(loc="best", fontsize=7)
    # The gap between pion and proton m^2 bands leaves space for provenance.
    label(axs[0,1], "MC-labelled development templates\nTOF-"+cal["tof"],
          "synthetic" in cal["sample_description"].lower(), y=.72)
    save(fig, out, "pid_calibration")
    pd.concat([pd.DataFrame(v["slices"]).assign(species=name)
               for name,v in cal["species"].items()]).to_csv(Path(out)/"pid_calibration_slices.csv", index=False)


def wilson(k, n):
    p = np.divide(k, n, out=np.full_like(np.asarray(k,float), np.nan), where=np.asarray(n)>0)
    den = 1 + 1/np.maximum(n,1)
    centre = (p+1/(2*np.maximum(n,1)))/den
    half = np.sqrt(p*(1-p)/np.maximum(n,1)+1/(4*np.maximum(n,1)**2))/den
    return centre-half, centre+half


def plot_selection(d, cal, report, out):
    apply_style()
    tof = cal["tof"]
    valid = d[d.pid_valid & d.charge_sign.eq(1)]
    sel = d[d.selected_proton]
    if valid.empty:
        raise ValueError("No valid PID tracks; cannot produce a selection QA")
    fig, axes = plt.subplots(1,2,figsize=(6.84,2.8))
    for ax, y, ylab in ((axes[0],"dedx",r"$dE/dx$ [a.u.]"),
                        (axes[1],"m2_"+tof,r"$m^2$ [GeV$^2/c^4$]")):
        ax.hist2d(valid.p, valid[y], bins=60, norm=matplotlib.colors.LogNorm(), cmap="viridis")
        # 60 display bins describe the detector bands, not a selection. All
        # track bounds come from the plotted data, so tails remain visible.
        x = np.linspace(valid.p.min(),valid.p.max(),200)
        for name,color in (("proton","#2b2b2b"),("pion","#B22222")):
            v=cal["species"][name]
            if y=="dedx":
                ax.plot(x,bethe_bloch(x,v["mass_GeV"],v["bethe_parameters"]),color=color,lw=.7)
            else:
                ax.axhline(v["mass_GeV"]**2,color=color,lw=.7,ls="--")
        ax.set_xlabel(r"rigidity $p/|q|$ [GeV/$c$]")
        ax.set_ylabel(ylab)
    synthetic="synthetic" in report["calibration_sample"].lower()
    extra="test fixture" if synthetic else "Ekin/A=2.87 GeV"
    label(axes[0],"reconstructed tracks; TOF-"+tof+"\ncounts; uncorrected; "+extra,
          synthetic,x=.98,ha="right")
    save(fig,out,"pid_reconstructed_bands")
    if not d.mc_pdg.ne(0).any():
        return
    # QA only: MC labels measure purity and efficiency. They do not change
    # selected_proton. This scan shows the cut tradeoff required by AGENTS.md.
    eligible = d.pid_valid & d.charge_sign.eq(1)
    if report["cuts"]["from_vertex"]:
        eligible &= d.from_vertex.eq(1)
    if report["cuts"]["max_chi2_ndf"] is not None:
        eligible &= d.chi2_ndf.ge(0) & d.chi2_ndf.le(report["cuts"]["max_chi2_ndf"])
    if report["cuts"]["max_dca_cm"] is not None:
        eligible &= d.dca_cm.ge(0) & d.dca_cm.le(report["cuts"]["max_dca_cm"])
    truth = d.mc_pdg.eq(2212)
    rows=[]
    for radius in np.arange(1.,4.01,.25):
        for veto in np.arange(1.,5.01,.5):
            s=eligible & d.r_proton.lt(radius) & d.r_pion.gt(veto)
            n=int(s.sum()); k=int((s & truth).sum()); total=int((eligible & truth).sum())
            lo,hi=wilson(np.array([k]),np.array([n]))
            rows.append({"proton_radius":radius,"pion_veto":veto,"selected":n,
                         "purity":k/n if n else np.nan,"efficiency":k/total if total else np.nan,
                         "purity_lo":lo[0],"purity_hi":hi[0]})
    scan=pd.DataFrame(rows)
    scan.to_csv(Path(out)/"pid_cut_scan.csv",index=False)
    fig,axs=plt.subplots(1,2,figsize=(6.84,2.9))
    for ax,value in zip(axs,("purity","efficiency")):
        table=scan.pivot(index="pion_veto",columns="proton_radius",values=value)
        im=ax.imshow(table.to_numpy(),origin="lower",aspect="auto",
                     extent=[.875,4.125,.75,5.25],vmin=0,vmax=1,cmap="viridis")
        # Extents are half-bin outer edges of the displayed scan grid.
        ax.plot(report["cuts"]["proton_radius_less_than"],report["cuts"]["pion_radius_greater_than"],
                "x",color="#B22222",ms=7)
        ax.set_xlabel(r"proton radius cut $r_p$")
        ax.set_ylabel(r"pion veto distance $r_\pi$")
        ax.set_title("MC QA: "+value+" given reconstructed track",fontsize=8)
        fig.colorbar(im,ax=ax)
    fig.suptitle("Synthetic PID test" if synthetic else "BM@N simulation: PID cut scan; no centrality selection",fontsize=8)
    save(fig,out,"pid_cut_tradeoff")
