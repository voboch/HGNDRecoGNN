# HGND + GNN sensitivity paper — draft skeleton

Target journal: **Phys. Rev. C** (backup: NIM A, EPJ C, JHEP).
Style: `revtex4-2` (aps), twocolumn, superscript numeric refs.

## Files

| File | Purpose |
|------|---------|
| `main.tex`   | Paper draft — self-contained, compiles with placeholder figures once you populate `figs/`. |
| `figs/`      | Drop-in target for final PDFs. Filename convention below. |
| `main.bib`   | (to be extracted from the inline `thebibliography` when >20 entries) |
| `Makefile`   | `make` → `pdflatex + bibtex + pdflatex + pdflatex`. |

## Placeholder taxonomy — how to close the paper

Every unresolved item is marked with a coloured command in `main.tex`.
Search for them in this order before submitting:

| Marker | Meaning | Grep |
|--------|---------|------|
| `\TODO{...}`         | Missing prose or number that requires downstream input. | `grep -n '\\TODO' main.tex` |
| `\NUM{val}{unit}{tag}` | Placeholder number sourced from the Dombay slides or smoke run; needs to be re-measured on the final SLURM sweep. | `grep -n '\\NUM' main.tex` |
| `\CITENEEDED{tag}`   | Missing bib entry. | `grep -n '\\CITENEEDED' main.tex` |
| `%% TODO(<tag>)`     | Inline reviewer note about a whole section or figure. | `grep -n 'TODO(' main.tex` |

## Figure filename ↔ notebook output

The paper text references placeholder file names in `figs/`. The
matching notebook outputs live under
`../notebooks/results/sensitivity_full_<model>_<jobid>/`:

| Paper `figs/<name>.pdf` | Source | Notebook / script |
|---|---|---|
| `detector_layout_placeholder` | Redraw or reuse Fig. 2 of arXiv:2412.00455 | manual |
| `gnn_architecture_placeholder` | Manual tikz / PPT | manual |
| `performance_roc_placeholder`  | `performance_defaultSpot.png` | `sensitivity_full_analysis.ipynb` §2c |
| `ereco_vs_etrue_placeholder`   | `performance_defaultSpot.png` (panel d) | same |
| `multiplicity_placeholder`     | To reproduce: cell 34 template from `NeutronRecoGNN/results_smash.ipynb` + confusion matrix from Dombay slide 15 | manual (new notebook cell) |
| `mctruth_spectra_placeholder`  | `dataset_check_grid.png` (col 1) or `mctruth_spectra.png` | `sensitivity_full_analysis.ipynb` §2b |
| `efficiency_vs_ekin_placeholder` | `efficiency_vs_ekin.png` | `sensitivity_full_analysis.ipynb` §3 |
| `sensitivity_yield_placeholder` | `sensitivity_yield_analysis.png` | §4 |
| `sensitivity_ratios_placeholder` | `sensitivity_ratios.png` | §5 |
| `sensitivity_threshold_scan_placeholder` | `sensitivity_threshold_scan.png` | §6 |

Convention: export as PDF (vector) at 150 dpi minimum. When you drop
a real figure in, remove the `_placeholder` suffix from the file name
and the corresponding `\TODO(fig:...)` note in `main.tex`.

## Filling order (recommended)

1. **Fix the reconstruction pipeline blockers** flagged in
   `../notebooks/results/critical_review.md`
   (closure bias, binning mismatch, MC-truth definition drift). These
   determine whether Sec. 5 is a positive result, a null result, or a
   methods-only story.
2. **Run the full SLURM sweep**: `sbatch --array=0-2 slurm/sensitivity_full.sbatch`.
   Wait for artefacts under `notebooks/results/sensitivity_full_.../`.
3. **Regenerate figures** by re-executing
   `notebooks/sensitivity_full_analysis.ipynb` with
   `HGND_RESULTS_DIR=<slurm_output_dir>`. Copy the produced PNGs to
   `paper/figs/` (or symlink for a live pipeline).
4. **Replace `\NUM{...}` and `\TODO{...}` in order they appear**:
   abstract → intro → performance → sensitivity → summary.
5. **Populate `main.bib`** from the inline bibliography; add the
   missing entries flagged with `\CITENEEDED{}`.
6. **Author list confirmation** and acknowledgements finalisation.
7. **Compile with `make`**, then `make clean && make` once more before
   submitting the tarball.

## Compile locally

```bash
cd paper
make            # pdflatex + bibtex + 2× pdflatex
make clean      # drop *.aux/*.log/*.bbl/*.blg
```

Requires a working `pdflatex` + `bibtex`. On macOS,
`brew install --cask mactex-no-gui` (~800 MB) or the smaller
`brew install --cask basictex` (~100 MB, then
`sudo tlmgr install revtex physics siunitx booktabs hyperref
xcolor pdftex-def`) works.

## Source references

- `../docs/2412.00455v1.pdf` — Morozov et al., HGND detector paper.
  Sections 1, 2, 3 provide the detector/simulation context reused
  in Secs. 2–3 of the draft.
- `../docs/VB_BMN_Dombay.pdf` — V. Bocharnikov, 26 Feb 2026 BM@N
  A&D meeting.  Slides 8–17 supply the current ML performance
  numbers; slides 21–24 are the qualitative sensitivity precursor
  to Sec. 5.
- `../../NeutronRecoGNN/notebooks/results_smash.ipynb` — plot
  templates (ROC, purity/eff, E_reco vs E_true, multiplicity, eToF).
- `../notebooks/results/critical_review.md` — pre-submission
  self-review; every open item there must be closed or explicitly
  deferred before submission.
