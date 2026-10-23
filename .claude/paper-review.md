# Paper-review manifest: HGND

Read by the four shared critics (`~/.claude/agents/critic-{technical,methodology,text,manager}.md`)
and the `paper-critic` skill. Project specifics live here, never in the agents.

## Project

- **Paper:** HGND neutron reconstruction with graph neural networks and the
  symmetry-potential (`S_pot`) sensitivity of neutron spectral hardness at
  BM@N.
- **Target journal:** Physical Review C (REVTeX `aps,prc`).
- **Target experiment:** BM@N at NICA, Highly Granular Neutron Detector
  (HGND); the proton arm comes from BM@N tracking.
- **Review language:** Russian, matching the existing `critical_review_v*.md`
  series.

## Paths

| role | path |
|---|---|
| code repo | `/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN` (branch `reco-calibration-and-audit`) |
| paper repo | `/Users/vovvy/Project/BM@N/HGND/HGNDPaper` (branch `main`) |
| draft | `../HGNDPaper/main.tex`; build with `make` (`make bib` runs bibtex); PDF `main.pdf` |
| manuscript corrections | `../HGNDPaper/manuscript_corrections_20261017.md` |
| writing policy | No HGND-specific policy yet. Use `/Users/vovvy/Project/HIC-NeuralFusion-paper/SCIENTIFIC_WRITING_POLICY.md` by reference, including its scope rule: no development history in the paper |
| binding conventions | `/Users/vovvy/Project/thesis-kb/conventions/` (methodology, result representation, development process, glossary; note glossary §1 on `E_kin/A = 2.87 GeV` versus `√s_NN = 2.98 GeV`) |
| local rules | `AGENTS.md` (every fixed parameter needs a plot and a written justification at the point of use), `PIPELINE.md`, `HPC.md`, `HPC_DATA.md` |
| progress and decision record | `docs/HGND_PROGRESS_GUIDE.md` (established, withdrawn, blocking, next) |
| plotting rules | `docs/PLOT_CONVENTIONS.md`, `docs/plotting_style_protocol.md` (the shared thesis-kb result-representation file supersedes both where they differ) |
| other notes | `docs/CENTRALITY_CONVENTION.md`, `docs/NOTE_beam_energy_frame_conflict.md`, `docs/NP_RATIO_PLAN.md`, `docs/PROTON_PID_*.md` |
| experiment state | latest `HANDOFF_*.md`, plus `logs/` and `reports/`. There is no single state file; the manager critic should say so |
| raw outputs | `output/`, `results/`, `reports/`, `checkpoints_hpc/` |

## Key code

- `models/`: GNN variants (`hetero_gnn.py`, `hetero_gat.py`,
  `hetero_hgt.py`, `hetero_sage.py`, `default_net.py`, `spectral.py`) and
  thesis baselines
- `analysis/`: efficiency, energy bands, metrics, sensitivity
- `scripts/`: DCM development and grouping, acceptance, systematics,
  SMASH-source audits
- `ncx/`, `proton_pid/`: reconstruction and particle identification
- Tests: `pytest tests/`

## External owners

To be confirmed with the user: co-author sign-offs, data provenance (DCM and
SMASH samples), funding.

## Known artefact signatures (methodology critic: check whether each still holds)

- **Per-dataset `StandardScaler`.** `eToF` carries the neutron energy, so a
  per-dataset fit divides out part of the effect. Rebuilding with one shared
  scaler moved the differential response from −30.7 ± 8.6 MeV (3.55σ) to
  −11.4 ± 9.8 MeV (1.16σ) on the identical checkpoint. Any comparison must
  share one fitted normalisation.
- **The polar-angle band `θ ∈ [8.9°, 13.1°)` is not the detector.** The
  front face is an offset rectangle covering 20 % of azimuth. The generator
  fixes the reaction plane, so azimuth cannot be weighted away.
- **Detection efficiency spans a factor of 39.6 in `E_kin`.** The
  conventional denominator of `R` sits where efficiency is 2 %, and it is
  not recoverable by unfolding.
- **The purity lock and the calibration use truth information.**
- **HGND-band n/p yield ratio:** null and non-monotonic at full statistics;
  not an `S_pot` probe.
- **`S_pot` resolution of about 2 MeV:** truth-level Monte-Carlo reach only,
  not an experimental resolution or an EOS-parameter uncertainty.
- **The energy annotation `√s_NN = 2.87 GeV` on figures is wrong:** the
  sample is `E_kin/A = 2.87 GeV`.

## Preflight (read by `preflight-reviewer`)

- **Launchers:** `slurm/*.hpc.sbatch`, `slurm/hpcjob.sh`; Python entry
  points in `scripts/` (DCM development and grouping, acceptance,
  systematics).
- **HPC:** `HPC.md` and `HPC_DATA.md` are directive. Smoke on the test
  partition first; never train on the login node.
- **Local rule (`AGENTS.md`):** every fixed parameter (cut, threshold,
  binning, fit range) needs a plot of the objective or feature and a written
  justification at the point of use. A launch that introduces an
  unjustified constant is a finding.
- **Data hazards to check:**
  - a shared `StandardScaler` across compared datasets (a per-dataset fit is
    a domain leak);
  - selection in the real detector acceptance, not a polar-angle band;
  - truth information in the purity lock and calibration.
- **State:** there is no single state file. Read the newest `HANDOFF_*.md`
  and `logs/`.

## Number sources (read by `number-auditor`)

- **Canonical chain.** HGND has no single canonical-results builder. The
  paper `\input`s tables from `../HGNDPaper/tables/*.tex`
  (`dcm_grouped_numbers.tex`, `dcm_grouped_points.tex`,
  `dcm_validation_20261016.tex`, `truth_scan.tex`, `archived_ladder.tex`).
  Their data and provenance are in `../HGNDPaper/data/*.json`
  (`manuscript_provenance.json`, `dcm_grouped_20261017.json`,
  `historical_validation_selector_20261017.json`).
  The absence of one builder is itself a finding under the thesis-kb
  canonical-table rule.
- **Typed numbers.** Look them up in `docs/HGND_PROGRESS_GUIDE.md`,
  `../HGNDPaper/manuscript_corrections_20261017.md`, `reports/`, `results/`
  and `output/`.
- **Figures:** `../HGNDPaper/figs/`; label fixes are in
  `../HGNDPaper/data/figure_label_corrections.json`.
- **Precision rules:** the HIC `SCIENTIFIC_WRITING_POLICY.md`, by reference.

## Reviews

- **`reviews_dir`:** `../HGNDPaper/`
- **Naming:** continue the series `critical_review_v<N>.md`; the next is v9.

---

For the template, see the end of
`/Users/vovvy/Project/HIC-NeuralFusion/.claude/paper-review.md`.
