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

## Reviews

- **`reviews_dir`:** `../HGNDPaper/`
- **Naming:** continue the series `critical_review_v<N>.md`; the next is v9.

---

For the template, see the end of
`/Users/vovvy/Project/HIC-NeuralFusion/.claude/paper-review.md`.
