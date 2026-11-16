# Paper-review manifest: HGND

Read by the four shared critics (`~/.claude/agents/critic-{technical,methodology,text,manager}.md`)
and the `paper-critic` skill. Project specifics live here, never in the agents.

## Project

- **Paper:** HGND neutron reconstruction with graph neural networks at BM@N.
  Neutron spectral hardness on common simulated events is the example for
  comparing reconstruction and selecting an operating point.
- **Target journal:** Nuclear Science and Techniques (`nst.cls`, `submitting`).
- **Target experiment:** BM@N at NICA, Highly Granular Neutron Detector
  (HGND). The manuscript evaluates neutron reconstruction on simulation.
- **Review language:** Russian, matching the existing `critical_review_v*.md`
  series.

## Paths

| role | path |
|---|---|
| code repo | `/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN` (branch `smash-train-20261028`) |
| paper repo | `/Users/vovvy/Project/BM@N/HGND/HGNDPaper` (branch `main`) |
| manuscript source | [`main.tex`](../../HGNDPaper/main.tex), [`nst.cls`](../../HGNDPaper/nst.cls), [`data_availability.tex`](../../HGNDPaper/data_availability.tex) and the paper repository's `tables/` and `figs/`. `make` regenerates results, figures, `main.pdf` and `output/pdf/hgnd_manuscript_revised.pdf`. Verify the PDF against the selected source snapshot. |
| latest reviewed NST render | [`hgnd_reconstruction_nst_energy_procedure.pdf`](../../HGNDPaper/output/pdf/hgnd_reconstruction_nst_energy_procedure.pdf), with scope and SHA-256 in the [v19 audit](../../HGNDPaper/critical_review_v19.md). The audit identified `main.pdf` and the default export as older v15 renders. |
| writing policy | [`AGENTS.md`](../AGENTS.md) and shared conventions §20, §25 and §30.8. |
| binding conventions | `/Users/vovvy/Project/thesis-kb/conventions/` (methodology, result representation, development process, glossary; note glossary §1 on `E_kin/A = 2.87 GeV` versus `√s_NN = 2.98 GeV`) |
| local rules | `AGENTS.md` (every fixed parameter needs a plot and a written justification at the point of use), `PIPELINE.md`, `HPC.md`, `HPC_DATA.md` |
| progress and decision record | `docs/HGND_PROGRESS_GUIDE.md` (established, withdrawn, blocking, next) |
| plotting rules | `docs/PLOT_CONVENTIONS.md`, `docs/plotting_style_protocol.md` (the shared thesis-kb result-representation file supersedes both where they differ) |
| other notes | `docs/CENTRALITY_CONVENTION.md`, `docs/NOTE_beam_energy_frame_conflict.md`, `docs/NP_RATIO_PLAN.md`, `docs/PROTON_PID_*.md` |
| experiment state | [16 November audit snapshot](../reports/project_audit_20261116/scope.json), [architecture plan](../reports/smash_architecture_20261112/plan.json), [fit accounting](../reports/project_audit_20261116/monitor/accounting.json) and the corresponding reports and receipts. |
| raw outputs | `output/`, `results/`, `reports/`, `checkpoints_hpc/` |

## Key code

- `models/`: GNN variants (`hetero_gnn.py`, `hetero_gat.py`,
  `hetero_hgt.py`, `hetero_sage.py`, `default_net.py`, `spectral.py`) and
  thesis baselines
- `analysis/`: efficiency, energy bands, metrics, sensitivity
- `scripts/`: DCM development and grouping, acceptance, systematics,
  SMASH-source audits
- `ncx/`, `proton_pid/`: reconstruction and particle identification

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
- **HPC:** `HPC.md` and `HPC_DATA.md` are directive. Training runs on compute
  nodes. The user's waiver directs the current campaign to proceed without
  test-queue jobs or extra unit tests. Source, input, receipt and production
  preflight checks apply. See the [architecture probe preflight](../reports/smash_architecture_20261112/probe_preflight.md)
  and the [detector-only preflight](../reports/detector_only_20261112/preflight.md).
- **Local rule (`AGENTS.md`):** every fixed parameter (cut, threshold,
  binning, fit range) needs a plot of the objective or feature and a written
  justification at the point of use. A launch that introduces an
  unjustified constant is a finding.
- **Data hazards to check:**
  - a shared `StandardScaler` across compared datasets (a per-dataset fit is
    a domain leak);
  - selection in the real detector acceptance, not a polar-angle band;
  - truth information in the purity lock and calibration.
- **State:** verify completion and scientific acceptance in the campaign's own
  plan, result export and acceptance receipts.

## Number sources (read by `number-auditor`)

- **Canonical chain:** the [input manifest](../../HGNDPaper/data/canonical_inputs.sha256.json)
  pins the sources. The [builder](../../HGNDPaper/scripts/build_results.py)
  generates [`canonical_results.json`](../../HGNDPaper/data/canonical_results.json),
  the number macros and the current manuscript tables. The
  [v19 reproduction ledger](../reports/project_audit_20261116/reproduction_audit.json)
  records the verified scope, including the historical v15 derived-data replay.
- **Reconstruction and threshold:** use the [split](../results/smash_train_20261028/split/split_manifest.json),
  [even-file comparison](../reports/rule_tof_benchmark_20261110/contrast.json),
  [even-file cut decision](../reports/rn_score_cut_20261110/choice.json),
  [odd-file plan](../reports/rule_tof_odd_20261110/plan.json) and
  [odd-file comparison](../reports/rule_tof_odd_20261110/results.json).
  The truth soft band is $0.3 \le E < 1$ GeV and the candidate lower edge
  is strict. The open hard band is $E \ge 2$ GeV. The common-range check
  uses $2 \le E < 4$ GeV. The odd-file check is retrospective at frozen cuts.
- **Energy treatment and detector-only replay:** see the [energy decision](../../HGNDPaper/energy_procedure_decision_20261111.md),
  [matched and odd energy results](../reports/energy_head_capped_20261111/reduction/results.json)
  and [replay acceptance](../reports/detector_only_20261112/result_4506482/README.md).
  The replay checked preparation on 286 files and CPU inference on three
  witness files, one per fold.
- **Precision:** report fixed-model file-bootstrap sample SD in percentage
  points for contrast spread and both Q16 and Q84 for paired intervals.
  Distinguish between-seed SD and SE from file-sampling variation.
- **Submission:** the [NST notes](../../HGNDPaper/nst_submission_notes.md),
  [data availability statement](../../HGNDPaper/data_availability.tex) and
  [owner checklist](../../HGNDPaper/output/submission/nst_v15_open/owner_approval_checklist.md)
  track deposit access, production records and author approvals. Use the
  checklist with the named source/PDF/archive release selected for review.
- **Historical records:** the v10 [action report](../reports/v10_actions_20261026.md)
  and [manuscript corrections](../../HGNDPaper/manuscript_corrections_v10_20261026.md)
  remain available for earlier content and truth studies.

## Reviews

- **`reviews_dir`:** `../HGNDPaper/`
- **Naming:** continue the `critical_review_v<N>.md` series from the
  [v19 audit](../../HGNDPaper/critical_review_v19.md), dated 16 November 2026.

---

For the template, see the end of
`/Users/vovvy/Project/HIC-NeuralFusion/.claude/paper-review.md`.
