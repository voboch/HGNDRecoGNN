# PRC referee review v2 — `paper/main.tex`

Adversarial pass of `paper/main.tex` (1000 lines) after the 2026-08-26
HPC checkpoint was retrieved and evaluated on the defaultSpot test
split + three-dataset smoke sensitivity comparison
(`results/sensitivity_hpc_smoke/sensitivity/sensitivity_summary.csv`).
Written toward Phys. Rev. C acceptance; the earlier pass lives at
`notebooks/results/critical_review.md`.

## 1. One-line verdict

**Desk-reject** in current form. The headline physics claim is
contradicted by
`results/sensitivity_hpc_smoke/sensitivity/sensitivity_summary.csv`,
and the manuscript is a placeholder skeleton (>30 unresolved
`\TODO|\NUM|\CITENEEDED` markers).

## 2. Blockers

**B1. Sensitivity signal is sign-inverted vs MC truth and closure is
non-uniform.** From `sensitivity_summary.csv` (HPC checkpoint, t=0.5):

| dataset | U_sym | N_reco/ev | N_true/ev | N_MC/ev | closure |
|---|---|---|---|---|---|
| zeroSpot    | 0  | 0.355 | 0.566 | 0.699 | 0.810 |
| defaultSpot | 18 | 0.296 | 0.448 | 0.715 | 0.627 |
| bigSpot     | 90 | 0.345 | 0.551 | 0.686 | 0.803 |

Reco ratios: zero/def = 1.26, big/def = 1.23. MC-truth ratios: 0.98
and 0.96. `Ntrue(0)/Ntrue(18)−1` and `Nmc(0)/Nmc(18)−1` have opposite
signs. Both reco ratios are dominated by the 37 % closure deficit on
the trained dataset, not by physics. `main.tex:113–121` (abstract) and
`main.tex:629–659` (Sec. 5.4) currently promise the opposite
conclusion. Smallest fix: reframe methods-first (see §6) and quote
closure as a systematic before quoting a ratio.

**B2. `defaultSpot` closure swung from +52 % (old
`critical_review.md:31–42`) to −37 % here** with no explanation.
Referees will read this as pipeline instability. `main.tex:783`
defers a headline number to the SLURM sweep, but pre-sweep closure is
already the story. Fix: closure-vs-training-statistics plot, or
retire old smoke numbers explicitly.

**B3. Placeholder skeleton.** Abstract `main.tex:100–128`, Sec. 5
summary table `main.tex:661–678` (all-TODO), Sec. 6 systematics
`main.tex:743–772` (every impact TODO), every figure
`main.tex:274,428,466,488,522,564,594,617,649,715`. A PRC editor will
not send this to review.

**B4. Author list `main.tex:74–96` still has `%% TODO(authors)`.**
Submitting without BM@N sign-off is a hard reject.

**B5. Contradiction with own supporting artefact.**
`main.tex:456–461` quotes AUC ≈ 0.97, eff ≈ 0.80, purity ≈ 0.87
(Dombay). `evaluate_hpc_checkpoint.ipynb` reports AUC 0.944, cluster
purity 0.813, cluster eff 0.632 at t=0.5. 3 pp AUC gap, 17 pp
efficiency gap. Fix: one authoritative table, consistent use.

## 3. Major concerns

**M1. Sensitivity claim not defensible with current closure spread.**
Closure differential 0.63 vs 0.80–0.81 = 20+ pp; MC-truth per-event
range is only 4 % (0.686–0.715). Systematic ceiling exceeds signal by
5×. Sec. 6 item 2 (`main.tex:749–751`) correctly names this; abstract
still headlines U_sym sensitivity. Reframe.

**M2. Sec. 5.6 operating-point discussion under-justified.**
`main.tex:680–711` fixes purity-locked π = 0.7 without citing the
value from `evaluate_hpc_checkpoint.ipynb` (t≈0.31, purity 0.706),
and picks a sensitivity-optimal FoM `|N(0)−N(90)|/√N(18)` optimised
over ~19 thresholds × 12 E_kin bins on the same data as the ratio —
no look-elsewhere correction. Fix: quote Morozov-baseline threshold,
use held-out split for FoM optimisation.

**M3. Sec. 6 systematics is a bullet list, not a budget.** Seven
sources named, zero quantified impacts. Items 3 (model-family) and 4
(seed) are impossible to fill — one model, one seed
(see `critical_review.md:180–194`). At minimum quote
`max|closure−1| = 0.37` from the current CSV.

**M4. Paper–Dombay AUC gap (0.97 → 0.944) is not the important gap;
the eff gap is.** Cluster eff dropped from 0.80 (Dombay) to 0.632
(HPC ckpt at t=0.5) — 21 % loss, likely from purity/eff
operating-point conflation. State in Sec. 5 that t=0.5 is a
classifier convention, quote purity-locked (t≈0.31) numbers from
`evaluate_hpc_checkpoint.ipynb` as physics-relevant efficiency.

**M5. MC-truth ordering is 4 % non-monotonic in U_sym.**
N_MC/ev = 0.699/0.715/0.686 for 0/18/90 MeV. Even ignoring
reconstruction, MC truth is non-monotonic at ~1600-event scale. Real
MC ratio is 2–4 %; `critical_review.md:198–203` power estimate of
O(50k)/dataset for 3σ on a 20 % ratio pushes ~O(10⁶)/dataset for the
real signal. Quote power calculation.

**M6. Reproducibility claim unsupported.** `main.tex:817–819` has
`\CITENEEDED{repo url + tag}`; `meta.json` shard manifests
unpopulated. Appendix promises identical local/cluster paths;
`critical_review.md:218–225` documents divergences.

## 4. Minor issues

- Abstract `main.tex:101–128`: rewrite; `\NUM{80}{\%}`/`\NUM{87}{\%}`
  at line 124 are smoke numbers, not the HPC checkpoint.
- `main.tex:161` cites `LongWei2022`; bib entry `main.tex:947–951` is
  a K₀ paper, not U_sym. Recite or reframe.
- `main.tex:642–646`: define "closure" formally
  (`= ∫Ntrue dE / ∫Nmc dE over [0.3,5] GeV`).
- `main.tex:170` says 2023 Xe+CsI at 3.8 AGeV; SMASH sample at
  `main.tex:315` is 2.87 AGeV. Explain energy mismatch.
- `main.tex:239–256`: "2×968" vs 121×8=968/arm — state symmetry
  usage.
- `main.tex:298`: 300 000 events is `\NUM` placeholder; verify from
  bmnroot manifest.
- `main.tex:391`: measure and quote parameter count.
- `main.tex:461`: purity ≈ 0.87 inconsistent with 0.813 (t=0.5) /
  0.706 (purity-locked). Pick one with threshold.
- `main.tex:509–516`: 10⁹/month projection vs 2k-event analysis
  sample — state the delta.
- `main.tex:661–678`: fill Table 1 from `sensitivity_summary.csv`.
- Bibliography `main.tex:851–997` is inline; migrate to `main.bib`
  (already flagged in `paper/README.md:11–12`).

## 5. Retire from `critical_review.md`

- §2.4 (all-zero smoke at t=0.5) — closed. New CSV has non-zero
  N_reco everywhere.
- §2.5 (schema drift) — likely closed for HPC ckpt (v2 preprocessed).
  Confirm in appendix, remove `allow_stale_schema=True` from
  `scripts/evaluate.py:70`.
- §3.2 (--split test on OOD) — partially closed. New smoke uses
  matched 4-shard subsets (~1600–2000 events).
- §3.6 (protons) — closed; `main.tex:794–796` correctly defers.

Still open: §2.1 non-uniform closure (worse: 0.63/0.80/0.81); §2.2
e_pred vs e_true binning (see inflation in `sensitivity_defaultSpot.csv`
row 2: `n_true=27, n_pass=3, n_reco=4 → n_true_solved=36±27`); §2.3
signal-in-error-bars, now sign-inverted; §3.1 scaler contamination
(acknowledged `main.tex:743–748`, unquantified); §3.3 seed/model
variance; §3.4 power calc; §3.5 MC-truth definition
(`parquet(per-neutron)` now consistent, but paper does not state);
§3.7 smoke vs full divergence.

## 6. Recommended reframing

Current framing **not defensible**. Methods-first paper:

- **Title:** "A heterogeneous graph neural network for cluster-level
  neutron reconstruction in the BM@N HGND".
- **Abstract:** hetero-GNN as evolution of the two-network baseline
  of Morozov et al.; quote AUC 0.944, purity-locked (π=0.7) cluster
  eff 0.735, per-event eff 0.841, per-E_kin eff
  0.32/0.54/0.73/0.82/0.81 across
  [0.3,0.5)/[0.5,1)/[1,2)/[2,3)/[3,5) GeV; quote energy
  linearity/resolution. Position U_sym study as
  *feasibility/systematics probe*: closure 0.63/0.80/0.81 is the
  current-generation systematic ceiling, quote events/dataset needed
  for 3σ on a 20 % ratio, explicitly defer physics-quality U_sym
  measurement until closure uniform ≤5 % and OOD scaler shift
  controlled.
- **Section swap:** Move current Sec. 5 to shortened "Sensitivity
  outlook and closure benchmark" after systematics; expand Sec. 4
  with purity-locked operating point, per-E_kin efficiency curve,
  rule-based-baseline comparison. NIM A is a more natural home for
  this reframing than PRC.
- **Retain:** detector recap, simulation samples, GNN architecture,
  rule-based baseline subsection, sensitivity Eqs. 3–5 as
  future-analysis machinery.

## 7. Fastest path to submission (5 ordered PRs)

1. Reframe methods-first: rewrite abstract, retitle, downgrade Sec. 5
   to "outlook", headline closure spread as systematic-ceiling
   result.
2. Fill Sec. 4 from `evaluate_hpc_checkpoint.ipynb`: AUC 0.944,
   purity-locked t≈0.31 numbers, per-E_kin eff curve. Close
   `\NUM`/`\TODO` in `main.tex:456–531`.
3. Quantify Sec. 6: table with closure spread (0.37) plus a
   shared-scaler re-eval impact (one-day rerun); drop
   model-family/seed items until runs exist.
4. Fix `LongWei2022` mis-citation (`main.tex:161`,`947`),
   abstract-vs-Sec. 4 AUC inconsistency, closure definition in
   Sec. 5.4.
5. Author sign-off, populate `main.bib`, close remaining >30
   `\TODO`/`\NUM`/`\CITENEEDED` markers, compile, submit.
