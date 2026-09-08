# PRC referee review v5 — `paper/main.tex` (commits 87a1949 → 46bf219)

## 1. One-line verdict

**Major revision — still not submittable.** Text-side v4 blockers
largely close (title retitle done, sys-budget filled,
tab:closure/tab:purity_locked numbers reproduce from CSV to 3 dp,
seed-variance quantified), but the figure suite has three hard
defects — a **stale-statistics MC-truth figure**, a **completely
empty sensitivity_ratios plot**, and an **efficiency figure showing
the wrong dataset for its cited section** — any one of which is a
desk-reject at PRC. Distance to submittable: ~1 focused week of
figure-production + a body/CSV threshold reconciliation.

## 2. Blockers

**B1. `fig:mctruth` and `fig:yield` use stale defaultSpot
statistics.** `figs/mctruth_spectra.pdf` legend:
`defaultSpot ... N_ev=76,600` alongside
`zeroSpot N_ev=640,333`, `bigSpot N_ev=619,603`.
`per_variant_summary.csv:3` and `main.tex:817` both quote
**838,284** defaultSpot events at full stats. The MC-truth curve
for defaultSpot in `mctruth_spectra.pdf` sits ~30 % below the
other two — a suppression that reproduces in
`sensitivity_yield.pdf`. Regenerate against the 838 k input.

**B2. `figs/sensitivity_ratios.pdf` is a data-less shell.**
Rendered figure has axis frame, gridlines, legend entries for four
curves (MC ratios ×2, reco ratios ×2) — but **zero curves are
drawn in the plot area**. This is the section headline plot for
`sec:closure` (`main.tex:826–835`). Either the ratios are all
outside [0, 2.5] or the plotting script silently dropped the data.
Referee opens Fig 6 first and desk-rejects.

**B3. `figs/efficiency_vs_ekin.pdf` is not what its caption
promises.** Caption at `main.tex:657–659` says "Per-dataset
reconstruction efficiency $\eff_n(\Ekin)$ at $t=0.5$ with Wilson
binomial intervals" — three datasets. The figure shows **one
dataset** (defaultSpot only, purity-locked and $t=0.5$ curves).
It is really the Sec 5.1 per-$\Ekin$ efficiency of Table 2, not
the Sec 5.2 three-sample efficiency of Eq. (2). Two options:
retitle/move to Sec 5.1 and re-cite; or regenerate with all three
SMASH samples.

**B4. Abstract cites the wrong dataset's operating point.**
`main.tex:114–115` and body `main.tex:485` both quote
"$\pi = 0.702$, $t^\star = 0.305$" as the defaultSpot working point.
`purity_locked_pi70.csv:3` (defaultSpot) reports
`locked_threshold = 0.31`. **$t^\star = 0.305$ is bigSpot's
threshold** (`purity_locked_pi70.csv:2`). Change 0.305 → 0.310.

**B5. Two placeholder figures still ship.**
`main.tex:280` (`detector_layout_placeholder`) and `main.tex:446`
(`gnn_architecture_placeholder`) — no PRC editor sends a
placeholder-figure draft to referees.

## 3. Major concerns

**M1. Fig 2 (`ereco_vs_etrue`) contradicts the text on energy
resolution.** `main.tex:539` claims "linearity within 10 % for
$0.7 \lesssim \Ekin \lesssim 5$~GeV" and `main.tex:543`
"$\sigma_E/E < 10\%$ over the same interval". Right panel of
`ereco_vs_etrue.pdf` visibly shows bias $\gtrsim +20\%$ at
$\Ekin < 1$ GeV, $\gtrsim -25\%$ at $\Ekin \sim 5$ GeV, and error
bars ($\sigma$) spanning $\pm 25$–$30\%$ across the entire quoted
interval. Either tighten the quoted interval to $[1, 3]$ GeV, or
quote the actual number.

**M2. Fig 3 (`multiplicity`) is one panel, caption promises two.**
Caption at `main.tex:587–589`: "(left) event-level $N_n$ per
event, (right) confusion matrix". Figure shows only the confusion
matrix. Add the left panel or rewrite caption.

**M3. Fig 7 (`sensitivity_threshold_scan`) still labelled "smoke,
pooled scaler".** Full-stats sensitivity has landed (job 4300982).
Title reads `... (smoke)` and y-axis `... (smoke, pooled scaler)`.
This is the v4 B2 residue in figure form: text uses full-stats
numbers, figure still says smoke. Regenerate.

**M4. Sec 6 item 4 deferral (model family) is honest but exposes
an unshipped systematic.** As phrased ("bounded above by the
aggregated systematic ... from the other items") this is an
argumentative circularity. Consider quoting even a rough
one-off HeteroGNN score as an upper bound.

**M5. `tab:purity_locked` C_c defaultSpot row disagrees with source
CSV division.** `main.tex:911` reports `defaultSpot: C_c = 1.454`.
From `purity_locked_pi70.csv:3`: `locked_N_true = 658324.6`,
`N_MC_clusters = 449069` → C_c = 658324.6 / 449069 = **1.466**.
Zero/bigSpot agree with CSV to 3 dp; only defaultSpot is 0.012
low. Recompute.

**M6. FoM look-elsewhere (v3 M2, v4 open) still not addressed.**
`main.tex:870–877` scans $t \in [0.05, 0.95]$ with no held-out
split.

## 4. Minor issues

- `main.tex:309, 322, 560` — three `\NUM{}{}` markers still red.
- `main.tex:959` — `\TODO{re-test at full SMASH statistics ...}`.
- `main.tex:1144` — inline bibliography.
- `main.tex:1254` — `\CITENEEDED{SMASH DOI update ...}`.
- `main.tex:96` — `%% TODO(authors)`.
- Sec 6 item 1 (`main.tex:948`) still references "285:1" pooled-
  scaler dominance — that was smoke-scale.

## 5. v4 items closed / still open

**Closed.**
- v4 M1 title mismatch: `main.tex:68-71`.
- v4 M4 three purity values: aligned at 0.702; see B4.
- v4 M5 job number: `main.tex:802,811` cites 4284505 + 4284506 + 4300982.
- v4 M4 sys budget: Sec 6 items 4 (technical blocker deferral,
  `main.tex:982-998`) and 5 (seed variance, `main.tex:999-1017`).
  Closed by 46bf219.
- v3 M4 seed variance reproduce exactly from `seed_variance.csv`:
  1.1173 → 1.117, 0.0068 → 0.007, 0.0034 → 0.003.

**Still open.**
- v4 B1 threshold-label drift — B4 above.
- v4 B2 figure "smoke" residue — M3 above.
- v4 B3 markers: 30 → 11.
- v4 M2 FoM held-out — M6 above.
- v4 open bib migration.
- v4 open author sign-off `%% TODO(authors)`.
- v4 M5 figures at full-stats — B1 above.

## 6. Figure audit

| Figure | Verdict | Justification |
|---|---|---|
| `performance_roc.pdf` | **Ready** | AUCs match main.tex:475–476; publication-clean. |
| `ereco_vs_etrue.pdf` | **Needs-fix** | Right panel contradicts "<10 %" resolution claim at main.tex:539,543. |
| `multiplicity.pdf` | **Needs-fix** | Only confusion matrix; caption promises two panels. |
| `efficiency_vs_ekin.pdf` | **Reject** | One dataset while caption promises three-SMASH samples. |
| `mctruth_spectra.pdf` | **Reject** | defaultSpot legend N_ev = 76,600 vs table's 838,284. |
| `sensitivity_yield.pdf` | **Reject** | Inherits B1; reco > MC while tables say C_ν < 1. |
| `sensitivity_ratios.pdf` | **Reject** | Empty plot area — legend but no curves. |
| `sensitivity_threshold_scan.pdf` | **Needs-fix** | Still labelled "smoke, pooled scaler". |

## 7. Fastest path to PRC submission — 5 ordered PRs

1. **Figure regeneration (2 days).** Re-run figure producers
   against `sensitivity_full_hpc_finish_4300982`:
   (a) fix defaultSpot $N_{ev}=838\,284$ in mctruth + yield,
   (b) debug empty `sensitivity_ratios` — probably a masking bug,
   (c) regenerate `efficiency_vs_ekin` with three SMASH samples
   per Eq. (2) or move fig to Sec 5.1 and reword,
   (d) add missing left panel to `multiplicity`,
   (e) regenerate `sensitivity_threshold_scan` at full stats.
2. **B4 numerical fix + M5 C_c recompute (½ day).** Change
   $t^\star = 0.305 \to 0.310$ in abstract and body; recompute
   defaultSpot $C_c$ in `tab:purity_locked` (should be 1.466);
   reconcile M1 energy-resolution text with Fig 2.
3. **Placeholder + marker retirement (1 day).** Produce
   detector_layout + gnn_architecture; retire 11 markers.
4. **Author sign-off + bib migration (½ day + round-trip).**
5. **FoM held-out or paragraph deletion (½ day).**

**Wall-time estimate: ~1 week + author round-trip.** All
prerequisite artefacts (full-stats pickles, seed-variance CSV,
purity-lock CSV) are present. Remaining work is figure/text
discipline, not new physics.
