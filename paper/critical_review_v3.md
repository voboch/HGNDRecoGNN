# PRC referee review v3 — `paper/main.tex`

> **Editorial note (2026-08-26):** the author explicitly chose to keep
> the target at **Phys. Rev. C (or similar Q1)**, overriding this
> review's §6 recommendation to retarget to NIM A. All fastest-path
> items are therefore executed against the PRC bar. The physics
> reframing (headline = pipeline calibration; U_sym sensitivity in
> outlook) is still adopted since it strengthens rather than weakens
> the PRC-relevant claims.

Second adversarial pass after the 2026-08-26 closure-refactor commits.
Compared against `paper/critical_review_v2.md` (desk-reject). Numbers
cross-checked against
`results/sensitivity_hpc_pooled_smoke_fixed/{e_pred,e_true}/sensitivity_summary.csv`
and `results/sensitivity_hpc_pooled_smoke_purityLocked/purity_locked_pi70.csv`.

## 1. One-line verdict

**Still desk-reject** in current form, but the technical spine is now
close to a viable **major-revision** paper — a rewrite + one
full-scale run away. The v2 headline blocker (B1) is genuinely
dissolved by the `C_ν = k·C_c` decomposition; four new blockers
replace it, three prose- and one physics-.

## 2. Blockers

**B1 (new). k(U_sym) is not U_sym-monotonic.**
`results/sensitivity_hpc_pooled_smoke_fixed/e_pred/sensitivity_summary.csv`:
k(0)=0.687, k(18)=0.511, k(90)=0.663. zero↔big differ by 90 MeV in
U_sym and by 3.5 % in k; default sits 25 % below either extreme.
`main.tex:709–713` reads this as "defaultSpot's higher effective
neutron density increases the multi-neutron cluster-merger probability
by ~40 %", which is true for the *sample* (N_MC/ev 0.699/0.715/0.686
— default is 3 % denser) but is not the U_sym dependence. The paper's
proposed final observable at `main.tex:715–720` and
`main.tex:922–930` — a k(U_sym)-conditional unfold — is undermined at
smoke scale: at these statistics k tracks per-sample neutron density
more than U_sym, and any downstream physics claim will be dominated
by the same sample-level noise the v2 review flagged for N_MC/ev
(§M5). Fix: quote k(U_sym) with a Poisson error band and a
statistical vs systematic breakdown, or reframe the target observable
as a *shape* variable (e.g. `dN_true/dE_kin` slope) that is less
contaminated by the k-multiplier.

**B2 (holdover from v2 B3+B4). Placeholder skeleton persists.**
`grep -cE '\\(TODO|NUM|CITENEEDED)\{|%% TODO' main.tex` = **40**
markers (v2 quoted >30, i.e. *increased*). Notable unresolved:
abstract still holds smoke placeholders `\NUM{80}{\%}` /
`\NUM{87}{\%}` at `main.tex:123–124`; `%% TODO(authors)` at
`main.tex:95` still awaiting BM@N sign-off; `\TODO{quote from final
SLURM sweep…}` at `main.tex:126`; every figure still `_placeholder`
(`main.tex:274,428,467,489,523,565,595,646,745,815`); Table
`tab:summary` at `main.tex:766–776` all cells `\TODO{}` while the
adjacent Table `tab:closure` (`main.tex:722–741`) is populated with
the same data — redundant. A PRC editor will not send this to review.

**B3 (holdover from v2 B5/M4). AUC and eff still wrong in Sec 5.**
`main.tex:454–460` still quotes `AUC ≈ \NUM{0.97}`, `eff ≈
\NUM{0.80}`, `purity ≈ \NUM{0.87}`. `evaluate_hpc_checkpoint.ipynb`
reports AUC 0.944, purity 0.813, eff 0.632 at t=0.5. v2 M4 explicitly
flagged this; fix not applied.

**B4 (new). ±3 % unfold-vs-direct claim is false.**
`main.tex:635–638`: "The resulting Ntrue(Etrue) agrees with the
direct Epred-binned correction to ±3 %". Actual gaps from summary
CSV (direct C_ν vs unfold C_ν, e_pred basis):
zero 0.846→0.816 (**−3.5 %**),
default 0.647→0.621 (**−4.0 %**),
big 0.853→0.822 (**−3.6 %**).
All three violate ±3 %; worst is 33 % over the quoted tolerance.
Trivial to fix (change to ±5 % or tighten Tikhonov λ), but the
number as written is refutable in one line.

## 3. Major concerns

**M1. Purity-lock does not equalise closure — the opposite.**
`results/sensitivity_hpc_pooled_smoke_purityLocked/purity_locked_pi70.csv`:
reference (t=0.5) C_ν spread max−min = 0.853−0.647 = 0.206;
purity-locked (t≈0.31) spread = 0.980−0.736 = **0.244**.
`scripts/purity_locked_closure.py:189–196` explicitly prints
"purity-lock WORSENS closure spread — investigate" for this exact
outcome, and that diagnostic never surfaces in the paper.
`main.tex:790–796` names the purity-locked benchmark but quotes no
numbers. Either move to appendix as a *negative* result, or explain
why a fixed-purity cut increases dataset dependence.

**M2. Sensitivity-optimal FoM (v2 M2) untouched.**
`main.tex:797–804`: scan t ∈ [0.05, 0.95] over 12 E_kin bins; no
held-out split, no look-elsewhere correction, no degrees-of-freedom
statement.

**M3. C_c > 1 by 25 % attribution is partially defensible but
underquantified.**
Paper (`main.tex:702–708`, `main.tex:874–880`) attributes C_c ≈ 1.25
uniformly to classifier background leak. `analysis/sensitivity.py:207–228`
builds the response matrix from signal clusters only; the unfold gap
for default (1.267 direct → 1.216 unfold) isolates ~5 pp as
background-leak component. The remaining ~22 pp is not background
leak — likely (a) per-bin division by small ε in low-stat bins
(defaultSpot ekin 0.3–0.6 in `sensitivity_defaultSpot.csv`:
n_true=15, n_pass=3, ε=0.2, n_reco=4 → n_true_solved=20, inflating
by 33 % in that bin), or (b) e_pred-binned ε using signal clusters
whose e_pred spills across bin edges. Sec 6 item 3 promises a C_c(t)
table across t ∈ {0.3, 0.5, 0.7}; no numbers appear.

**M4. Systematic budget still not a budget.** Sec 6 lists 8 items;
only item 1 (scaler drift +0.009/0.000/+0.008) has a pp-on-observable
number; items 2 and 3 quote qualitative spreads; items 4–8 have
`\TODO` or no impact.

**M5. Full-scale run still absent.** `slurm/train_valloss.hpc.sbatch`
"not yet executed on cluster". Every physics claim rests on
1600–2000-event smoke subsets. With per-dataset N_MC^ν ~ 1100–1500,
k has Poisson σ/N ~ 3 %, comparable to the observed inter-dataset
spread outside defaultSpot. Statistical significance of the k spread
is not computed anywhere.

## 4. Minor issues

- `main.tex:1087–1091`: `LongWei2022` bib entry still the K₀ paper
  (v2 minor, not fixed); citation at `main.tex:161` supports U_sym
  sensitivity, wrong reference.
- `main.tex:170` (Xe+CsI at 3.8 AGeV) vs `main.tex:315` (SMASH at
  2.87 AGeV) — energy mismatch still unexplained.
- `main.tex:391`: parameter count still `\NUM{6.4e6}` placeholder.
- `main.tex:509–516`: 10⁹/month projection vs 1600-event smoke sample
  — delta not stated.
- `main.tex:766–776`: `tab:summary` all-`\TODO{}` while `tab:closure`
  above carries the numbers — merge or delete.
- `main.tex:850`: "285:1" hit-count dominance in pooled scaler is
  strong; cite the CSV path.
- `main.tex:552–561` (Sec 5.1 MC-truth prose) still reads per-cluster
  while `main.tex:900–903` clarifies per-neutron; align.
- Bibliography still inline (`main.tex:989–1137`).
- Purity-locked cluster-eff (0.674/0.773/0.71) is not the per-Ekin
  curve 0.32/0.54/0.73/0.82/0.81 from session context — pick one
  benchmark.

## 5. v2 items closed / still open

**Closed by v3.**
- v2 §5 "§2.1 non-uniform closure": *reframed*, technically sound
  (C_c uniform at 1.25 ± 0.03; C_ν spread absorbed into k). Accept
  as closed.
- v2 §5 "§3.5 MC-truth definition": explicit at `main.tex:684–685`,
  `main.tex:900–903`. Closed.
- v2 §5 "§2.2 e_pred vs e_true binning": now the design principle
  at `main.tex:614–627` with explicit warning against the mis-binned
  variant, backed by `analysis/efficiency.py:1–23`. Closed.
- v2 §5 "§3.1 scaler contamination": quantified at
  `main.tex:840–861` (+0.009/0.000/+0.008) with
  `scripts/run_pooled_scaler_sensitivity.py` shipping. Closed.

**Still open.**
- v2 M2 sensitivity-optimal FoM without held-out split — see M2.
- v2 M3 systematics-as-bullet-list — see M4.
- v2 M4 AUC/eff gap in Sec 5 — see B3.
- v2 M6 reproducibility (`\CITENEEDED{repo url + tag}` at
  `main.tex:959`, `%% TODO` at `main.tex:981`).
- v2 §5 "§2.3 signal-in-error-bars": no significance calculation
  anywhere.
- v2 §5 "§3.3 seed/model variance": Sec 6 items 4–5 still unfilled.
- v2 §5 "§3.4 power calc": still absent.
- v2 §5 "§3.7 smoke vs full divergence": still deferred.

**Rhetorically closed but not numerically.**
- Sec 6 item 3 (`main.tex:874–880`) says C_c > 1 is a
  threshold-dependent systematic, but no C_c(t) table.
- Sec 7 (`main.tex:922–930`) headlines the `k(U_sym)`-conditional
  unfold, but B1 shows k isn't U_sym-monotonic at current statistics
  — premature.

## 6. Recommended reframing

**The C_ν = k·C_c decomposition is defensible and should stay.**
Bridge identity verified row-by-row from `sensitivity_summary.csv`:
0.687·1.230=0.845 vs reported 0.846; 0.511·1.267=0.647 vs 0.647;
0.663·1.287=0.853 vs 0.853. This is the strongest single paragraph
in the draft.

**But the physics headline must retreat further.** k is not
monotonic in U_sym at these statistics, so the paper cannot claim
sensitivity to U_sym *through* k without a full-statistics
measurement of k(U_sym) with error bars. Reframe:

- **Headline**: "The HGND + hetero-GNN pipeline achieves per-cluster
  closure C_c = 1.25 ± 0.03 uniformly across three SMASH U_sym
  samples at t=0.5. Per-neutron closure C_ν is dominated by the
  physical clusters-per-neutron ratio k, which varies 0.51–0.69
  across the three datasets — a dataset-level effect that
  constitutes the current systematic ceiling on any per-event yield
  ratio interpretation of U_sym."
- **Defer**: per-event `N_true/ev(U_sym)`. Promote *shape* variables
  (`dN_true/dE_kin` slope in [1, 3] GeV) less contaminated by k, or
  `N_true/N_MC^cl` at fixed E_kin.
- **Journal**: NIM A remains more natural than PRC for a methods-
  first paper whose physics claim is "sensitivity-limited by
  clustering, not by the ML backbone".

## 7. Fastest path to submission (5 ordered PRs)

1. **Content correctness pass.** Fix B3 (AUC 0.944 not 0.97, eff
   0.632, purity 0.813 in Sec 5; add purity-locked t≈0.31 numbers
   from `purity_locked_pi70.csv` in Sec 5.6); fix B4 (change ±3 % to
   ±5 %); fix `LongWei2022` mis-citation; delete or merge redundant
   `tab:summary`.
2. **Reframe physics headline.** Rewrite abstract and Sec 5 intro
   around C_c uniformity + k spread; move "sensitivity to U_sym" to
   Sec 7 as future work; drop `k(U_sym)-conditional unfold` from
   Sec 7 until full-statistics k has monotonic ordering.
3. **Purity-lock as negative result.** Add purity-locked closure
   table (0.98/0.736/0.967 for big/default/zero) to Sec 5.6 with
   explicit "spread does not decrease" statement — the honest
   reading of `scripts/purity_locked_closure.py:189–196`.
4. **Systematics as budget.** Fill Sec 6 items 2, 3, 4, 5, 6 with
   pp impacts. For items 4–5 (model-family, seed) without runs, mark
   "future work" in one line and delete from Sec 6 — do not leave
   `\TODO`.
5. **Full-scale run + author sign-off + bibliography.** Execute
   `slurm/train_valloss.hpc.sbatch`, re-run sensitivity + purity
   scripts on the full sample, drop into Table 3 and Fig 6. Fill
   `%% TODO(authors)` at `main.tex:95`; migrate bibliography to
   `main.bib`; close 40 → 0 `\NUM`/`\TODO` markers. Submit.

Estimated wall time: PR-1 half a day; PR-2 one day; PR-3 half a day;
PR-4 two days (k statistical error + per-Ekin C_c table are derivable
from current pickles); PR-5 gated on ~6 h SLURM run + ~1 week author
round-trip. **~2 weeks minimum** to a NIM-A-viable draft; PRC still
requires the full-statistics k(U_sym) with monotonic ordering, which
is not on the current work plan.
