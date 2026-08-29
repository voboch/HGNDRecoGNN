# PRC referee review v4 — `paper/main.tex`

> **Editorial context (2026-08-29).** Author has fixed the target at
> **Phys. Rev. C** and re-audited the abstract to a methods-first
> framing (commit 279883c), and populated Sec 5.4 with
> full-statistics numbers (commit 87a1949). This pass tests whether
> that survives at PRC and whether the residual internal-consistency
> defects still trip a desk-reject.

## 1. One-line verdict

**Major revision** (borderline reject-to-major at PRC). The v3
desk-reject dissolves: three of four v3 blockers close with
87a1949/279883c, and the $C_\nu = k \cdot C_c$ decomposition is now
backed by full-statistics tables that arithmetically survive
spot-checks. The paper is one careful editing pass plus one
physics-framing decision away from being sendable to a PRC referee.
Three new consistency defects prevent submission today.

## 2. Blockers

**B1 (new). Abstract-body numeric drift on the headline efficiency.**
Abstract (`main.tex:114`) quotes purity-locked "$\pi = 0.706$,
cluster efficiency 0.735 and per-event neutron detection efficiency
0.841". Body (`main.tex:480–482`) says "$\varepsilon_{\text{cl}} =
0.739$ and $\varepsilon_{\text{ev}} = 0.844$ (purity $0.702$,
$t^\star = 0.305$)". Three of three headline numbers disagree
between abstract and body by 4–5 in the third decimal.
`purity_locked_pi70.csv` (defaultSpot row) reports
`locked_purity_actual=0.7`, `locked_cluster_eff=0.731`,
`locked_N_reco/ev=0.56`. So *neither* the abstract nor the body
matches the source CSV. A PRC editor bounces this on first read.

**B2 (new). Sec 5.6 threshold-scan section is still smoke-scale.**
`main.tex:867–871`: "At smoke scale the locked thresholds are
$t^\star = 0.333 / 0.303 / 0.313$ … $C_c = 1.44 / 1.40 / 1.47$ and
$C_\nu = 0.967 / 0.736 / 0.980$". The full-stats
`purity_locked_pi70.csv` gives $t^\star = 0.308/0.310/0.305$,
$C_c = 1.457/1.468/1.461$, $C_\nu = 1.04/0.984/1.04$. The paragraph
then argues (`main.tex:872–877`) that "purity lock does *not*
reduce $C_\nu$ spread — max$-$min widens from 0.206 at $t=0.5$ to
0.244 at the locked point". At full stats the same numbers are
0.045 (from tab:closure) → 0.056 (from tab:purity_locked): the
"negative result" narrative is now numerically wrong (both spreads
are 4–6 %) and internally contradicts the paper's own Table 5
caption which says "$C_\nu$ spread (0.056) is compatible with the
$\sim 4.4\%$ spread in $k(\Usym)$". Either delete the paragraph or
rewrite around the full-stats numbers.

**B3 (holdover). Placeholders persist at 30 markers.**
`grep -cE '\\(TODO|NUM|CITENEEDED)\{|%% TODO' main.tex` = **30**
(down from v3's 40 — real progress but not zero). Non-cosmetic
residues: every figure is still `_placeholder` (10 figures at
`main.tex:279,442,492,546,586,626,656,715,830,894`);
`\CITENEEDED{momentum-dependent Usym transport reference}` at
`main.tex:327` for the load-bearing transport-sensitivity claim;
`\CITENEEDED{repo url + tag}` at `main.tex:1070`;
`%% TODO(authors)` at `main.tex:95` still awaiting BM@N sign-off.
**No PRC editor sends a placeholder-figure draft to referees.**

## 3. Major concerns

**M1. Methods-first abstract at PRC — survivable, but retitle.**
PRC accepts detector/methods results (see e.g. STAR EPD, PHENIX HBD
papers) *if* the physics motivation is in the abstract and the
methods enable a measurable observable. Current abstract does the
physics-motivation work in one sentence at line 126 — barely. The
**title** still promises "Sensitivity ... to the nuclear symmetry
potential", which the abstract then walks back to "provisional" and
"deferred to a follow-up paper". This title–abstract mismatch is
the single largest editor-desk risk at PRC. Retitle to something
like *"A heterogeneous graph-neural-network reconstruction of the
BM@N high-granularity neutron detector and its calibration across
three symmetry-potential simulations"*. Keep U_sym in the title,
move "sensitivity to" out.

**M2. k(U_sym) V-shape is defensible only because the paper defers
the physics claim.** Full-stats k = 0.714, 0.670, 0.712 (from
`sensitivity_summary.csv`) is non-monotonic in U_sym = 0/18/90 MeV.
Since the paper now labels the sensitivity provisional and defers
to a follow-up, this is *not* a blocker — but any physicist
referee will notice that defaultSpot's k is 6 % below the two
extremes while its N_MC/ev is 0.80 vs 1.10, a 27 % deficit.
**The V-shape is fully consistent with the 100k-event preprocessing
cap on defaultSpot suppressing high-multiplicity cluster-merger
events, not with U_sym physics.** The paper should say this
explicitly at `main.tex:783` (currently reads "uniform to 6.5%" —
technically true but hides the artefact origin).

**M3. defaultSpot N_MC/ev = 0.80 vs 1.10 artefact — publishable
with caveat, not with silence.** `tab:closure` caption mentions
the 100k cap and "to be lifted before the final table". Adequate
for a *submitted* draft, not for a published one. Since jobs
4290339+4290340 will presumably complete on the days-scale, hold
the table until they land.

**M4. Purity claim in abstract vs source CSV.**
Abstract line 114: purity 0.706. `purity_locked_pi70.csv`
defaultSpot row: `locked_purity_actual=0.7`. Body line 482:
purity 0.702. Three different values. Pick one and propagate.

**M5. tab:closure caption cites the wrong job.**
`main.tex:806` says "job 4284505". User's task description and
`tab:purity_locked` caption at line 911 both cite job 4284506.
Trivial fix but points to a broader integrity issue for reviewers
who chase job IDs into the artefacts.

## 4. Minor issues

- `main.tex:170` (Xe+CsI at 3.8 AGeV) vs `main.tex:320` (SMASH at
  2.87 AGeV) — energy mismatch still present but now
  `main.tex:325–330` gives a *reason* ("chosen to bracket the
  maximum-density working point"). Cite the actual Sorensen
  figure/section.
- `main.tex:1070` `\CITENEEDED{repo url + tag}` — PRC requires a
  code/data availability statement.
- `main.tex:404`: parameter count 12{,}479{,}173 is now hardcoded,
  closing v3 minor. Good.
- `main.tex:1099` "Inline biblist for the sketch; migrate to
  main.bib before submission" — still inline. Editorial only.
- `main.tex:846`: "at smoke scale the corresponding numbers are
  quoted in Table~\ref{tab:closure} above" — Table 3 is now
  full-stats. Delete "smoke scale".
- `main.tex:976`: Sec 6 item 2 still describes k as ranging
  "0.51–0.69" (smoke) — update to 0.670–0.714 (full).

## 5. v2/v3 items closed vs still open

**Closed by 87a1949 + 279883c.**
- v3 B1 (k non-monotonic destroys physics headline): **closed by
  deferral.** Abstract line 126–132 labels sensitivity provisional;
  k V-shape now non-fatal.
- v3 B3 (AUC 0.97 wrong): **closed.** `main.tex:472` now correctly
  quotes AUC 0.944, purity 0.813, eff 0.632.
- v3 B4 (±3 % unfold claim): **closed.** `main.tex:700` now says
  "±5%" and quotes actual deltas −3.5/−4.0/−3.6 %.
- v3 M3 (redundant tab:summary): **closed** — table deleted.
- v3 minor `LongWei2022` mis-citation: appears removed.

**Still open.**
- v3 B2 (placeholder skeleton): **partially open** — 40 → 30 markers.
- v3 M2 (FoM look-elsewhere): `main.tex:878–886` still scans
  t ∈ [0.05, 0.95] with no held-out split.
- v3 M4 (systematics-as-budget): Sec 6 items 4, 5 still one-line
  "TODO".
- v3 M5 (full-scale run): closed for Tables 3/5; still open for
  Fig 6 and threshold-scan plot.

**Newly opened by the retreat.**
- B1 abstract/body numeric drift.
- B2 Sec 5.6 smoke residue contradicting Table 5.

## 6. Reframing for PRC

**Current PRC framing survives with the retitle proposed in M1.**
No detector/companion paper split needed — Sec 2 already recaps the
detector at 0.75 pages with the correct citation to Morozov 2024.
Do *not* add an n/p physics section; that would over-promise. The
single load-bearing edit is the title. Second-order: promote the
"$C_c$ uniform to 1.4 %" claim from Sec 5.4 to its own subsection
header in Sec 5, so that the pipeline-calibration contribution
reads as a *result* and not a *cross-check* — the current word
"cross-check" in the abstract sells the contribution short.

## 7. Fastest path to PRC submission — 5 ordered PRs

1. **Consistency pass (½ day).** Fix B1 (make abstract, body, and
   `purity_locked_pi70.csv` agree on one triple:
   purity/eff_cl/eff_ev); fix B2 (rewrite `main.tex:867–890` around
   full-stats numbers or delete the "purity-lock does not
   equalise" paragraph — the full-stats Table 5 tells the opposite
   story); fix M4 (single purity value); fix M5 (job number
   4284506 in tab:closure); fix minor `main.tex:846` and
   `main.tex:976`.
2. **Retitle + subsection promotion (½ day).** Retitle per M1; add
   subsection header before line 745 "Per-cluster closure across
   three U_sym samples"; upgrade abstract line 118 verb from
   "cross-check" to "calibration".
3. **defaultSpot 100k-cap retire (gated on jobs 4290339+4290340).**
   Re-run `scripts/sensitivity.py` on the lift-cap output,
   regenerate Tables 3 and 5, confirm N_MC/ev for defaultSpot
   approaches ~1.10 and k rises above 0.68 (which would also
   flatten the V-shape). If k monotonicity emerges, promote a
   *conditional* physics-claim sentence back into Sec 5.4 (still
   deferring the full measurement).
4. **Figures (1–2 days).** Produce the 10 placeholder figures — the
   underlying pickles and CSVs are there per Sec 5.4 and the
   notebook path at `main.tex:486`. Prioritize Figs 5
   (`sensitivity_yield`), 6 (`sensitivity_ratios`), 7
   (`sensitivity_threshold_scan`) — these are what a PRC referee
   opens first.
5. **Marker retire + bib + author sign-off (½ day + author
   round-trip).** 30 → 0 markers; migrate bibliography to
   `main.bib`; fill `\CITENEEDED{momentum-dependent Usym transport
   reference}` at `main.tex:327` (Li–Chen 2008 or Baran 2005 are
   the standard cites); resolve `%% TODO(authors)` at line 95;
   add a Code and Data Availability statement (PRC requires it).

**Wall-time estimate to submittable-at-PRC state: ~1 week +
defaultSpot lift-cap turnaround + author round-trip.** The
physics-quality U_sym measurement stays deferred to the follow-up
paper on re-simulated SMASH data — that split is the right
editorial choice and the current draft executes it honestly.
