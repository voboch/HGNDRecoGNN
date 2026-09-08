# PRC referee review v6 — `paper/main.tex` (through c2c617d + 46bf219)

## 1. Verdict

**Minor revision, borderline.** c2c617d clears every one of v5's five
figure/table blockers and closes the three cosmetic M-defects that
made the figure suite embarrassing. The paper is now internally
consistent between the abstract-quoted purity-lock working point,
`tab:purity_locked`, and the CSV source, and the sensitivity figure
suite is publishable. What remains is (a) a body/figure
inconsistency in Sec 5.2 that survived the abstract retreat (M1),
(b) three cosmetic red `\NUM{}` markers and one
`\TODO`/`\CITENEEDED` string, (c) the still-shipping placeholder
detector and architecture figures (B5), and (d) a caption/axis-label
mismatch on `efficiency_vs_ekin.pdf`. None of these are physics
defects; none require a re-analysis; all can close in one pass. A
PRC editor would forward to referees on submission — this is no
longer desk-reject territory.

## 2. Remaining blockers

**B-1 (was B5). Two placeholder figures still ship.**
`main.tex:281` (`figs/detector_layout_placeholder`, caption
"(Placeholder)") and `main.tex:447`
(`figs/gnn_architecture_placeholder`, caption "(Placeholder)").
Only remaining hard blocker to the submit button.

## 3. Major concerns

**Ma-1. Body text still contradicts Fig 2.** Abstract now says
"<25 % for 1 ≲ E ≲ 3, ~30 % toward edges" (`main.tex:118–119`)
matching `ereco_vs_etrue.pdf`. But `main.tex:544–549` was not
touched: still asserts "Linearity is within 10 % for 0.7 ≲ E ≲ 5
GeV" and "σ_E/E < 10 % over the same interval". Referee reads
Sec 5.2, marks "major" the same day.

**Ma-2. `efficiency_vs_ekin.pdf` axis label ≠ caption axis.**
X-axis of the regenerated PDF is `E_pred [GeV]`; caption at
`main.tex:663–664` promises $\varepsilon_n(\Ekin)$ (E_true).
One-word caption + paragraph patch.

**Ma-3. Sensitivity yield bumps at 0.75 and 2.55 GeV — Sec 6 does
not name them.** Right panel of `sensitivity_yield.pdf` shows local
maxima in all three N_true curves. Diagnosis (background-leak at
300 MeV ToF threshold + regressor migration toward spectrum peak;
signal-only correction collapses 0.75 bump from closure 1.70 → 1.04
and 2.55 bump from 1.81 → 1.49) is well-defined. Sec 6 item 3
addresses integrated $C_c > 1$ but not per-bin locations. Add two
sentences after `main.tex:987`.

**Ma-4. FoM look-elsewhere (v3 M2, v5 M6) still not addressed.**

**Ma-5. Sec 6 item 4 argumentative circularity survives verbatim.**

## 4. Minor issues

- Three red `\NUM{}` at `main.tex:310, 322, 566`.
- `\TODO{re-test at full SMASH statistics}` at `main.tex:965–967`.
- `\CITENEEDED{SMASH DOI update}` at `main.tex:1260`.
- `%% TODO(authors)` at `main.tex:96`.
- Inline `thebibliography` at `main.tex:1151`, not migrated.
- Sec 6 item 1 (`main.tex:955`): 285:1 pooled-scaler is smoke-scale.
- `sensitivity_threshold_scan.pdf` title truncates.
- `sensitivity_yield.pdf`: zeroSpot hidden behind bigSpot.

## 5. v5 retire list

| Blocker | Status | Evidence |
|---|---|---|
| B1 MC-truth stale N_ev | CLOSED | mctruth_spectra.pdf N_ev=838,284 |
| B2 empty ratios | CLOSED | 4 populated curves in [0.7, 1.25] |
| B3 wrong-dataset efficiency | CLOSED (Ma-2 caveat) | three-dataset now |
| B4 abstract t*=0.305 error | CLOSED | body 489–493 reconciles |
| B5 placeholder figures | OPEN | → B-1 |
| M1 Fig 2 vs "<10 %" text | PARTIAL | abstract done, body 544–549 not → Ma-1 |
| M2 multiplicity 1-panel | CLOSED | two panels |
| M3 threshold "smoke" label | CLOSED | now "full stats, lift-cap" |
| M4 model-family circularity | OPEN | → Ma-5 |
| M5 defaultSpot C_c = 1.454 | CLOSED | main.tex:917 now 1.466 |

## 6. Figure audit

| Figure | Verdict | Justification |
|---|---|---|
| `performance_roc.pdf` | Ready | AUC/AP match main.tex:475–476,481. |
| `ereco_vs_etrue.pdf` | Needs-fix | Fig honest; body text at 544–549 still lies. |
| `multiplicity.pdf` | Ready | Two panels, correct labels. |
| `efficiency_vs_ekin.pdf` | Needs-fix (caption) | Three-SMASH ε; caption promises E_true but axis is E_pred. |
| `mctruth_spectra.pdf` | Ready | N_ev correct; 30 % suppression is disclosed sim artefact. |
| `sensitivity_yield.pdf` | Ready-with-caption-patch | Two-bump structure visible; caption should name them. |
| `sensitivity_ratios.pdf` | Ready | 4 curves, y-range 0.4–2.0. |
| `sensitivity_threshold_scan.pdf` | Ready-with-cosmetic | 5-threshold scan; title truncates. |

## 7. Bumps: paper text change needed

Yes — two sentences in Sec 6 item 3, no analysis change. Suggested
patch after `main.tex:987`:

> "Per-bin, this bias concentrates at two locations: near
> $E_{\text{pred}} \approx 0.75$ GeV, where the classifier purity
> locally drops to $\sim 0.61$ due to background leakage across the
> 300 MeV ToF threshold and the $1/\eff$ correction inflates the
> leak by a factor of $\sim 3.7$; and near
> $E_{\text{pred}} \approx 2.55$ GeV, where the energy regressor's
> modal migration toward the neutron spectrum peak accumulates
> events. Signal-only correction (numerator restricted to $y=1$
> clusters) collapses the 0.75 GeV bump from closure = 1.70 to 1.04
> and reduces the 2.55 GeV bump from 1.81 to 1.49. Both structures
> track across the three $\Usym$ samples to $\leq 11\%$ spread and
> therefore do not bias the cross-dataset closure metric of
> Sec.~\ref{sec:closure}."

## 8. Newly opened concerns

- N1: `efficiency_vs_ekin.pdf` E_pred axis vs E_true caption — Ma-2.
- N2: DCM t*=0.305 collides numerically with bigSpot SMASH
  t*=0.305 (`main.tex:867, 918`). Add "(DCM)" in abstract L115.
- N3: `tab:closure` cites preprocess 4284505+4290339,
  sensitivity 4284506+4300982; `tab:purity_locked` cites
  4300982+4290339. Two-line audit note.

## 9. Fastest path to PRC submission — 5 ordered PRs

1. **Placeholder retirement (1 day, blocker).** Produce clean
   `detector_layout` and `gnn_architecture` TikZ/PPT; delete
   "(Placeholder)" prose.
2. **Body/Fig 2 reconciliation + caption patches (½ day).**
   (a) Rewrite `main.tex:544–549` to match abstract wording,
   (b) reword `main.tex:663–664` caption to `E_pred`,
   (c) add "(DCM)" disambiguation to abstract L115.
3. **Sec 6 item 3 two-bump paragraph + tab:closure/tab:purity_locked
   provenance audit (½ day).**
4. **Marker + TODO retirement + FoM held-out or paragraph deletion
   (1 day).**
5. **Bib migration + cosmetic figure passes (½ day + round-trip).**

**Wall-time estimate:** 3 days + author sign-off round-trip.
Submit-ready by end of week if the placeholder figures land.
