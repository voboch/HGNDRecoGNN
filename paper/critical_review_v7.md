# PRC referee review v7 — `paper/main.tex` (through 15ca130)

## 1. Verdict

**Major revision.** The rework is a large scientific improvement on v6: the
paper now controls centrality properly, states the acceptance geometry, and
reports where the reconstruction loses the signal instead of quoting a
provisional sensitivity. Those are the three things v6 could not have asked for
because the analysis did not exist yet.

The revision also introduces a self-contradiction serious enough to block
submission on its own. Section~\ref{sec:acceptance} argues that the polar-angle
band is not the detector acceptance — 0.0879 sr against 0.0114 sr, and with a
fixed reaction plane the azimuth cannot be weighted away. The headline table
three paragraphs later, and the summary, then present band-computed numbers as
acceptance numbers. A referee who reads the two sections in order will stop
there.

Verdict is not "minor revision" because B-2 and B-3 require either a
re-analysis in the stated acceptance or an explicit, consistent relabelling
throughout, and because the paper now contains two mutually unreconciled
characterisations of the same checkpoint's energy performance (Ma-1).

## 2. Blockers

**B-1 (carried from v6, unresolved). Two placeholder figures still ship.**
`main.tex:297` (`figs/detector_layout_placeholder`) and `main.tex:463`
(`figs/gnn_architecture_placeholder`). Unchanged since v5.

**B-2 (new). Band numbers presented as acceptance numbers.**
Sec.~`sec:acceptance` establishes that the acceptance is a front-face rectangle
covering 12.9 % of the solid angle of the `[8.9°, 13.1°)` band, and that the
azimuthal restriction carries physics because `v1` is energy dependent
(+0.090 above 2 GeV, +0.005 below 1 GeV). `tab:truthscan` and
`fig:spotresponse` are nevertheless the band-integrated full-production
results. The summary then states outright that "neutron and proton spectral
hardness **in the detector acceptance** respond to `Usym`" (`main.tex:1336`).
They are band results. Either relabel every occurrence as the angular band and
say plainly that acceptance-corrected numbers are pending, or recompute.
The manuscript currently claims the stronger of the two.

**B-3 (new). The ladder inherits the same ambiguity.**
`tab:ladder` row 1 reads "primary neutrons in the angular band", and rows 2–4
are detector-level. The table therefore mixes a band selection with detector
selections down its rows without saying that the first row is not the same
acceptance as the rest. The step from +2.65 % to +3.67 % between rows 1 and 2
is partly an acceptance change, not only a transport effect, and the text
attributes it to transport.

## 3. Major concerns

**Ma-1. Two incompatible statements of the same checkpoint's energy
performance.** Sec.~5.2 says linearity is "within ~25 % for 1 ≲ E ≲ 3 GeV"
(`main.tex:~500`). Sec.~8 reports the previous checkpoint at +57.5 % bias at
1.4–1.8 GeV with 26–34 % resolution. Both are defensible — 5.2 is the DCM test
split and 8 is the SMASH test split — but neither passage says so at the point
of contradiction, and the abstract quotes the 5.2 characterisation while the
new figure shows the 8 one. Reconcile explicitly, naming the sample in both.

**Ma-2. Closure uniformity and hardness sign reversal are never reconciled.**
Sec.~7 reports per-cluster closure uniform across the three samples to 0.62 %
at the locked working point (`main.tex:1022, 1057`). Sec.~8 reports that the
same pipeline reverses the sign of the hardness response. Both can hold —
closure is a yield-integrated quantity and hardness is a spectral-shape
quantity, and a response that is uniform in normalisation can still tilt — but
the paper never says this. As written, a referee reads Sec.~7 as evidence that
the pipeline is sample-independent and Sec.~8 as evidence that it is not.
One paragraph closes it.

**Ma-3. The retrained estimator is four epochs of twenty, and the abstract
does not say so.** `main.tex:144` quotes the retrained linearity without the
caveat that appears only in Sec.~8.4. The number is a lower bound on the
architecture and should be labelled as such wherever it is quoted.

**Ma-4 (carried from v3/v5/v6, now worse). Look-elsewhere.** v6 flagged the
original FoM. Sec.~8.3 adds a second scan over 289 configurations of score
threshold and energy window, selects one, and quotes the test value. The
protocol is correct — selection on train+val, one configuration to test — but
the paper gives no trials-factor discussion for the development-set number it
reports alongside, and the reader is shown the top eight configurations of 289
in the accompanying analysis output. State that the development FoM is not a
significance.

**Ma-5. The 284-job requirement is quoted without its assumptions.** The
statement that ~284 test jobs are needed for 3σ assumes the error scales as
N^(-1/2) with job count and that the truth effect is exactly 4.27 %. Both are
reasonable; neither is stated. It is also computed from the band effect, which
B-2 says is the wrong acceptance.

## 4. Minor issues

- Editorial markers still ship: `\NUM{}` at `main.tex:326, 339, 585`;
  `\CITENEEDED` at `main.tex:1564`; `%% TODO(authors)` at `main.tex:96`.
- `figs/b_distributions.pdf` is committed but never referenced.
- Terminology drift: the new sections say "generation jobs", the older ones say
  "runs" and "files" for the same unit. Fix to one term, and define it where
  the job-level bootstrap is introduced, since the whole error treatment
  depends on what a job is.
- Sec.~6.1 states the reweighting "costs 0.1 % of the effective statistics and
  discards no events" — true for the full production, but the injection test
  quoted immediately after discarded a large fraction by construction. Separate
  the two.
- The abstract is 436 words. PRC allows it, but the third paragraph carries six
  numbers that are developed properly in the body; two would serve.

## 5. What improved since v6

- v6 Ma-1 is resolved: Sec.~5.2 no longer claims 10 % linearity over 0.7–5 GeV.
- The centrality treatment is new and is the strongest part of the paper. The
  job-level bootstrap, the 2.85 inflation factor, and the two-directional
  validation (same-sample null, injected −1.56 fm bias) are exactly what a
  referee would have demanded and are supplied unprompted.
- The negative results are stated as negative: the `n/p` yield ratio is
  excluded on a monotonicity argument, and the held-out test is called
  inconclusive rather than null.
- The `Usym` reach is correctly labelled a truth-level statistical reach and
  explicitly not a bound on `L`.

## 6. Recommendation

Reject-and-resubmit is not warranted; the analysis is sound and the exposition
is mostly honest. Close B-1 through B-3 and Ma-1 through Ma-3, and the paper is
a minor revision. B-2 is the one that cannot be argued away: the manuscript
currently makes a claim about the detector acceptance that its own acceptance
section forbids.

## 7. Post-review fixes applied (commit following this review)

Addressed in the manuscript immediately after the review was written:

- **B-2** — every truth-level number is now labelled as the polar-angle band,
  in the abstract, the table, the table caption and the summary. The summary
  additionally states that the front-face acceptance covers 12.9 % of that
  solid angle and that re-reducing the production in it is outstanding, and
  the section notes that the acceptance roughly doubles the response on the
  sub-samples where both selections exist, making the band numbers the
  conservative ones.
- **B-3** — `tab:ladder` row 1 is relabelled and the caption states that the
  step from row 1 to row 2 folds an acceptance change together with transport.
- **Ma-1** — both energy-performance passages now name their sample, and
  Sec. 5.2 points forward to the domain-transfer result.
- **Ma-2** — a paragraph reconciling yield closure with spectral tilt.
- **Ma-3** — the four-of-twenty-epochs caveat now travels with the number in
  the abstract.
- Minor — `figs/b_distributions.pdf` is now referenced as `fig:bdist`.

Withdrawn from this review: the `sec:reproducibility` item. The label exists at
`main.tex:1444`; the build message was a normal first-pass warning, not a
defect. Recorded rather than deleted, since a review that quietly drops its own
errors is worth less than one that does not.

Not addressed, and still blocking: **B-1**, the two placeholder figures.
Still open: **Ma-4** (look-elsewhere), **Ma-5** (assumptions behind the 284-job
requirement), and the remaining editorial markers.
