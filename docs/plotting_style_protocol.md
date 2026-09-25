# Plotting style protocol

Conventions for every figure produced by this repository, aligned with what
referees at Physical Review C, Nuclear Physics A, Computer Physics
Communications and NeurIPS/ICML expect. The aim is that a figure can be read
without its caption and cannot mislead when read quickly.

Implemented in `src/hic_fusion/plotting/style.py`; scripts should call
`apply_style()` rather than setting rcParams themselves.

---

## 1. Size and type

Figures are drawn at **the size they will be printed** and included with no
`width=` scale factor. Drawing a 7-inch figure and scaling it into a 3.3-inch
column shrinks every label by the same factor: the paper's figures were at one
point rendering at ~4 pt against Elsevier's 7 pt minimum.

| target | width | LaTeX |
|---|---|---|
| single column (`cas-dc`, a4) | 238.25 pt = 3.30 in | `\begin{figure}` + `\includegraphics{}` |
| full width | 494.51 pt = 6.84 in | `\begin{figure*}` + `\includegraphics{}` |

Measure `\the\columnwidth` **at the figure's location**, not before
`\maketitle` — the frontmatter is single-column and reports the wrong value.

- Base font 8 pt, ticks and legends 7 pt. Nothing below 7 pt after printing.
- **Vector output only** (PDF) for anything in the paper. PNG at 200 dpi is
  acceptable for the weekly report, which is read on screen.
- `pdf.fonttype: 42` — embed TrueType, not Type 3. Elsevier rejects Type 3.
- Aspect ratio ~1.4:1 to ~2:1 for single panels. Never taller than wide unless
  the observable demands it.

## 2. Colour and encoding

**Colour is never the only channel.** Roughly 5% of readers cannot separate red
from green, and figures are still printed in greyscale.

| role | colour | line | marker |
|---|---|---|---|
| reference (UrQMD / AAMCC) | `#2b2b2b` near-black | solid, filled band | `o` |
| prior (divisive sampler) | `#B22222` firebrick | dashed | `s` |
| model (MixFlow) | `#1f77b4` blue | solid | `^` |
| peripheral variant of any arm | same hue, lighter | dotted | open marker |

- The reference is always the darkest and always plotted **first**, so the
  comparison arms sit on top of it.
- Sequential data: `viridis`. Diverging residuals: `RdBu_r`, centred on zero,
  with symmetric limits.
- Never `jet`.

## 3. Uncertainties

**Every point that carries an uncertainty shows it.** A figure with no error
bars asserts that the uncertainty is negligible, which is a claim.

- Flow coefficients and any per-event observable: **event-level bootstrap**.
  Per-particle standard errors treat tracks within an event as independent and
  understate the error; they are not acceptable for anything quoted.
- Where arms share events (the usual case here — the same events transported),
  plot the **paired difference** with its own bootstrap, not two separately
  errored curves that a reader will difference by eye.
- Bootstrap resamples stated in the caption.
- Bands for continuous quantities, capped bars for binned points.

## 4. Ratio and residual panels

Any comparison against a reference carries a **ratio or residual sub-panel**,
height ratio 2.5:1, sharing the x-axis. The eye cannot resolve a 10% difference
between two log-scale histograms; the ratio panel is where a referee looks.

- Ratio panels get a unity line and a symmetric y-range.
- If the reference arm is missing, the figure **must fail**, not silently drop
  the curve. A two-curve plot under a three-arm title has already happened here
  once.

## 5. Axes and labels

- Physical quantity **and unit**: `$p_T$ [GeV/$c$]`, `$b$ [fm]`. Dimensionless
  quantities say so or carry none.
- Distinguish `$\eta$` from `$y$` in the label **and** in the caption. They
  differ by a factor ~7 in the $v_1$ slope at 2.87 AGeV; conflating them is a
  physics error, not a typo.
- State the particle species and the selection in the axis title or caption.
  "participant protons" and "all participant hadrons" are different observables
  and the meson fraction differs between arms.
- Log scale whenever the dynamic range exceeds ~50×; say so on the axis.
- Impact-parameter window in every panel title or the caption. No figure in
  this project is window-independent.

## 6. What the caption must contain

Four things, in order:

1. **What is plotted** — observable, species, selection, $b$-window.
2. **What the arms are** — and which is the reference.
3. **The quantitative result**, with its uncertainty. A caption that says
   "good agreement" is not a result; one that says "KS $=0.011$ vs.\ $0.170$
   for the prior" is.
4. **Provenance** — the script and the committed artefact, so a referee can
   regenerate it.

If any panel of a multi-panel figure contradicts the caption's headline claim,
the caption **names that panel**. A four-panel claim that one panel falsifies
is the kind of thing a referee finds and an author should have.

## 7. Central vs peripheral

This model is two specialists (or one unified model — see the CP25 decision),
and almost every conclusion differs between the windows. Any figure reporting a
central result should either show the peripheral counterpart alongside it or
state in the caption that the peripheral behaviour differs and where it is
reported.

Convention: **central left / peripheral right**, or central top / peripheral
bottom for 2×2 layouts. Shared y-limits within a row so the panels can be
compared by eye.

## 8. Paper vs appendix

| main text | appendix |
|---|---|
| physics observables a referee judges the method on | training diagnostics, loss curves |
| the headline comparison and its failure modes | per-seed breakdowns |
| anything an abstract claim rests on | conservation and budget closure checks |
| participant **and** spectator performance | throughput and cost tables |

A figure whose only audience is the authors belongs in the appendix or in the
weekly report, not in the main text.

## 9. Honesty rules

These are the ones that have actually been violated in this project, each at
least once:

1. A figure must not be drawn without its reference arm.
2. A caption must not claim an improvement that one of its panels contradicts.
3. Pseudorapidity must not be labelled rapidity.
4. A mixed-species curve must not be labelled by a single species.
5. Per-particle errors must not be presented as event-level errors.
6. A quantity that is a property of the prior must not be presented as a
   property of the learned model.
7. Two arms measured on different populations, sample sizes or estimators must
   not be plotted as if commensurable.

---

## Appendix A — conformance notes from the HGND spectral-hardness report

Applying this protocol to `results/spectral_hardness/` surfaced the following,
recorded so the next figure starts from them.

**§3 / §9.5 — the error-bar rule needs measuring, not assuming.** The protocol
requires event-level bootstrap because per-particle errors understate. That is
correct for per-event yields here: the HGND-band single-species yields have an
event-clustered error **2.31×** the per-particle value, multiplicity
fluctuations dominating. But for the spectral-hardness ratio R the measured
inflation is **1.01 / 1.09 / 0.98** — consistent with unity — because R's
numerator is rare (~0.37 high-energy neutrons per event) and carries little
intra-event correlation. The rule is therefore better read as *measure the
clustering factor and state it* than as *always bootstrap*. Both numbers now
appear in the captions that quote them.
(`results/spectral_hardness/event_bootstrap_check.txt`, 400 resamples.)

**§5 — the b-window line is load-bearing.** "No figure in this project is
window-independent" forces an explicit statement when the impact parameter is
*unavailable*, which is the situation in the primary exports (`B = -1`). Writing
"no impact-parameter selection (B unavailable)" on the panel surfaces the gap on
the figure itself rather than leaving it to a systematics section.

**§9.7 — grouping beats a single ranked list.** The natural presentation of
"which observable is most sensitive" is one sorted bar chart, but the candidates
are measured on different event samples, sizes and estimators. Figure 2 of that
report is grouped by population with the grouping stated in the caption, and
comparisons are meaningful within a block only.

**§2 in a two-theme document.** The reference-darkest rule is a print
convention. For an on-screen artifact rendered in either theme the role ordering
is preserved and the luminance adapted: reference `#2b2b2b` in light mode and
`#E8E8E8` in dark, with the firebrick and blue arms adjusted to match. The
non-colour channels (line style, marker shape) are theme-invariant and carry the
identity on their own.

**§1 — axis limits must be derived from the data, and containment asserted.**
Figure 1 panel (a) was first drawn with hardcoded limits of 0.03–3 GeV⁻¹ for a
spectrum spanning 0.056–21.3: **75 % of the plotted points sat above the top of
the panel**, the worst by a factor 7. The XML well-formedness and
bounds-in-viewBox checks both passed, because the marks were valid SVG at
coordinates outside the *panel* rectangle but inside the *canvas*. Two rules
follow: compute limits from the plotted values rather than typing them, and
assert that every mark and error-bar end lies inside its panel, not merely
inside the figure. Snapping limits to a 1–3–10 sequence rather than full decades
keeps the data filling the panel (86 % of the height here, against 65 % for
decade rounding). `results/spectral_hardness/chkfig.py` performs the check.

**Mechanical.** SVG text labels containing `<` or `>` — routine in physics
labels such as `|y_cm| < 0.5` — must be escaped as `&lt;`/`&gt;` or the figure
silently fails to parse. This broke two figures before an XML well-formedness
check caught it; that check is now part of the build.
