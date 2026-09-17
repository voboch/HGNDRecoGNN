# Plot conventions — BM@N HGND analysis figures

House style for every figure that leaves this repo (paper, note, talk, report).
It follows standard HEP practice so figures read correctly to this audience and
drop into a paper without rework.

Sources consulted: the [ALICE policy for official
figures](https://alice-figure.web.cern.ch/policy_for_official_figures) (required
elements and the statistical/systematic distinction), [ROOT
`TStyle`](https://root.cern/root/html534/TStyle.html) / [`TGaxis`](https://root.cern.ch/doc/master/classTGaxis.html)
defaults (the frame and tick behaviour that HEP eyes expect), and
["Plotting the Differences Between Data and Expectation",
arXiv:1111.2062](https://arxiv.org/pdf/1111.2062) (ratio/pull panels).

---

## 1. Frame and axes

- **Frame box on all four sides.** Not two open axes. The frame is what makes a
  plot read as a HEP figure at a glance.
- **Tick marks point inward**, on all four sides (`SetTicks(1,1)`), with minor
  ticks between majors. Never outward, never on two sides only.
- **No grid lines.** A grid is a spreadsheet convention. If a reference level
  matters (unity in a ratio), draw *that one line*, dashed — not a grid.
- **Axis titles carry units**, `E_kin (GeV)`, `p_T (GeV/c)`, or bracketed
  `[GeV]` — pick one and hold it across the figure set. Yields are normalised
  per event and say so: `(1/N_ev) dN/dE_kin (GeV^-1)`.
- **Log scale where the dynamic range demands it**, which for spectra it usually
  does. A log y-axis showing one decade is a linear axis wearing a costume.

## 2. Marks

- **Data: markers with error bars.** Never a line connecting data points — a
  connecting line asserts an interpolation the measurement does not have.
- **Simulation / theory: step histograms or smooth curves**, no markers. This is
  the visual grammar that separates "measured" from "calculated"; MC drawn with
  markers reads as data.
- **Binned quantities are steps, not curves.** A spectrum in bins is a histogram;
  drawing it as a smooth spline invents structure between bin centres.
- **Marker shapes must survive greyscale**: filled circle, open square, filled
  triangle, open circle. Shape carries identity, colour reinforces it — never
  colour alone (also the accessibility requirement).
- Markers ≥ 8 px so the error bar is readable against them.

## 3. Uncertainties — the non-negotiable

- **Statistical uncertainty: vertical error bars.** Horizontal bars show bin
  width only where that is the point; otherwise omit them.
- **Systematic uncertainty: open boxes or a shaded band**, drawn separately from
  the statistical bars. ALICE's convention is the vertical bar for the
  uncorrelated part and a band for the correlated part.
- **Every figure states which is which**, in the legend or caption. "Uncertainties
  are statistical only" is a complete and acceptable statement — silence is not.
- **A point without an uncertainty is not a measurement.** If a quantity is
  genuinely exact (a bin edge, a cut value), it is not a data point.

## 4. Ratio panels

- Ratio or double-ratio goes in a **lower panel sharing the x-axis**, with no gap
  and no repeated x labels between panels.
- **Dashed line at unity.** Always.
- Ratio panel y-range is chosen to show the deviation, not to flatter it — and
  the range is stated if it clips any point.
- The upper panel may be log while the ratio panel is linear; that pairing is
  standard.

## 5. Annotation

Every figure carries, inside the frame:

- **System and energy**: `Xe+CsI, √s_NN = 2.87 GeV` (or `E_kin = 3.8A GeV` as
  appropriate) — a plot that does not say what collided is not interpretable.
- **Status label**: `BM@N HGND simulation`, `work in progress`, or `preliminary`.
  Simulation figures must say they are simulation.
- **What is varied**, here `U_sym = 0 / 18 / 90 MeV`, with the potential given in
  physical units, not internal dataset names. `zeroSpot` / `defaultSpot` /
  `bigSpot` are repo identifiers and must not appear on a figure for an external
  audience; map them to U_sym values.
- **Legend inside the frame**, no heavy border, no fill that hides data.

## 6. Colour

Colour is the *secondary* channel — shape and line style carry identity first, so
the figure survives greyscale printing and colour-vision deficiency.

- **An ordered parameter gets an ordered ramp** (single hue, light→dark).
  `U_sym = 0, 18, 90 MeV` is ordered; giving it categorical hues implies the
  three are unrelated identities.
- **Unordered categories get a categorical palette**, assigned in fixed order.
- **A ratio around unity gets a diverging pair** with a neutral midpoint.
- Validate rather than eyeball: the palette must clear colour-vision-deficiency
  separation and contrast against the plot surface in both light and dark.

## 7. Before shipping

- [ ] Frame closed, ticks inward, no grid.
- [ ] Every data point has an uncertainty, and the caption says what kind.
- [ ] Axis titles have units; yields say what they are normalised to.
- [ ] System, energy, and simulation status are on the figure.
- [ ] U_sym in MeV, not dataset codenames.
- [ ] Legible in greyscale.
- [ ] Rendered and *looked at* — bounds, label collisions, overflow.
