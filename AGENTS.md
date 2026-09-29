
## Running training on HPC
When running, submitting, or editing training jobs on the HSE cHARISMa
cluster, follow HPC.md in the repo root. It is directive. Do not run
training on a login node; always smoke-test on --partition=test first.

## Justifying every fixed parameter

This is a scientific analysis, so no chosen constant may appear without its
justification attached. A number written into a script or quoted in a note —
an energy threshold, a score cut, a binning, a fit range, a rejection criterion
— must arrive with **both** of the following, at the point of use:

1. **A plot, whenever the choice is visually representable.** If the parameter
   was picked by optimising something, show the objective over the parameter,
   not only its argmax. A scan surface makes "a broad optimum" checkable and
   exposes the case where the optimum is a spike, a boundary artefact, or a
   tie. If the parameter follows from a physical feature, plot the feature.
2. **A comment saying why, in the code and in the prose.** State what the
   parameter trades off and what would change if it moved. "Chosen on
   development data" is provenance, not justification.

The failure mode this prevents has already occurred here more than once: a
polar-angle band inherited without checking it against the detector geometry,
an energy denominator placed where the detector records two per cent of what
arrives, and `~2.2 GeV` quoted as a window centre with nothing behind it. Each
read as settled because it was written down.

Worked example, the hardness window: `scripts/hardness_threshold_opt.py`
searches the centre and half-gap, `scripts/plot_hardness_surface.py` draws the
response and significance over the whole scanned plane with the unordered
region masked, and `fig_np_spectra_relation` shows the isovector signal
crossing unity near 1.8–2.0 GeV, which is the physical reason the optimum sits
just above it. The number, the surface it came from, and the physics that
explains the surface are all in the note.

A parameter that genuinely has no better justification than convention should
say so in those words, so a reader can see it is unexamined rather than
assume it was derived.
