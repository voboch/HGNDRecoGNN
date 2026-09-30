
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

## Writing: no AI slop

Prose in this repository — notes, the manuscript, commit messages, docstrings,
figure captions — is edited against the `no-ai-slop` checklist
(https://github.com/petergyang/no-ai-slop). Load the skill when writing or
revising any of them. The patterns it removes are the ones that make technical
writing sound authoritative while saying less:

| pattern | fix |
|---|---|
| binary contrast — "It's not X, it's Y" | state Y |
| throat-clearing openers | delete, start at the point |
| faux-insight setups — "what most people miss" | make the claim stand alone |
| colon reveals, dramatic fragments | plain sentences |
| trailing "-ing" clauses posing as explanation | give the mechanism or the consequence |
| importance puffery — "plays a vital role", "pivotal" | state the fact, let the reader weigh it |
| weasel attribution — "studies show" | name the source or cut the claim |
| synonym cycling | repeat the clear word |
| fake-profound endings, summary recaps | end on the last concrete point |

Cut outright: delve, foster, leverage, utilize, facilitate, empower,
streamline, robust, cutting-edge, paradigm shift, game changer, transformative,
elevate, embark, harness. Cut unless load-bearing: just, simply, actually,
truly, fundamentally, importantly, crucially, it is worth noting, at its core,
in order to.

Two rules matter more here than in general writing. **Replace abstraction with
the number**: "improved resolution" is not a result, "resolution fell from 26%
to 15% at 2 GeV" is. And apply the **portability test** — a sentence that would
be equally true of any analysis is filler; delete it.

This does not license flattening the prose. Keep hedges that carry real
uncertainty, keep the caveat that a referee would ask for, and keep the
sentence that explains why a null is uninformative rather than negative.

## Marking MC truth

Every figure and distribution states whether it is **MC truth** or
**reconstructed**, in the caption and on the panel itself
(`plot_b_full.mark_provenance`). Truth quantities use information available only
in simulation — the generated nucleon list, true kinetic energy, the
truth-matched cluster label — and describe what a perfect detector would
measure. Reconstructed quantities use only what the pipeline derives from hits.

A figure travels without its caption into talks and slides, which is why the
panel carries the label too. Mixing the two silently in one figure made an
earlier draft read as though the detector had measured the truth-level response.
