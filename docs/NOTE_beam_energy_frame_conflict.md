# Note: the shared glossary's beam energy is right here and wrong next door

**2026-10-06.** Raised from the HIC-NeuralFusion side. Short note for whoever
owns the shared layer; no HGND file is changed by it.

## The shared rule

`thesis-kb/conventions/00-glossary.md` §1, dated 2026-10-05, states that
`2.87 GeV` is the beam **kinetic** energy per nucleon, that
$\sqrt{s_{NN}} = \sqrt{2 m_N (E_\mathrm{kin} + 2 m_N)} = 2.9845$ GeV, and that
**any figure annotated `√s_NN = 2.87 GeV` is mislabelled** and must be
regenerated or relabelled.

## It applies to HGND

`docs/PLOT_CONVENTIONS.md` §5 currently instructs exactly that annotation:

> **System and energy**: `Xe+CsI, √s_NN = 2.87 GeV` (or `E_kin = 3.8A GeV` as
> appropriate)

Both halves are wrong by the glossary's arithmetic, and this repository's own
`docs/plotting_style_protocol.md` §5 writes the beam as **2.87 AGeV**, i.e. a
kinetic energy per nucleon, which is consistent with the glossary and not with
the annotation. So HGND figures carrying `√s_NN = 2.87 GeV` are mislabelled and
the correct annotation is `√s_NN = 2.98 GeV`, or `E_kin/A = 2.87 GeV`.

**Nothing here is changed yet.** Relabelling touches generated figures, so it
belongs to whoever is regenerating them.

## What is wrong in the shared rule is its universality

The glossary opens "Both projects study the same system." The two projects do not
share a frame.

Measured on the HIC-NeuralFusion audit dump
(`outputs_hpc/conservation_cap20260930/cons_capcentral_s42.npz`), the UrQMD
reference record for its Xe+Cs sample carries

| | measured | required if fixed-target at $E_\mathrm{kin}/A = 2.87$ GeV |
|---|---|---|
| $\langle p_z \rangle$ | $-10.0$ GeV/$c$ | $+476$ GeV/$c$ |
| $\langle E \rangle$ | $374$ GeV | $616$ GeV |

A factor 47 in longitudinal momentum. That sample is generated **in the centre of
momentum**, where `2.87` is $\sqrt{s_{NN}}$ and not a kinetic energy, so
HIC-NeuralFusion keeps $\sqrt{s} = 375.94$ GeV and keeps writing
$\sqrt{s_{NN}} = 2.87$ GeV. Its written exception is
`HIC-NeuralFusion/docs/exception_beam_energy_convention.md`.

So `2.87` and `2.98` are not two labels for one number. They belong to different
frames: HGND fixed-target kinetic, HIC-NeuralFusion centre-of-momentum
$\sqrt{s_{NN}}$. The shared layer currently has this project's frame written over
the other's.

## Suggested shape for the shared fix

Keep the §1 correction, drop the claim that one frame covers both, and give the
table a column per project. Neither side's provenance is documented from
generator cards; on the HIC side that is its blocking checklist item C7, so the
measurement above is the evidence, not a citation.
