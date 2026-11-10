# Matched energy response at the frozen SMASH cut

The accepted first-hit export covers the 286 even files. The tighter
SMASH cut leaves 67,645, 67,354 and 65,150 matched links in the three
repeats. The original DCM cut and the shared physical-timing mask
remain part of the match. Thus these figures describe a nested subset
of the original matched timing cohort. The comparison uses the
earliest retained hit, a straight path from the origin and the
exported hit time without a separate event start. MC association
already includes a ToF condition.

First-hit ToF gives MAE 166.2 ± 6.6 MeV, versus 89.8 ± 0.4 MeV for the
raw SMASH-trained estimate (file SD). Its median absolute error is
44–45 MeV across repeats, versus 54–55 MeV for the raw network. At
4–5 GeV calibration moves the median relative residual from −8.09%
to +3.10% and widens the central residual half-width from 8.48% to
15.43%. The 5–6.5 GeV bin contains 29–37 objects per repeat.

On these links, the raw-network minus first-hit-ToF 90/0 contrast
difference is −0.21 p.p., with paired Q16–Q84 [−1.38, +0.91] p.p.
The contrast comparison is conditional on these matches in the even
selection sample. The full odd-file results use the accepted
reconstructed candidates and their fixed primary reference.

[Linearity](energy_linearity.pdf) · [Resolution](energy_resolution.pdf) ·
[Values by bin](bins.csv) · [All matched contrasts and widths](matched_summary.json).
Reproduce with `python replay.py` in this directory using the project
environment. The runner checks the frozen choice SHA and replays the
original comparison before reducing the tighter match.
