# Rules plus ToF versus the fixed hardness-selected GNN policy

Use the 282 truth-bound odd parts 1/3/5 of the accepted cross-fit split,
the processed Row spans and the 400 archived odd-file draws. Retain the
nine GNN score cuts from the 90% validation-purity target selected on
even files by the signed 90/0 hardness contrast under its precision cap.
Keep saved calibrated energies and use every selected cluster. The rules
arm uses the existing 26-neighbor spatial components, no hits in the two
front layers and at least two cells. The maximum-speed hit from the
accepted jittered observations defines the head. Reject its component if
the head does not satisfy 0 < beta < 1. Use no score or energy fit on the odd files.

The primary comparison uses the open hard band E >= 2 GeV from the
registered cut selection, with the [2,4) GeV band as a check within the
nominal timing range. Use [0.3,1) GeV for geometric MC primaries and
the inherited strict >0.3 GeV candidate floor. Check that the strict
floor reproduces the archived GNN soft counts exactly. Compare the same
processed spans even when either reconstruction has zero candidates.
The geometric MC reference is the forward ray in the nominal rectangle.
Replay the exact certified per-event counters from the pinned primary
CSV, then count the additional [2,4) truth band within the same span.
Candidate comparisons precede background subtraction and deconvolution.

Sum bands over files, compute Rn and the three signed contrasts within
each seed, then take the equal-seed mean. Compute signed differences
against MC and between methods in each common draw. Report empirical
Q16–Q84, its half-width and the paired difference of absolute residuals.
Widths describe variation over files at fixed models. The odd files
were excluded from cut selection and had previously reported outcomes.
No new evaluation scope, model calls or fitting is used.
