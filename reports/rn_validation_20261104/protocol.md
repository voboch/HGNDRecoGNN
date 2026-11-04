# Fixed-window Rₙ after no-floor training

04.11.2026. User authorised continuation, paper revision, critics and commits.
Copilot remains disabled. This is a local reduction of saved predictions;
no training, inference, F scoring, cut fit or calibration is performed.

Population: the accepted 286 truth-bound validation contents, excluding the
four unbound groups only from physical-truth comparisons. This validation
participated in epoch/cut selection; it is not a fresh holdout. Compare native
raw B0 and no-floor raw, selected e_cls/e_en, same fold/seed B0 cut, score>cut.
Candidate identity and baseline prediction order must follow the pinned actual
producer, not a positional merge assertion alone. Verify the baseline source
digest against git commit d739c54 and its archived complete records.

Rₙ=N(E≥2 GeV)/N(0.3≤E<1 GeV), with no high-energy upper bound. These are
inherited observable bands, held fixed to isolate the label intervention;
this analysis does not optimise them. C1: native-positive, native energy;
C2: selected native-positive, native energy; C3: selected native-positive,
arm raw energy; C4: all selected, arm raw energy. Thus both arms use the same
native reference and T1; C1 and T1 must be exactly invariant. Expanded labels
are retained only for attributing the added-soft contribution to C4.

Reuse certified per-event primary counters and existing per-file truth only
after checking primary member, all candidate rows, max(candidate Row)+1,
number of candidate events and all T0/T1/T2 counts. Span assumes sequential
processing from Row0. The old report-fold is not the current validation-fold;
do not reuse old reconstructed predictions/cuts. No new raw-primary reduction.

Estimate each potential Rₙ by sum(high)/sum(low) over whole contents. Use the
inherited 400 file-bootstrap draws, seed20261026, independently within each
potential; this count gives about3.5% Monte Carlo relative error of an SD.
Reuse identical draws across arms, rungs, seeds and all derived contrasts.
Compute 18/0,90/0,90/18 within each replica. Report paired file SD for arm
differences and C4−T1; training-seed differences additionally have mean±SE
and conditional t(df2), with no multiplicity-adjusted generalisation claim.
Stop on nonfinite inputs, zero denominators or any failed input identity.

Prediction before this reduction: T1/C1 changes exactly zero by construction.
The direction of the C4 arm change is unspecified: new training changes both
scores and migration, and the added MC targets are outside the truth bands.
Large soft-target MAE improvement alone does not predict Rₙ closure. The
physical diagnostic decides whether a separate frozen F evaluation merits
planning; no such evaluation is part of this calculation.

Expected cost: one local pass over the already extracted exports and certified
event counters, minutes of CPU and no new GPU allocation. No new unit tests
or test-queue jobs under the existing human exception; retain source, key,
native-label, numerical and coverage assertions on the executed data path.
