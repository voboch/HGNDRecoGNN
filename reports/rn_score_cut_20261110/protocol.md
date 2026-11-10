# Classifier score cut for neutron hardness

The existing operating point is the first score cut on the 401-point
[0.001, 0.999] grid reaching 70% native cluster count purity on pooled
validation for each fold and training repeat. It was selected independently
of the neutron hardness contrast.

The new policy maximizes signed reconstructed 90/0 hardness contrast
on the 286 even, truth-bound report files. Keep the SMASH-trained models,
score heads, energy calibration and analysis window fixed. Count all
accepted reconstructed clusters with score strictly above the cut.
The fixed denominator is 0.3 ≤ E < 1 GeV and the numerator is E ≥ 2 GeV.
The objective is the equally weighted mean of three training-repeat
values of 100 × [R_n(90)/R_n(0) − 1]. Each R_n is a ratio of summed
file counts. The cut for a given model is identical at all potentials.

The one-dimensional policy parameter is validation count purity. Use
60%, 65%, …, 95%, inherited from the 29 October protocol to bracket 70%
and cover stricter and looser cuts. The grid spacing is a convention.
For each policy and model, use the first original score-grid cut reaching
that purity in that model's pooled validation. This avoids assuming
scores are interchangeable between independently trained models. The
result is a maximum over this finite policy family.

Use saved out-of-fold test predictions on the even files. The model
that supplies scores for a file trained and validated on other files.
Each model's original purity curve was fitted on its validation files.
Resample 400 whole files within each potential and file part, with the
same weights across thresholds and repeats. Require finite positive
band counts in every potential and repeat in all draws, and a
seed-mean contrast SD no larger than twice that at 70%. This inherited
precision bound excludes the tail where denominator loss makes the
maximum unstable. Among eligible policies take the largest signed
central value, with the lower purity target winning an exact tie.
Show the full objective and the selected population so this convention
can be audited. Report its paired change from the baseline and the
corresponding separation in units of file SD as a diagnostic.

Write the frozen policy before reading odd-part candidate predictions.
On the 282 odd files evaluate only the baseline and the frozen policy,
with paired file uncertainty. No feedback from the odd files enters
selection. The odd baseline has been seen previously, so this is a
retrospective check, with no claim of fresh holdout. Report the gap to
the fixed primary MC contrast separately from the optimized objective.
