# SMASH energy-head fine-tuning with capped weights

Registered 11 November 2026. Use the accepted Dombay-style experiment's
native packs, energy checkpoints, score predictions, cuts, folds and seeds.
Freeze the same regression input and fit only the regression branch for
15 epochs with the same optimizer, batch order and dropout stream. Reuse
the accepted ordinary-MSE branch as the unit-weight validation control.

Train three arms for each fold and seed. Use ten equal-width bins of native
positive training targets. For populated bins with occupancy fraction p,
use w = min(cap, k/sqrt(p)), where k gives a mean object weight of one.
The caps 2, 4 and 8 span a factor of four with at most an eightfold
contribution per object relative to the unit-weight control. Square-root
weights temper the inverse-occupancy gradient before clipping. The old
uncapped weights exceeded 3,600 in every fold. Keep the same target and
batches so only the training weights differ.

Select the cap separately for each fold and seed on that run's native
validation positives. Rank by smallest absolute median relative error in
4–5 GeV among candidates with whole-range MAE at most 102% of the unit
control and 4–5 GeV central half-width at most 105% of the unit control.
These tolerances permit about 2 MeV whole-range MAE and 0.4 p.p. tail width
above the previous raw benchmark. They constrain different aspects of
performance; they are practical margins, not significance thresholds.
If all caps exceed a limit, select the smallest maximum fractional
overrun of the two limits for the requested comparison and label the
result as failing the validation guard. Break ranking ties by smaller
cap. Record all validation metrics. Fit the existing monotone
inverse-median calibration to the selected head on the same validation
positives and freeze it before reading the fold's evaluation packs.
Calibration is reported as a separate variant and cannot select the cap.

The expected result is less positive overcorrection at 3–4 GeV and a smaller
4–5 GeV scale correction than with full inverse occupancy. The matched energy
endpoints are the 4–5 GeV median and Q16–Q84 half-width, with all prior bins
and overall MAE recorded. Replay the same 286 even-file first-hit cohort
and three seed-specific masks. Report initial raw, the ordinary-MSE control,
selected capped raw and its calibrated counterpart. The raw comparison is
primary. Odd-file contrasts use the same 282 files, candidate scores and
cuts, geometric truth, energy bands and 400 paired file replicas as the
accepted 11 November study. Primary contrast 90/0, secondary 18/0 and 90/18
and the bounded 2–4 GeV numerator. Report paired changes from original raw
and the unit control, width and signed and absolute displacement from MC.
There is no hypothesized contrast improvement. The odd files have been
examined previously and cannot choose a cap, cut, calibration or band.

Nine fixed fold/seed tasks with at most two running. V100, 4 CPUs, 55 minutes,
mem=0, no requeue. The preceding jobs took 13–17 minutes including two new
fits. This job reuses the unit fit and adds three capped fits from the
same cached representation. The standing user waiver of test-queue jobs
applies. A production task can be retained as the first execution check.
