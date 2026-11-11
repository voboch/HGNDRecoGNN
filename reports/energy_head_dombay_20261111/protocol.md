# Energy-weighted fine-tuning of the SMASH regression branch

Registered 11 November 2026. Start each of the nine accepted fold/seed runs
at its selected energy epoch. Freeze the GraphSAGE representation in eval
mode and update only `shared_reg` and `reg_head`. The classification score
comes from the separately selected classification epoch and is reused from
the verified baseline. Keep the native packs, targets, folds and scalers.
The branch initialisation, event batches, dropout stream and schedule match
between the two new arms. Save the final branch at epoch 15. Validation
supplies a diagnostic loss and does not select an epoch.

Use Dombay's ten equal-width bins over the positive training target range,
mean-one inverse-occupancy weights, 15 epochs, Adam at 1e-4 with decay 1e-5,
and a factor 0.1 at epoch 5. Empty bins carry zero weight and receive no
training objects. The notebook continued a multitask fit. Here a frozen
representation and score isolate the regression branch. The notebook's
event batch size 256 is retained. Compare weighted MSE with unit-weight MSE
under the identical fine-tuning recipe, and with the original raw SMASH
checkpoint. The notebook's stored weighting cells use the last test-loader batch. Here
the weights use the whole training fold so no evaluation energies enter the fit.
Preserve the final branch, occupancy, weights, learning curves and input
hashes. Flag a high weight and report effective sample size, without capping
or adjusting it after seeing outputs.

The predicted effect is a few percentage points less under-response near
4–5 GeV (baseline −8.3%) at a possible cost in central width. The 4–5 GeV
median and central half-width are the primary energy response endpoints.
Report their paired changes in all pre-existing bins and MAE on the original
286-file first-hit cohort. Keep exact candidate identities, baseline score
selections, timing mask and MC truth energies. The 5–6.5 GeV bin is sparse
and descriptive. A movement of the median toward zero comes with its
paired width change; the file intervals and values by training repeat
give finite-file precision and sensitivity to the initial checkpoint.

On the 282 odd files, use the frozen fold/seed cuts selected on even files
and the same selected candidates, including native-label negatives. Evaluate
all raw arms on the same files, plus the published calibrated baseline for
context. Keep `0.3 < E < 1` GeV and `E >= 2` GeV and the same geometric
primary MC counts. The established `2 <= E < 4` GeV range is a secondary
tail diagnostic. Use the archived 400 potential × part file draws; sum
counts and form ratios by seed before averaging the three contrasts.
Report 90/0 as primary and 18/0 and 90/18 as correlated secondary endpoints.
Report each contrast, its Q16–Q84 half-width, paired displacement from MC,
paired change from original raw and the unit-weight control, and change in
absolute displacement. No directional claim about contrast is registered:
a common energy shift can cancel in the ratio and migration across 2 GeV
depends on the potential. Odd outcomes cannot choose weights, cuts, bands
or calibration. The odd baseline has been inspected previously.

Resources. Nine tasks, one per fold/seed, with at most two active. Each
uses a V100, four CPUs, and at most 75 minutes. The accepted fold scoring
jobs took 45–48 minutes for three seeds and multiple checkpoints; each new
task caches one frozen representation per train/validation/test file and
trains only the two regression branches from the same cache. The first
full-scope task can serve as the execution pilot under the user's standing
instruction to continue without test-queue jobs. Its artifact is retained
as one of the nine tasks. No automatic requeue or full-model training.
