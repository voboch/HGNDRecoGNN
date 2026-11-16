# Slurm monitoring: SMASH architecture fits

Read-only status check on 2026-11-16. `fit_submissions.json` recorded the launch-time state as two running tasks in array 4511720, with remaining tasks limited by `JobArrayTaskLimit` and three dependent arrays pending. That snapshot is superseded by live Slurm accounting.

All 72 top-level tasks in arrays 4511720, 4511721, 4511722, and 4511723 are now `COMPLETED` with exit code `0:0`; `squeue` has no remaining tasks. Each of the four epoch segments has 18 fit receipts, and every receipt reports `state: complete`. Receipt coverage is therefore 18/18 per segment for 0–8, 8–16, 16–23, and 23–30. No failed, cancelled, or timed-out Slurm tasks were reported. This confirms execution and publication of fit summaries through epoch 30; it does not establish scientific acceptance or validate the model comparison.

The receipts are on the cluster at `/home/vbocharnikov/hgnd_experiments/smash_architecture_20261112_2d31e3e25c45_receipts/`, with fit logs in its `logs/` subdirectory. Checkpoint and campaign outputs are under `/scratch/vbocharnikov/hgnd/experiments/smash_architecture_20261112_2d31e3e25c45/`. Local plan: [fit_launch_plan.json](../../smash_architecture_20261112/fit_launch_plan.json); submission snapshot: [fit_submissions.json](../../smash_architecture_20261112/fit_submissions.json).

For scientific acceptance, inspect the receipt-bound `summary.json` files and checkpoints in the campaign output, compare the intended validation metrics and epoch-selection rule, and produce the planned comparison/acceptance artifact with provenance links. Preserve the existing split and preparation manifest bindings. No acceptance artifact was inferred from Slurm state or receipt counts.

Raw compact status and receipt counts: [slurm_snapshot.txt](slurm_snapshot.txt). No project source or result files were changed.
