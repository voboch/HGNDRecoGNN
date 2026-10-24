# Remaining corrected-SMASH transfer work

The full CPU audits4437503/4437504/4437505 and test CUDA smoke4437645 passed.
Read `launch_status.json`, `validation.json`, `full_audit_validation.json`,
`gpu_smoke_validation.json` and `full_transfer_latest_status.json` first.
All are local records, not permission for another job or retry.

Full frozen transfer jobs4437674 zeroSpot,4437675 defaultSpot,4437676 bigSpot
already exist. Inspect actual Slurm state and publication receipts before
any action. Do not duplicate these jobs. They were PENDING at13:40MSK on
23October; that status is stale after this timestamp. Effective limits are
1h/1h/30min, no-requeue. bigSpot depends afterany4437674; concurrent limit2.
Any new scientific/infrastructure failure is preserved and stops that arm;
no automatic retry is authorized. Do not resume old DCM campaigns or automation.

Frozen code `/home/vbocharnikov/hgnd_experiments/smash_full_20261023_c206ad68c9`,
SHA c206ad68c9146c087f72e18bcc2473923cd2490e4d86890cb1ca560ce5cc89d1.
Scratch has the same basename under `/scratch/vbocharnikov/hgnd/experiments`.
Home receipts/logs are in the code path plus `_submission`.
The pinned external bound inventory SHA is
a3baf9d9379fb48f797a0ce47f1de1fba73e4495aa27edacf1208bab258723f4.
Index `audit_bundles.json` SHA is
0ef77203263b97b129576f17f6e83723d209ca2846de9b0218b8af9e798927c1.

After each completed job, its home receipt is
`transfer_<tag>_<jobid>.json`; output is one `inputs.tar` plus `manifest.json`
in scratch `transfer_bundles/<tag>_<jobid>`. Require Slurm COMPLETED/exit0,
receipt/manifest/archive hashes, then strict `smash_input_bundle.unpack` locally.
Merge the three disjoint tag trees into `full_transfer_inputs/transfer/`.
Create `full_transfer_summary_bundle.json` with complete state, job_ids,
summary_sha256 and `JobID|State|ExitCode` Slurm accounting. Independently run
`scripts/summarize_smash_full_transfer.py`; check exact header/alias membership,
all per-file output hashes, counts and checkpoint/cut provenance as well.
Inspect both generated figures; save QA and update status/results reports.

Smoke predictions have identical observations, identities, labels, targets,
AUC and selected sets to4435668, with sub-micro float32 differences in scores
and energies. Do not claim bitwise parity; see saved diagnostic and protocol.
No fresh holdout, generated-event denominator, SMASH training, new cut fit,
physical spectrum or symmetry-potential sensitivity measurement is available.
