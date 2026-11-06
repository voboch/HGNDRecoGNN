# Native feature ledger: technical preflight

06.11.2026. **GO** for one CPU job with the reviewed source
`3e514b7191fd56f5c98e346c36ac66036d2b30fd570afdde178a0b323fc3b417`.
This verdict does not authorize a repeated failed job or additional science.

## Progress and scope

Latest science: accepted response diagnostics on 06.11.2026; latest model
jobs completed 04.11.2026. The parent reports today's actual Slurm check:
all 18 segments complete with exit 0:0, queue empty. This is a new finite
scientific reduction rather than infrastructure recovery. No drift blocker.

Reviewed protocol, plan, analysis, Slurm wrapper and source package. The job
reads existing native tensors, constructs no model and computes no scores,
fit, threshold, new observation or graph. S286 membership is the exact
truth-bound even 0/2/4 set. Every B member uses its original TEST fold.

## Checks

- All 286 content IDs and their TEST folds agree with accepted split
  848ebf49. All 572 selected A/B member paths, hashes and byte sizes agree
  with prepare4465983; cluster counts sum to 1,798,225.
- The accepted prepare manifest SHA agrees with the plan, including archive
  SHA and its 6,871,214,080-byte size. The selected members total
  1,706,670,520 bytes. Runtime checks cover whole-archive copy/hash/fstat,
  safe regular-member extraction and every selected member hash/size.
- Candidate key order, native targets and cluster sizes have explicit digest
  assertions against the seed42 CSV evidence. That seed is an identity
  reference; no model scores or score selection enter this reduction.
- Feature order matches accepted `FEATURES`. All non-x fields now require
  identical dtype, shape and C-order bytes. SurfaceHit has a byte-equality
  and binary-value control. Other feature differences remain descriptive.
- Node/cluster pointer mapping, the 16 side/label/size groups, float64
  sums/squares, squared encoding differences and maxima have consistent
  dimensions. Zero-count groups remain represented. Counts are node-weighted;
  no per-hit independence or unique causal attribution is asserted.
- The Slurm argv matches the readers. Caps are rocky/normal, type_d,
  4 CPU, 0 GPU, mem=0, 45 minutes, no-requeue; two workers use one Torch and
  BLAS/OpenMP thread each. Work is node-local; only the compact result is
  sent through the three pinned inherited publisher helpers.
- Existing output is rejected before the failure handler. A failed new run
  preserves traceback and node-local evidence. New per-job destinations and
  receipts protect prior work; no automatic job retry/controller exists.
- All seven new source entries match their checksums and are read-only.
  The tar, plan and reviewed local files match `source_package.json`.
  The earlier 327b94dce8ad package is superseded before submission and remains
  preserved; the verdict applies to the new 3e514b7191fd package.

Two original guard findings were closed before this GO: value-only
`torch.equal` was replaced by dtype/shape/byte checks, and the exception path
can no longer add failure evidence inside an existing completed output.

Monitoring remains the requested Luna/low, fork-none, read-only status task.
The parent must verify the new remote source and actual queue/receipts before
the single submission. Runtime scientific acceptance remains separate.

Reviewed analysis SHA256:
`d613aa138db5c2ad1b2a39c7b32db0725f513aeaec128e470a4d74a52fd304b7`.
Plan SHA256:
`f0c33dab8b2e90ce4563323de19af75ac79650ac998c761a752047e4e38a639a`.
No reviewed code, tests, models, data calculation or cluster command was
executed. The only written artifact is this authorized review.
