# Handoff → full-statistics HPC sensitivity run

Context: post 2026-08-26 smoke work (per-cluster/per-neutron closure,
paper PR-1 correctness pass). The paper's PRC-target physics
interpretation now rests on smoke-scale numbers (~1600–2000 events
per dataset). Full statistics require preprocessing the two OOD
SMASH datasets on cHARISMa and re-running evaluate + sensitivity on
all three at 200 k events each.

**All jobs must be submitted from cHARISMa `login-02` — I cannot
trigger them from the laptop session.**

---

## 1. Prerequisites (verify before submitting)

Log in and check that the raw data and env are in place:

```bash
ssh charisma
ssh -A -o IdentitiesOnly=no -o StrictHostKeyChecking=no login-02
cd ~/HGNDRecoGNN
git pull
```

Then verify:

```bash
# HPC checkpoint from 2026-08-26 run
ls -lh /scratch/vbocharnikov/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_after4278192/model.pt

# defaultSpot v2 cache (already built, job 4278192)
ls /scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_defaultSpot_v2/processed/meta.json

# Raw data for the two OOD datasets — MUST be present before preprocess
ls /scratch/vbocharnikov/hgnd/data/smash_xecs_2.87gev_hardSkyrme_zeroSpot/    | head
ls /scratch/vbocharnikov/hgnd/data/smash_xecs_2.87gev_hardSkyrme_bigSpot/     | head

# Env pin
conda activate hgnd-env
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
# expect: 2.5.1+cu121 2.6.1
```

If the OOD raw data is not yet on scratch, upload it (see
`HPC_DATA.md`) before running Step 2. If the checkpoint is missing,
see `HANDOFF_hgnd_run_2026-08-26.md` §4 for its provenance.

---

## 2. Preprocess the two OOD datasets (~15–20 min each)

Submit as an array job. Index 0=zeroSpot, 1=defaultSpot, 2=bigSpot;
the sbatch skips defaultSpot if its v2 cache already exists.

```bash
sbatch --array=0,2 slurm/preprocess_all_smash.hpc.sbatch
```

Watch:

```bash
squeue -u $USER
tail -f logs/hgnd-preprocess-v2-smash_*_0.out
```

Expected per-array-task outcome (mirrors job 4278192):
- state COMPLETED, elapsed ~15–20 min, MaxRSS ~75 GB
- `/scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_{zeroSpot,bigSpot}_v2/`
  populated with `meta.json` (schema_version=2, num_graphs=200 000,
  num_shards=196) and 196 `shard_*.pt` files.

If any task fails on the file-descriptor sharing bug, the fix from
`data/graph_dataset.py` should be active (`file_system` strategy).
Verify with `grep 'multiprocessing tensor sharing' logs/hgnd-*.out`.

---

## 3. Run the full-statistics sensitivity study (~30–60 min)

Depends on Step 2 for both OOD caches. Once they land:

```bash
# Basic invocation (uses the train-loss-best HPC checkpoint from 2026-08-26)
sbatch slurm/sensitivity_full.hpc.sbatch

# OR chain to Step 2's array job for hands-off pipeline:
JOBID_PREP=$(sbatch --array=0,2 --parsable slurm/preprocess_all_smash.hpc.sbatch)
sbatch --dependency=afterok:$JOBID_PREP slurm/sensitivity_full.hpc.sbatch

# OR run against the val-loss-best checkpoint from slurm/train_valloss.hpc.sbatch:
sbatch --export=ALL,CKPT=/scratch/vbocharnikov/hgnd/checkpoints/defaultSpot_v2_seed42_20ep_valloss_XXXXX/model.pt \
       slurm/sensitivity_full.hpc.sbatch
```

Watch:

```bash
squeue -u $USER
tail -f logs/hgnd-sensitivity-full_*.out
```

Expected outputs land under
`/scratch/vbocharnikov/hgnd/results/sensitivity_full_hpc_${SLURM_JOB_ID}/`:

```
sensitivity_full_hpc_<JOBID>/
├── defaultSpot_hpc/pred_{hits,clusters,edges}_hpc_full.pkl
├── zeroSpot_hpc/pred_{hits,clusters,edges}_hpc_full.pkl
├── bigSpot_hpc/pred_{hits,clusters,edges}_hpc_full.pkl
├── sensitivity_e_pred/
│   ├── sensitivity_summary.csv         # dual-closure, unfold, all datasets
│   ├── sensitivity_{ds}.csv per dataset
│   ├── sensitivity_unfold_{ds}.csv per dataset
│   └── sensitivity_yield.png
├── sensitivity_e_true/                 # same shape, e_true efficiency basis
└── purity_locked_pi70/
    └── purity_locked_pi70.csv          # per-dataset t*(pi=0.7), C_c, C_nu
```

Per-dataset evaluation: ~5–10 min each on V100 32 GB
(~100 k test graphs each). Sensitivity CLI: ~1 min.
Purity-lock CLI: ~1 min.

---

## 4. Retrieve to laptop

`login-02` is not directly reachable from the laptop (`ssh charisma`
lands on `sms`, per §7 of the original `HANDOFF_hgnd.md`), so a
one-shot `rsync login-02:...` will not work without extra ssh
config. Two reliable options:

**Option A — one-shot two-hop tar** (no ssh-config changes):

```bash
JOB=<JOBID>
mkdir -p ~/Project/BM@N/HGND/HGNDRecoGNN/results
ssh charisma "ssh -A -o IdentitiesOnly=no -o StrictHostKeyChecking=no login-02 \
    'tar czf - -C /scratch/vbocharnikov/hgnd/results sensitivity_full_hpc_$JOB'" \
    > ~/Project/BM@N/HGND/HGNDRecoGNN/results/sensitivity_full_hpc_$JOB.tar.gz
tar xzf ~/Project/BM@N/HGND/HGNDRecoGNN/results/sensitivity_full_hpc_$JOB.tar.gz \
    -C ~/Project/BM@N/HGND/HGNDRecoGNN/results/
```

**Option B — rsync via ProxyJump** (persistent one-line rsync).
Add to the laptop's `~/.ssh/config` (leaves the cluster config
untouched per §7):

```
Host login-02 login-*
    HostName login-02
    User vbocharnikov
    ProxyJump charisma
    ForwardAgent yes
```

Then:

```bash
rsync -av login-02:/scratch/vbocharnikov/hgnd/results/sensitivity_full_hpc_$JOB \
      ~/Project/BM@N/HGND/HGNDRecoGNN/results/
```

The `results/` tree is gitignored (see `.gitignore`), so the CSVs
and pickles stay local. Small artefacts (plots, summary CSVs) can
be copied into `paper/figs/` and committed manually.

---

## 5. Update the paper with full-statistics numbers

Once the summary CSVs land locally, the paper edits are mechanical:

- **Table `tab:closure`** (paper `main.tex:722–741`): replace the
  smoke `k`/`C_c`/`C_nu` values with the full-stats row from
  `sensitivity_full_hpc_<JOB>/sensitivity_e_pred/sensitivity_summary.csv`.
- **Table `tab:purity_locked`** (`main.tex:889–910`): replace with
  the full-stats row from
  `sensitivity_full_hpc_<JOB>/purity_locked_pi70/purity_locked_pi70.csv`.
- **Table `tab:eff_ekin`** (`main.tex:504–516`): regenerate via
  `notebooks/evaluate_hpc_checkpoint.ipynb` cell `cfe7357b`
  pointed at the new full-stats defaultSpot prediction pickle.
- **Fig `fig:yield`** (`main.tex:707–711`): copy
  `sensitivity_full_hpc_<JOB>/sensitivity_e_pred/sensitivity_yield.png`
  → `paper/figs/sensitivity_yield.pdf`.
- **B1 blocker in `paper/critical_review_v3.md`**: recompute the
  k(U_sym) monotonicity at full stats. If k(0)/k(18)/k(90) becomes
  monotonic, the physics headline can be strengthened in Sec 5.4.

Then re-run the paper-critic subagent (`Agent` tool with the
`critical_review_v3.md` template) to produce a v4 review over the
full-statistics numbers.

---

## 6. Optional companion runs

- **Val-loss checkpoint**: `sbatch slurm/train_valloss.hpc.sbatch`
  (see `HANDOFF_hgnd_run_2026-08-26.md` §8 point 2). ~1h40m on V100.
  Then re-run Step 3 against the new checkpoint to quantify
  seed/checkpoint-policy contribution to the systematic budget.

- **Model-family sweep**: adapt `slurm/train.hpc.sbatch` with
  `--model {hetero_hgt, hetero_sage, spectral_dynedge}` and the
  same seed. Each run ~1–3 h. Fills Sec 6 items 3–4.

- **Seed variance**: `--seed {123, 456}` sweeps of the val-loss run.
  Fills Sec 6 item 5.

None of these are blockers for the first full-stats sensitivity
paper submission but each removes a `\TODO` in Sec 6.
