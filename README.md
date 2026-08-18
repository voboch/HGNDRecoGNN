# HGNDRecoGNN

GNN-based neutron reconstruction for the BM@N **HGND** (High Granularity
Neutron Detector).

The pipeline covers CSV parsing → heterogeneous-graph construction →
model training → per-cluster / per-hit / per-edge prediction export →
physics-analysis primitives (ROC, efficiency, energy resolution) → the
cross-dataset **neutron-yield-vs-Ekin sensitivity study** across symmetry
potentials (`zeroSpot` / `defaultSpot` / `bigSpot` ↔ 0 / 18 / 90 MeV).

The same code paths back both the Jupyter notebooks (for iteration on
Mac/MPS) and the SLURM jobs (for production runs on the CUDA cluster).

---

## Repository layout

```
HGNDRecoGNN/
  device.py                       CUDA→MPS→CPU auto-detection + DeviceMap
  data/graph_dataset.py           HGNDGraphDataset (sharded PyG on disk)
  models/
    __init__.py                   REGISTRY: register()/get()/available()
    default_net.py                v1 hit+cluster Net (registered as net_default)
    hetero_{sage,gat,hgt}.py      vkr26 HeteroGNN variants (SAGE / GATv2 / HGT)
    thesis_baselines.py           thesis DynEdgeConv / SAGE / GATv2 (no gate)
    spectral.py                   spectral-gated Challenger variants
    adapters.py                   LapPE per-cluster helper etc.
  training/
    train.py                      fit(cfg) — orchestrator
    eval.py                       predict(model, loader) → 3 DataFrames
    losses.py                     composite losses (default & hetero-cluster)
    checkpoint.py                 v1 checkpoint format (state_dict + arch)
  analysis/
    metrics.py                    ROC/PR, purity, efficiency, resolution, mult.
    efficiency.py                 ε_n(Ekin) with Wilson intervals
    sensitivity.py                cross-dataset N_n(Ekin) comparison
scripts/
  preprocess.py                   build shards from CSVs
  train.py                        train any registered model
  evaluate.py                     load ckpt, export prediction DataFrames
  sensitivity.py                  compare datasets, write CSVs + plot
  convert_checkpoint.py           migrate v0 full-pickle → v1 checkpoint
  run_local.sh                    Mac/MPS convenience wrapper
slurm/
  preprocess.sbatch               CPU-heavy dataset build
  train.sbatch                    single (model, dataset) job
  sweep.sbatch                    array job over the full grid
notebooks/
  preprocessing_dataloader.ipynb  thin driver — build, train, predict
  model_comparison.ipynb          head-to-head across registered models
  sensitivity_study.ipynb         drives the cross-dataset yield comparison
  results_smash.ipynb             legacy analysis (call analysis/metrics.py)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
# torch — pick the wheel that matches your hardware (CPU/CUDA/MPS)
# from https://pytorch.org/get-started/locally/
pip install torch torchvision
# torch-geometric — see https://pytorch-geometric.readthedocs.io/…/install
pip install torch_geometric pyg_lib torch_scatter torch_sparse \
            torch_cluster torch_spline_conv \
            -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__)")+cpu.html
pip install -r requirements.txt
```

The directory name must be `HGNDRecoGNN` — it is the Python package name.

---

## Data

```
HGNDRecoGNN/
  data/
    smash_xecs_2.87gev_hardSkyrme_defaultSpot/
      <run_id>/
        0001_hits.csv    detector hits
        0001_vacs.csv    MC truth (particles)
        …
    smash_xecs_2.87gev_hardSkyrme_zeroSpot/   (to be transferred/simulated)
    smash_xecs_2.87gev_hardSkyrme_bigSpot/    (to be transferred/simulated)
```

Preprocessing is CPU-only; a full dataset takes several hours on 8 workers.

---

## Local workflow (Mac / MPS / CPU)

```bash
# 1) Build shards (once per dataset).
./scripts/run_local.sh preprocess defaultSpot --num-workers 8

# 2) Train.
./scripts/run_local.sh train    defaultSpot net_default --epochs 20

# 3) Export prediction DataFrames.
./scripts/run_local.sh evaluate defaultSpot net_default

# 4) Cross-dataset sensitivity (defaultSpot alone until others arrive).
./scripts/run_local.sh sensitivity net_default defaultSpot
```

For interactive iteration, open `notebooks/preprocessing_dataloader.ipynb`
(thin driver) or `notebooks/model_comparison.ipynb` (head-to-head across
registered models).

Device is auto-selected in priority order **CUDA → MPS → CPU**. Override
with `--device cuda|mps|cpu`. Models declare which submodules must stay
on CPU via `ModelSpec.default_cpu_pinned` (e.g. `DynamicEdgeConv` on MPS).

---

## Cluster workflow (SLURM)

```bash
# 1) Build shards (CPU, ~4 h wall).
sbatch --export=ALL,DATASET_NAME=defaultSpot,CSV_DIR=/scratch/data/defaultSpot \
       slurm/preprocess.sbatch

# 2) One (model, dataset) training run.
sbatch --export=ALL,DATASET_NAME=defaultSpot,MODEL=net_default \
       slurm/train.sbatch

# 3) Full grid (3 models × 3 datasets = 9 tasks).
sbatch --array=0-8 slurm/sweep.sbatch
```

Adjust `MODELS`/`DATASETS` arrays at the top of `slurm/sweep.sbatch` to
change the grid; the array bound must equal `len(MODELS)*len(DATASETS)-1`.

---

## Registered models

Query at runtime with:

```python
from HGNDRecoGNN import models
print(models.available())
```

| name              | family          | description                                                 |
|-------------------|-----------------|-------------------------------------------------------------|
| `net_default`     | v1 legacy       | hit-branch EdgeConv+SAGE+GraphConv + DynEdgeConv cluster    |
| `hetero_sage`     | vkr26 hetero    | HeteroGNN with SAGEConv per relation                        |
| `hetero_gat`      | vkr26 hetero    | HeteroGNN with GATConv (heads=4) per relation               |
| `hetero_hgt`      | vkr26 hetero    | Heterogeneous Graph Transformer                             |
| `thesis_sage`     | thesis baseline | per-cluster GraphSAGE + scatter-mean pool                   |
| `thesis_gat`      | thesis baseline | per-cluster GATv2 + scatter-mean pool                       |
| `thesis_dynedge`  | thesis baseline | per-cluster DynamicEdgeConv + scatter-mean pool             |
| `spectral_sage`   | thesis spectral | Spectral-Gated GraphSAGE (LapPE gate)                       |
| `spectral_gat`    | thesis spectral | Spectral-Gated GATv2                                        |
| `spectral_dynedge`| thesis spectral | Spectral-Gated DynamicEdgeConv (thesis top-performing)      |

Adding a new model:

```python
# HGNDRecoGNN/models/my_model.py
from . import ModelSpec, register
from ..training.losses import HeteroClusterWeights, hetero_cluster_loss
from ..training.train import forward_hetero

@register('my_model')
def _factory(dataset, hidden: int = 128, **kw):
    model = MyModel(...)
    return model, ModelSpec(
        forward_fn=forward_hetero,
        loss_fn=hetero_cluster_loss,
        default_loss_weights=HeteroClusterWeights(),
        description='what my model does',
    )
```

Then add `from . import my_model` at the bottom of `models/__init__.py`.

---

## Schema versioning

`HGNDGraphDataset` bumps `SCHEMA_V` whenever the on-disk HeteroData
layout changes. Current schema is v2 (adds virtual `events`/`sides`
node types for hetero models). Loading shards with a mismatched schema
version triggers reprocess() rather than silently training on stale data.

Migrate v0 checkpoints (full pickle) once with

```bash
python -m HGNDRecoGNN.scripts.convert_checkpoint \
    notebooks/checkpoints/Hitcl_cluster_surface.pt --arch-name net_default
```

The v1 `.v1.pt` file is written next to the original.

---

## Sensitivity study

Given per-dataset per-cluster prediction pickles, compute

```
N_true_n(Ekin; d) = N_reco_n(Ekin; d) / ε_n(Ekin; d)
```

for each `d ∈ {zeroSpot, defaultSpot, bigSpot}` and compare. Efficiency
carries Wilson binomial errors; solved yield propagates them in quadrature
with Poisson uncertainty on N_reco.

Standalone:

```bash
python -m HGNDRecoGNN.scripts.sensitivity \
    --dataset zeroSpot=results/zeroSpot_net_default/pred_clusters_smash.pkl \
    --dataset defaultSpot=results/defaultSpot_net_default/pred_clusters_smash.pkl \
    --dataset bigSpot=results/bigSpot_net_default/pred_clusters_smash.pkl \
    --out-dir results/sensitivity --threshold 0.5
```

Interactively: `notebooks/sensitivity_study.ipynb`.
