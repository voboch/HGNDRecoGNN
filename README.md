# HGNDRecoGNN

GNN-based neutron reconstruction for the BM@N **HGND** (High Granularity Neutron Detector).

The repository contains:
- **`data/graph_dataset.py`** — disk-based PyG `Dataset` (`HGNDGraphDataset`) that loads raw CSV simulation files, builds heterogeneous graphs, and serialises them in sharded `.pt` files for efficient training.
- **`notebooks/preprocessing_dataloader.ipynb`** — end-to-end pipeline: CSV → graphs → DataLoader → model training & checkpoint export.
- **`notebooks/results_smash.ipynb`** — post-training evaluation: ROC curves, score distributions, efficiency/purity, energy resolution, neutron multiplicity.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url> HGNDRecoGNN
cd HGNDRecoGNN
```

> **Important:** the directory name must be `HGNDRecoGNN` (it is used as the Python package name).

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install PyTorch

Follow the official instructions at <https://pytorch.org/get-started/locally/> to install the build that matches your hardware (CPU / CUDA / Apple MPS).

Example — CPU-only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install PyTorch Geometric

Follow <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>.

Example — CPU wheels:
```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)"+cpu.html
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Data

Place simulation CSV files under `data/<run_folder>/`:

```
HGNDRecoGNN/
└── data/
    └── <your_dataset_name>/
        ├── <run_id_1>/
        │   ├── 0001_hits.csv
        │   ├── 0001_vacs.csv
        │   └── ...
        └── <run_id_2>/
            └── ...
```

Each `*_hits.csv` file contains detector hits; each `*_vacs.csv` file contains MC-truth particle information.  
Update the `HITS_CSV_DIR` variable in `preprocessing_dataloader.ipynb` to point to your dataset.

---

## Running the notebooks

```bash
cd notebooks
jupyter lab
```

### Workflow

1. **`preprocessing_dataloader.ipynb`**  
   - Parses CSVs, engineers features, builds heterogeneous graphs.  
   - Graphs are serialised to `notebooks/cache/<dataset_name>/processed/shard_*.pt`.  
   - Trains the GNN model; saves checkpoint to `notebooks/checkpoints/`.  
   - Exports per-hit, per-cluster, and per-edge prediction DataFrames to `notebooks/results/`.

2. **`results_smash.ipynb`**  
   - Loads prediction DataFrames from `notebooks/results/`.  
   - Merges with original CSV data via `load_hits()`.  
   - Produces all performance plots (saved to `plots/`).

---

## Graph structure

Each event is represented as a `HeteroData` object:

| Attribute | Shape | Description |
|---|---|---|
| `hits.x` | `(N_hits, 10)` | Scaled hit features (see `FEATURES` in `data/graph_dataset.py`) |
| `hits.y` | `(N_hits,)` | True neutron-hit label (0/1) |
| `hits→hits edge_index` | `(2, E)` | Spatial ∩ temporal neighbourhood edges |
| `clusters.x` | `(N_cl, 1)` | Placeholder cluster features |
| `clusters.y` | `(N_cl²,)` | True cluster–cluster link labels |
| `cluster` | `(N_hits, 1)` | Cluster assignment per hit |
| `cllabel` | `(N_cl,)` | True cluster label |
| `clenergy` | `(N_cl,)` | True cluster energy [GeV] |
| `Row` | scalar | Event index |
| `istop` | scalar | 1 = top half, 0 = bottom half |

---

## Key parameters (`preprocessing_dataloader.ipynb`)

| Variable | Default | Description |
|---|---|---|
| `RLOCAL` | `3.6` | Spatial radius for RadiusGraph (LayerId/RowId/ColumnId space) |
| `TWINDOW` | `1.5` | Temporal radius [ns] |
| `MAX_EVENTS` | `100000` | Events processed per half; set `None` for all |
| `NUM_SHARDS` | `100` | Shards loaded from an existing dataset; set `None` for all |
| `SHARD_SIZE` | `1024` | Graphs per shard file |
| `NUM_WORKERS_PROCESS` | `8` | Parallel workers for graph construction |
| `BATCH_SIZE` | `512` | Training batch size |
