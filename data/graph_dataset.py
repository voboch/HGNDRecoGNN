"""Disk-based PyG Dataset for BM@N neutron-detector heterogeneous graphs.

Graphs are serialised in **shards** (default 1024 graphs per .pt file) to
avoid filesystem overhead from millions of individual files.  Each shard is
a Python list of HeteroData objects.

Graph construction is parallelised across events via multiprocessing.

Usage
-----
>>> from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset
>>> ds = HGNDGraphDataset(root='./processed_graphs',
...                       hits_csv_dir='path/to/ndet_links_Jan_26',
...                       rlocal=3.6, twindow=1.5, num_workers=8)
>>> loader = DataLoader(ds, batch_size=512, shuffle=True)
"""

from __future__ import annotations

import os
import glob
import hashlib
import json
from typing import Optional, Sequence
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Dataset, HeteroData, Data
from torch_geometric.transforms import RadiusGraph
from tqdm import tqdm


# ── Feature list (must match training code) ──────────────────────────────────
FEATURES = [
    'fX', 'fY', 'fZ', 'LayerId', 'RowId', 'ColumnId',
    'fTime', 'fELoss', 'eToF', 'SurfaceHit',
]

# Default number of graphs per shard file.  With ~2M graphs this gives ~2k files
# instead of 2M, while each shard stays < 200 MB in memory.
DEFAULT_SHARD_SIZE = 1024


# ── Standalone helpers (must be picklable → module-level) ─────────────────────

def _dfToF(l: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Time-of-flight kinetic energy estimate."""
    E0 = 939.55          # neutron mass [MeV]
    c  = 299_792_458.0   # speed of light [m/s]
    beta = l / ((t * 1e-9) * c)
    beta = np.clip(beta, 1e-14, 1.0 - 1e-14)
    return np.clip(E0 * (1.0 / np.sqrt(1.0 - beta ** 2) - 1.0), 0.0, 10_000) * 1e-3


def _build_single_graph(args: tuple) -> Optional[HeteroData]:
    """Build one HeteroData graph for a single event.

    Parameters
    ----------
    args : tuple
        (evt_features, evt_meta, rlocal, twindow)
        where:
            evt_features : np.ndarray  – (N, 11) scaled features + SurfaceHit
            evt_meta     : dict        – per-hit arrays: n0_label, Ekin, Id, Row, istop
            rlocal       : float
            twindow      : float
    """
    evt_features, evt_meta, rlocal, twindow = args

    n_hits = evt_features.shape[0]
    if n_hits == 0:
        return None

    _rg_local = RadiusGraph(r=rlocal, loop=True)
    _rg_time  = RadiusGraph(r=twindow, loop=True)

    # ── Node features ─────────────────────────────────────────────────────
    hits_x    = torch.as_tensor(evt_features, dtype=torch.float32)

    # Use raw (unscaled) coordinates for edge construction when available,
    # matching the original reference code that builds RadiusGraph on raw
    # LayerId/RowId/ColumnId and raw fTime.
    if 'pos_local' in evt_meta:
        pos_local = torch.as_tensor(evt_meta['pos_local'], dtype=torch.float32)
    else:
        pos_local = hits_x[:, [3, 4, 5]].clone()
    if 'pos_time' in evt_meta:
        pos_time = torch.as_tensor(evt_meta['pos_time'], dtype=torch.float32)
    else:
        pos_time = hits_x[:, 6:7].clone()

    # ── Edge indices (spatial ∩ temporal) ──────────────────────────────────
    ei_loc = _rg_local(Data(pos=pos_local)).edge_index
    ei_tw  = _rg_time(Data(pos=pos_time)).edge_index

    stride  = n_hits + 1
    enc_loc = ei_loc[0] * stride + ei_loc[1]
    enc_tw  = ei_tw[0]  * stride + ei_tw[1]
    common  = torch.isin(enc_loc, enc_tw)
    edge_index = ei_loc[:, common]

    # ── Clustering (scipy connected components) ───────────────────────────
    n_edges = edge_index.shape[1]
    if n_edges > 0:
        row_idx = edge_index[0].numpy()
        col_idx = edge_index[1].numpy()
        adj = csr_matrix(
            (np.ones(n_edges, dtype=np.float32), (row_idx, col_idx)),
            shape=(n_hits, n_hits),
        )
        _, labels = connected_components(adj, directed=False, return_labels=True)
    else:
        labels = np.arange(n_hits)

    _, clusters_np = np.unique(labels, return_inverse=True)
    clusters   = torch.as_tensor(clusters_np, dtype=torch.long)
    n_clusters = int(clusters.max().item()) + 1

    # ── Cluster-level labels ──────────────────────────────────────────────
    n0_label_np = evt_meta['n0_label']
    ekin_np     = evt_meta['Ekin']
    etarget_np  = ekin_np * n0_label_np

    cllabel_np  = np.zeros(n_clusters, dtype=np.float32)
    np.maximum.at(cllabel_np, clusters_np, n0_label_np)

    clenergy_np = np.zeros(n_clusters, dtype=np.float32)
    np.maximum.at(clenergy_np, clusters_np, etarget_np)

    # ── Cluster-cluster edge index (full bipartite) ───────────────────────
    cl_idx       = torch.arange(n_clusters)
    row_cl       = cl_idx.repeat_interleave(n_clusters)
    col_cl       = cl_idx.repeat(n_clusters)
    edge_index_cl = torch.stack([row_cl, col_cl], dim=0)

    # ── True cluster links ────────────────────────────────────────────────
    cl_best_id = np.zeros(n_clusters, dtype=np.int64)
    id_np  = evt_meta['Id']
    order  = np.argsort(etarget_np)[::-1]
    seen   = np.zeros(n_clusters, dtype=bool)
    for idx in order:
        c = clusters_np[idx]
        if not seen[c]:
            cl_best_id[c] = id_np[idx]
            seen[c] = True
    true_links_cl = cl_best_id[row_cl.numpy()] == cl_best_id[col_cl.numpy()]

    # ── True hit links ────────────────────────────────────────────────────
    true_links = id_np[edge_index[0].numpy()] == id_np[edge_index[1].numpy()]

    # ── Connection mask ───────────────────────────────────────────────────
    cllabel_t = torch.as_tensor(cllabel_np)
    connection_mask = (
        cllabel_t[row_cl].bool()
        & cllabel_t[col_cl].bool()
        & (row_cl != col_cl)
    )

    # ── Assemble HeteroData (CPU only — device transfer at training time) ─
    graph = HeteroData()

    graph['hits'].x   = hits_x
    graph['hits'].pos  = pos_local
    graph['hits'].y    = torch.as_tensor(n0_label_np)
    graph['hits', 'hits'].edge_index = edge_index

    graph['clusters'].x = torch.empty((n_clusters, 1), dtype=torch.float32)
    graph['clusters', 'clusters'].edge_index = edge_index_cl
    graph['clusters'].y = torch.as_tensor(true_links_cl, dtype=torch.float32)
    graph['clusters'].__setitem__('connection_mask', connection_mask)

    graph.__setitem__('true_links', torch.as_tensor(true_links, dtype=torch.float32))
    graph.__setitem__('cluster',    clusters.unsqueeze(1))
    graph.__setitem__('cllabel',    cllabel_t)
    graph.__setitem__('clenergy',   torch.as_tensor(clenergy_np))
    graph.__setitem__('Row',        torch.as_tensor(evt_meta['Row']))
    graph.__setitem__('istop',      torch.as_tensor([evt_meta['istop']]))
    graph.__setitem__('evtlabel',   torch.as_tensor(float(n0_label_np.max())))

    return graph


# ── CSV loading & feature engineering ─────────────────────────────────────────

def _cache_key(datadir: str, runs: Sequence[str]) -> str:
    """Deterministic hash of the CSV directory content for cache invalidation.

    Uses run folder names + file names + sizes (not content — that would be
    too slow for millions of files).  A content-level change that does not
    alter file sizes will *not* invalidate the cache; delete the .parquet
    file manually in that rare case.
    """
    h = hashlib.sha256()
    h.update(os.path.abspath(datadir).encode())
    for run in runs:
        h.update(run.encode())
        run_dir = os.path.join(datadir, run)
        for f in sorted(os.listdir(run_dir)):
            st = os.stat(os.path.join(run_dir, f))
            h.update(f'{f}:{st.st_size}'.encode())
    return h.hexdigest()[:16]


def load_hits(
    datadir: str,
    runs: Sequence[str] | None = None,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load all *_hits.csv files, merge with MC truth, compute features.

    Parameters
    ----------
    datadir : str
        Path to the directory containing run sub-folders with CSV files.
    runs : list[str] | None
        Run sub-folder names.  ``None`` → auto-detect.
    cache_dir : str | None
        Directory to store/load a Parquet cache of the fully-engineered
        DataFrame.  ``None`` → ``datadir`` itself.  The cache file is named
        ``_hits_cache_<hash>.parquet`` where ``<hash>`` encodes the CSV
        directory layout so the cache auto-invalidates when files change.
    """
    if runs is None:
        runs = sorted(
            d for d in os.listdir(datadir)
            if os.path.isdir(os.path.join(datadir, d))
        )

    # ── Parquet cache ─────────────────────────────────────────────────────
    _cdir = cache_dir or datadir
    _ckey = _cache_key(datadir, runs)
    _cache_path = os.path.join(_cdir, f'_hits_cache_{_ckey}.parquet')

    if os.path.exists(_cache_path):
        print(f'Loading cached DataFrame from {_cache_path} …')
        return pd.read_parquet(_cache_path)

    print(f'No cache found — parsing CSVs in {datadir} …')

    # ── hits ──────────────────────────────────────────────────────────────
    dfs = []
    for i, run in enumerate(runs):
        run_dir = os.path.join(datadir, run)
        for f in sorted(os.listdir(run_dir)):
            path = os.path.join(run_dir, f)
            if f.endswith('hits.csv') and os.stat(path).st_size > 100:
                idf = pd.read_csv(
                    path, sep=',', skiprows=[0],
                    names=['Row', 'Instance', 'fX', 'fY', 'fZ',
                           'DetectorId', 'LayerId', 'RowId', 'ColumnId',
                           'fTime', 'fELoss'],
                    skip_blank_lines=True, engine='c',
                )
                idf = idf.dropna(axis=1)
                idf.columns = idf.columns.str.replace(' ', '')
                idf.Row += (int(f.split('_')[0]) - 1) * 500 + i * 200_000
                dfs.append(idf)

    df = pd.concat(dfs, ignore_index=True).dropna(axis=0)
    df = df.astype({
        'Instance': int, 'fX': float, 'fY': float, 'fZ': float,
        'fTime': float, 'fELoss': float,
    }).dropna(axis=0)

    df['fTime_nonoise'] = df['fTime']
    df['fTime'] = df['fTime'] + np.random.normal(0, 0.15, len(df))
    df = df[df.fELoss > 0.003]

    # ── MC particles ──────────────────────────────────────────────────────
    mcp_dfs = []
    for i, run in enumerate(runs):
        run_dir = os.path.join(datadir, run)
        for f in sorted(os.listdir(run_dir)):
            path = os.path.join(run_dir, f)
            if f.endswith('vacs.csv') and os.stat(path).st_size > 100:
                idf = pd.read_csv(
                    path, sep=',', skiprows=[0],
                    names=['Row', 'Instance', 'Id', 'PDG', 'Ekin', 'Rapid',
                           'fMotherId', 'Weight', 'DetectorID', 'Side',
                           'X', 'Y', 'Z', 'vX', 'vY', 'vZ', 'Px', 'Py', 'Pz'],
                    skip_blank_lines=True, engine='c',
                )
                idf = idf.dropna(axis=1)
                idf.columns = idf.columns.str.replace(' ', '')
                idf.Row += (int(f.split('_')[0]) - 1) * 500 + i * 200_000
                mcp_dfs.append(idf)

    mcpdf = pd.concat(mcp_dfs, ignore_index=True)
    mcpdf['PDG'] = np.abs(mcpdf.PDG)

    # ── eToF ──────────────────────────────────────────────────────────────
    dist = np.linalg.norm(np.asarray([df.fX, df.fY, df.fZ]) / 100, axis=0)
    df['eToF']          = _dfToF(dist, df.fTime)
    df['eToF_up_2sig']  = _dfToF(dist, df.fTime - 0.3)
    df['eToF_dn_2sig']  = _dfToF(dist, df.fTime + 0.3)

    # ── merge with MC truth ───────────────────────────────────────────────
    df = pd.merge(
        df,
        mcpdf[['Row', 'Instance', 'Id', 'fMotherId', 'PDG', 'Ekin', 'Side', 'X', 'Y', 'Z']],
        how='left', on=['Row', 'Instance'],
    )
    df = df.astype({'Id': int, 'PDG': int})

    # ── n0_label ──────────────────────────────────────────────────────────
    df['n0_label'] = np.where(
        df.PDG == 2112,
        (df.Ekin > 0.3) & (df.Ekin <= df.eToF_up_2sig) & (df.Ekin >= df.eToF_dn_2sig) & df.Side.isin([5, 6]),
        0,
    )

    # ── derived columns ───────────────────────────────────────────────────
    df['Ncl']      = df.groupby('Row')['Id'].transform('nunique')
    df['Nhits_cl'] = df.groupby(['Row', 'Id']).transform('size')
    df = df.groupby(['Row', 'Instance']).tail(1)

    # ── SurfaceHit flag ───────────────────────────────────────────────────
    sel = (
        (df.RowId == df.RowId.min()) | (df.RowId == df.RowId.max())
        | (df.LayerId == df.LayerId.min()) | (df.LayerId == df.LayerId.max())
        | (df.ColumnId == df.ColumnId.min()) | (df.ColumnId == df.ColumnId.max())
    )
    df['SurfaceHit'] = sel.astype(float)

    # ── Write parquet cache ───────────────────────────────────────────────
    try:
        os.makedirs(_cdir, exist_ok=True)
        df.to_parquet(_cache_path, index=False)
        print(f'Cached DataFrame ({len(df)} rows) → {_cache_path}')
    except Exception as exc:
        print(f'Warning: could not write parquet cache: {exc}')

    return df


def prepare_halves(df: pd.DataFrame):
    """Split into top/bot, drop events with PDG==0, fit scalers.

    Returns
    -------
    list[tuple[pd.DataFrame, StandardScaler, bool]]
        Each element is (half_df, fitted_scaler, istop).
    """
    df_top = df[df.fY > 0].copy()
    df_bot = df[df.fY < 0].copy()

    for half in (df_top, df_bot):
        to_drop = half[half.PDG == 0].Row.unique()
        mask = ~half.Row.isin(to_drop)
        half.drop(half[~mask].index, inplace=True)

    scaler_top = StandardScaler().fit(df_top[FEATURES[:-1]])
    scaler_bot = StandardScaler().fit(df_bot[FEATURES[:-1]])

    return [
        (df_top, scaler_top, True),
        (df_bot, scaler_bot, False),
    ]


def _prepare_events(
    df: pd.DataFrame,
    scaler: StandardScaler,
    istop: bool,
):
    """Yield (features_array, meta_dict) tuples, one per event.

    This is a **generator** to avoid materialising all ~1M events in memory.
    """
    scaled_all = scaler.transform(df[FEATURES[:-1]])
    surf_all   = df['SurfaceHit'].to_numpy(dtype=np.float32)

    for row_id, idx in df.groupby('Row').groups.items():
        idx_arr = idx.to_numpy() if hasattr(idx, 'to_numpy') else np.asarray(idx)
        pos_in_df = df.index.get_indexer(idx_arr)
        x_scaled = scaled_all[pos_in_df]
        x_surf   = surf_all[pos_in_df].reshape(-1, 1)
        feats    = np.hstack([x_scaled, x_surf])

        meta = {
            'n0_label': df.loc[idx_arr, 'n0_label'].to_numpy(dtype=np.float32),
            'Ekin':     df.loc[idx_arr, 'Ekin'].to_numpy(dtype=np.float32),
            'Id':       df.loc[idx_arr, 'Id'].to_numpy(dtype=np.int64),
            'Row':      int(row_id),
            'istop':    istop,
            # Raw (unscaled) coordinates for RadiusGraph edge construction
            'pos_local': df.loc[idx_arr, ['LayerId', 'RowId', 'ColumnId']].to_numpy(dtype=np.float32),
            'pos_time':  df.loc[idx_arr, ['fTime']].to_numpy(dtype=np.float32),
        }
        yield (feats, meta)


def _iter_chunks(iterable, chunk_size: int):
    """Yield successive chunks of *chunk_size* from an iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ── Main Dataset class ────────────────────────────────────────────────────────

class HGNDGraphDataset(Dataset):
    """Disk-based PyG Dataset for BM@N neutron-detector graphs.

    Parameters
    ----------
    root : str
        Root directory.  Processed graphs go into ``root/processed/``.
    hits_csv_dir : str
        Path to the directory containing run sub-folders with CSV files.
    runs : list[str] | None
        Run sub-folder names.  ``None`` → auto-detect.
    rlocal : float
        Spatial radius for RadiusGraph (default 3.6).
    twindow : float
        Temporal radius for RadiusGraph (default 1.5).
    num_workers : int
        Number of parallel workers for graph construction (0 = serial).
    shard_size : int
        Number of graphs per shard file (default 1024).  Reduces file count
        from ~2M to ~2k while keeping each shard < 200 MB.
    device : str | torch.device | None
        If given, each ``get()`` call moves the graph to this device.
    max_events : int | None
        If set, only the first ``max_events`` events **per half** (top/bot)
        are processed.  Useful for quick smoke-tests without rebuilding the
        full dataset.  ``None`` (default) → use all events.
    num_shards : int | None
        If set, only the first ``num_shards`` **already-processed** shard
        files are exposed.  No processing is triggered — the dataset must
        already exist on disk.  Useful for quickly loading a small slice of
        a large pre-processed dataset without re-running ``process()``.
        ``None`` (default) → expose all shards.
    """

    def __init__(
        self,
        root: str,
        hits_csv_dir: str,
        runs: list[str] | None = None,
        rlocal: float = 3.6,
        twindow: float = 1.5,
        num_workers: int = 0,
        shard_size: int = DEFAULT_SHARD_SIZE,
        device: str | torch.device | None = None,
        max_events: int | None = None,
        num_shards: int | None = None,
        transform=None,
        pre_transform=None,
    ):
        self.hits_csv_dir = hits_csv_dir
        self.runs = runs
        self.rlocal = rlocal
        self.twindow = twindow
        self.num_workers = num_workers
        self.shard_size = shard_size
        self._device = torch.device(device) if device is not None else None
        self.max_events = max_events
        self.num_shards = num_shards
        super().__init__(root, transform, pre_transform)

    # ── Metadata / file names ─────────────────────────────────────────────
    @property
    def raw_dir(self) -> str:
        return self.hits_csv_dir

    @property
    def raw_file_names(self) -> list[str]:
        # Existence check is delegated to processed_file_names
        return []

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def _meta_path(self) -> str:
        return os.path.join(self.processed_dir, 'meta.json')

    @property
    def processed_file_names(self) -> list[str]:
        # When a shard cap is requested, check those files directly on disk.
        # This avoids triggering process() when shards exist but meta.json
        # is missing (e.g. after an interrupted build).
        if self.num_shards is not None:
            files = [f'shard_{i}.pt' for i in range(self.num_shards)]
            # Only return the subset that actually exist; PyG won't call
            # process() if all returned names are present on disk.
            existing = [
                f for f in files
                if os.path.exists(os.path.join(self.processed_dir, f))
            ]
            if existing:
                return existing
            # None of the requested shards exist yet — fall through to process()
        if os.path.exists(self._meta_path):
            with open(self._meta_path) as f:
                meta = json.load(f)
            total = meta['num_shards']
            return [f'shard_{i}.pt' for i in range(total)]
        return ['meta.json']      # triggers process()

    @property
    def _progress_path(self) -> str:
        return os.path.join(self.processed_dir, '_progress.json')

    def _save_progress(self, graph_idx: int, shard_idx: int, phase: str):
        """Write a checkpoint so processing can resume after a crash."""
        prog = {'graph_idx': graph_idx, 'shard_idx': shard_idx, 'phase': phase}
        with open(self._progress_path, 'w') as f:
            json.dump(prog, f)

    def _load_progress(self) -> dict | None:
        if os.path.exists(self._progress_path):
            with open(self._progress_path) as f:
                return json.load(f)
        return None

    # ── Processing ────────────────────────────────────────────────────────
    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        # ── Resume from checkpoint if available ───────────────────────────
        progress = self._load_progress()
        if progress is not None:
            graph_idx = progress['graph_idx']
            shard_idx = progress['shard_idx']
            skip_phase = progress['phase']  # 'top' or 'bot'
            print(f'Resuming from checkpoint: {graph_idx} graphs, '
                  f'{shard_idx} shards, last completed phase={skip_phase}')
        else:
            graph_idx = 0
            shard_idx = 0
            skip_phase = None

        print('Loading CSVs …')
        df = load_hits(
            self.hits_csv_dir,
            runs=self.runs,
            cache_dir=self.processed_dir,
        )
        halves = prepare_halves(df)

        for half_df, scaler, istop in halves:
            tag = 'top' if istop else 'bot'

            # Skip already-completed phases on resume
            if skip_phase is not None:
                if tag == 'top' and skip_phase in ('top', 'bot'):
                    print(f'Skipping {tag} (already done)')
                    continue
                if tag == 'bot' and skip_phase == 'bot':
                    print(f'Skipping {tag} (already done)')
                    continue

            n_events = half_df.Row.nunique()
            # Optionally cap the number of events (useful for quick tests)
            if self.max_events is not None and self.max_events < n_events:
                keep_rows = half_df.Row.unique()[:self.max_events]
                half_df = half_df[half_df.Row.isin(keep_rows)]
                n_events = len(keep_rows)
                print(f'max_events={self.max_events}: using first {n_events} events '
                      f'for {tag} half')

            print(f'Building {tag} graphs ({n_events} events, '
                  f'workers={self.num_workers}, '
                  f'shard_size={self.shard_size}) …')

            # Stream events through a generator — never hold all 1M+ in memory
            event_gen = _prepare_events(half_df, scaler, istop)

            # Process one shard_size chunk at a time.
            # Keep a single Pool alive for the whole phase to avoid
            # spawn/teardown overhead per chunk.
            processed_in_phase = 0

            def _process_chunks(builder_fn):
                nonlocal graph_idx, shard_idx, processed_in_phase
                for chunk in _iter_chunks(event_gen, self.shard_size):
                    work = [(f, m, self.rlocal, self.twindow) for f, m in chunk]
                    shard_buf = []
                    for g in builder_fn(work):
                        if g is not None:
                            if self.pre_transform is not None:
                                g = self.pre_transform(g)
                            shard_buf.append(g)
                    del work, chunk

                    if shard_buf:
                        path = os.path.join(self.processed_dir,
                                            f'shard_{shard_idx}.pt')
                        torch.save(shard_buf, path)
                        graph_idx += len(shard_buf)
                        shard_idx += 1
                        processed_in_phase += len(shard_buf)
                        del shard_buf

                        # Checkpoint after every shard
                        self._save_progress(graph_idx, shard_idx, '')
                        print(f'  shard {shard_idx-1}: '
                              f'{processed_in_phase}/{n_events} events  '
                              f'({graph_idx} graphs total)', flush=True)

            if self.num_workers > 0:
                with Pool(self.num_workers, maxtasksperchild=512) as pool:
                    _process_chunks(
                        lambda w: pool.imap(_build_single_graph, w,
                                            chunksize=32))
            else:
                _process_chunks(
                    lambda w: (_build_single_graph(x) for x in w))

            # Save scaler & mark this phase complete
            scaler_path = os.path.join(self.processed_dir, f'scaler_{tag}.pkl')
            pd.to_pickle(scaler, scaler_path)
            self._save_progress(graph_idx, shard_idx, tag)
            print(f'{tag} done — {processed_in_phase} graphs')

        # Write final metadata & remove progress file
        meta = {
            'num_graphs': graph_idx,
            'num_shards': shard_idx,
            'shard_size': self.shard_size,
            'rlocal': self.rlocal,
            'twindow': self.twindow,
            'features': FEATURES,
            'max_events': self.max_events,
        }
        with open(self._meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Clean up checkpoint
        if os.path.exists(self._progress_path):
            os.remove(self._progress_path)

        print(f'Done — {graph_idx} graphs in {shard_idx} shards '
              f'saved to {self.processed_dir}')

    # ── Access ────────────────────────────────────────────────────────────
    def _load_meta(self) -> dict:
        """Read meta.json once and cache it on the instance.

        If meta.json is absent (e.g. interrupted build) the method falls back
        to inferring counts from _progress.json or by globbing shard files on
        disk, so the dataset is still usable for the shards that were written.
        """
        if not hasattr(self, '_meta_cache') or self._meta_cache is None:
            if os.path.exists(self._meta_path):
                with open(self._meta_path) as f:
                    self._meta_cache = json.load(f)
            else:
                # ── Infer from checkpoint or shard files on disk ───────────
                ss = self.shard_size
                inferred_shards = 0
                inferred_graphs = 0

                if os.path.exists(self._progress_path):
                    # _progress.json records the last completed shard count
                    with open(self._progress_path) as f:
                        prog = json.load(f)
                    inferred_shards = prog.get('shard_idx', 0)
                    inferred_graphs = prog.get('graph_idx', 0)
                else:
                    # Count shard files directly
                    shard_files = glob.glob(
                        os.path.join(self.processed_dir, 'shard_*.pt')
                    )
                    inferred_shards = len(shard_files)
                    inferred_graphs = inferred_shards * ss  # estimate

                self._meta_cache = {
                    'num_graphs': inferred_graphs,
                    'num_shards': inferred_shards,
                    'shard_size': ss,
                }
                if inferred_shards > 0:
                    print(
                        f'[HGNDGraphDataset] meta.json not found — inferred '
                        f'{inferred_shards} shards / {inferred_graphs} graphs '
                        f'from disk (incomplete build).'
                    )
        return self._meta_cache

    def _load_shard(self, shard_id: int) -> list[HeteroData]:
        """Load a shard from disk with a small LRU cache (last 4 shards)."""
        if not hasattr(self, '_shard_cache'):
            from collections import OrderedDict
            self._shard_cache: OrderedDict[int, list] = OrderedDict()
        if shard_id in self._shard_cache:
            self._shard_cache.move_to_end(shard_id)
            return self._shard_cache[shard_id]
        path = os.path.join(self.processed_dir, f'shard_{shard_id}.pt')
        shard = torch.load(path, weights_only=False)
        self._shard_cache[shard_id] = shard
        # Keep at most 4 shards in memory (~4 × shard_size graphs)
        while len(self._shard_cache) > 4:
            self._shard_cache.popitem(last=False)
        return shard

    # ── In-memory preloading ──────────────────────────────────────────────
    def preload(self, verbose: bool = True) -> 'HGNDGraphDataset':
        """Load all exposed shards into a flat list in memory.

        After calling this, ``get()`` bypasses shard I/O entirely and
        returns graphs from the pre-loaded list, eliminating the disk
        bottleneck that makes random-access training extremely slow
        (~180 s / batch → < 1 s / batch).

        Memory cost: ~5 MB per shard × N shards (100 shards ≈ 540 MB).

        Returns *self* so you can chain: ``dataset = HGNDGraphDataset(…).preload()``
        """
        import time as _time
        meta = self._load_meta()
        ss = meta.get('shard_size', self.shard_size)
        n_shards_disk = meta['num_shards']
        n_shards = min(n_shards_disk, self.num_shards) if self.num_shards else n_shards_disk

        if verbose:
            print(f'Preloading {n_shards} shards into memory …', end=' ', flush=True)
        t0 = _time.time()

        graphs: list[HeteroData] = []
        for sid in range(n_shards):
            path = os.path.join(self.processed_dir, f'shard_{sid}.pt')
            shard = torch.load(path, weights_only=False)
            graphs.extend(shard)
            del shard

        # Cap to len()
        n = self.len()
        if len(graphs) > n:
            graphs = graphs[:n]

        self._preloaded: list[HeteroData] = graphs
        elapsed = _time.time() - t0
        if verbose:
            mem_mb = sum(
                sum(v.nelement() * v.element_size()
                    for v in g.stores[0].values() if isinstance(v, torch.Tensor))
                for g in self._preloaded
            ) / 1024**2
            print(f'{len(self._preloaded)} graphs loaded in {elapsed:.1f}s '
                  f'(~{mem_mb:.0f} MB tensor data)')

        # Free the LRU shard cache — no longer needed
        if hasattr(self, '_shard_cache'):
            self._shard_cache.clear()

        return self

    @property
    def is_preloaded(self) -> bool:
        return hasattr(self, '_preloaded') and self._preloaded is not None

    def len(self) -> int:
        if self.is_preloaded:
            return len(self._preloaded)
        meta = self._load_meta()
        n = meta['num_graphs']
        ss = meta.get('shard_size', self.shard_size)
        # Cap to requested number of shards
        if self.num_shards is not None:
            n = min(n, self.num_shards * ss)
        # Cap to requested number of events (top + bot)
        if self.max_events is not None:
            n = min(n, self.max_events * 2)
        return n

    def get(self, idx: int) -> HeteroData:
        # ── Fast path: preloaded in memory ─────────────────────────────
        if self.is_preloaded:
            g = self._preloaded[idx].clone()
            if self._device is not None:
                g = g.to(self._device)
            return g

        # ── Slow path: load shard from disk ────────────────────────────
        meta = self._load_meta()
        ss   = meta.get('shard_size', self.shard_size)
        shard_id = idx // ss
        if self.num_shards is not None and shard_id >= self.num_shards:
            raise IndexError(
                f'idx={idx} maps to shard {shard_id}, but num_shards={self.num_shards}'
            )
        offset = idx % ss
        shard  = self._load_shard(shard_id)
        g = shard[offset]
        if self._device is not None:
            g = g.to(self._device)
        return g
