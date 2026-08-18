#!/usr/bin/env python
"""Build the sharded HGND graph dataset from raw CSV files.

Usage:
  python -m HGNDRecoGNN.scripts.preprocess \\
      --csv-dir  data/smash_xecs_2.87gev_hardSkyrme_defaultSpot \\
      --root     notebooks/cache/ndet_dataset_smash_defaultSpot \\
      --num-workers 8

The dataset is idempotent — re-running with the same args reuses cached
parquet + shards. Bump SCHEMA_V in graph_dataset.py to force reprocess.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _add_package_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(here, '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def main() -> int:
    parser = argparse.ArgumentParser(description='Build HGND graph shards.')
    parser.add_argument('--csv-dir', required=True,
                        help='Directory containing run subfolders with *_hits.csv and *_vacs.csv.')
    parser.add_argument('--root', required=True,
                        help='Dataset root — shards go under {root}/processed/.')
    parser.add_argument('--rlocal', type=float, default=3.6,
                        help='Spatial radius for RadiusGraph (LayerId/RowId/ColumnId units).')
    parser.add_argument('--twindow', type=float, default=1.5,
                        help='Temporal radius (ns).')
    parser.add_argument('--shard-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Parallel workers for graph construction. 0 = serial.')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Cap events per half. Useful for smoke tests.')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Delete existing processed/ before starting.')
    args = parser.parse_args()

    _add_package_to_path()
    from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset, SCHEMA_V

    if args.force_rebuild:
        import shutil
        processed = os.path.join(args.root, 'processed')
        if os.path.exists(processed):
            print(f'--force-rebuild: removing {processed}')
            shutil.rmtree(processed)

    t0 = time.time()
    print(f'Building dataset (schema v{SCHEMA_V}) …')
    ds = HGNDGraphDataset(
        root=args.root,
        hits_csv_dir=args.csv_dir,
        rlocal=args.rlocal,
        twindow=args.twindow,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
        max_events=args.max_events,
    )
    print(f'Done — {len(ds)} graphs in {(time.time() - t0):.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
