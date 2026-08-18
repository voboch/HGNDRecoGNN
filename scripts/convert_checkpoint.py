#!/usr/bin/env python
"""One-shot migration of v0 full-pickle checkpoints to the v1 format.

v0 checkpoints were saved as `torch.save(model, path)` — a full-model
pickle that breaks the moment the Net class moves modules or gets
renamed. v1 stores `{state_dict, arch_name, arch_kwargs, ...}`.

Usage:
  python -m HGNDRecoGNN.scripts.convert_checkpoint  path/to/old.pt

By default writes `<old>.v1.pt` alongside the source. Pass `--out` to
override. The original file is untouched.
"""

from __future__ import annotations

import argparse
import os
import sys


def _add_package_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(here, '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert v0 pickle → v1 checkpoint.')
    parser.add_argument('input', help='Path to v0 full-pickle checkpoint.')
    parser.add_argument('--out', default=None,
                        help='Output path. Default: <input>.v1.pt')
    parser.add_argument('--arch-name', default='net_default',
                        help='Architecture name to record. Default: net_default.')
    parser.add_argument('--hidden', type=int, default=512,
                        help='hidden_channels of the source model (for arch_kwargs).')
    parser.add_argument('--num-layers', type=int, default=8,
                        help='num_layers of the source model (for arch_kwargs).')
    args = parser.parse_args()

    _add_package_to_path()

    import torch
    from HGNDRecoGNN.training.checkpoint import save_checkpoint

    src = os.path.abspath(args.input)
    if not os.path.exists(src):
        print(f'error: {src} does not exist', file=sys.stderr)
        return 1

    print(f'Loading v0 pickle from {src} …')
    # Force pickle load — v0 checkpoints ARE pickled model objects.
    model = torch.load(src, map_location='cpu', weights_only=False)
    if not hasattr(model, 'state_dict'):
        print(f'error: {src} is not a torch.nn.Module — '
              f'may already be v1 format', file=sys.stderr)
        return 2

    out = args.out or (src + '.v1.pt')
    save_checkpoint(
        out, model,
        arch_name=args.arch_name,
        arch_kwargs={'hidden_channels': args.hidden,
                     'num_layers': args.num_layers},
        epoch=-1,
        metrics={'migrated_from_v0': 1.0},
    )
    print(f'Wrote v1 checkpoint: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
