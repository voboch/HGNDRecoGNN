"""Training utilities extracted from `notebooks/preprocessing_dataloader.ipynb`.

The notebook now imports from this package; the same code paths back the
`scripts/train.py` CLI so notebook and SLURM runs stay in lockstep.
"""

from .losses import (
    LossWeights, HeteroClusterWeights,
    default_net_loss, hetero_cluster_loss,
)
from .checkpoint import save_checkpoint, load_checkpoint
from .train import (
    TrainConfig, fit, train_epoch, eval_epoch, reindex_clusters,
    forward_hetero,
)
from .eval import predict

__all__ = [
    'LossWeights', 'HeteroClusterWeights',
    'default_net_loss', 'hetero_cluster_loss',
    'save_checkpoint', 'load_checkpoint',
    'TrainConfig', 'fit', 'train_epoch', 'eval_epoch', 'reindex_clusters',
    'forward_hetero', 'predict',
]
