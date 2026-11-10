"""BmnRoot FindClusters2 spatial components with a fixed ToF comparison selection."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

RULE_REJECTIONS = ("before_cuts", "front_layers", "single_cell", "invalid_tof",
                   "energy_floor", "energy_ceiling", "selected")


def reconstruct(hits):
    # BmnRoot FindClusters2 uses every touching cell in the 3D lattice.
    # Separate event/arm coordinates by more than one cell and do not
    # add a time adjacency cut to this source-backed spatial variant.
    row = hits.Row.to_numpy(np.int64)
    arm = hits.DetectorId.to_numpy(np.int64)
    position = hits[["LayerId", "RowId", "ColumnId"]].to_numpy(np.int64)
    require_geometry = (
        np.isin(arm, (1, 2)).all() and np.array_equal(arm == 1, hits.fY > 0)
        and ((position >= 1) & (position <= [8, 11, 11])).all()
        and not hits.duplicated(["Row", "DetectorId", "LayerId", "RowId", "ColumnId"]).any()
    )
    if not require_geometry:
        raise ValueError("Unexpected or repeated detector cell")
    xyz = np.column_stack([(2 * row + arm) * 2, position])
    edges = cKDTree(xyz).query_pairs(r=1, p=np.inf, output_type="ndarray")
    adj = csr_matrix((np.ones(len(edges), dtype=bool), edges.T), shape=(len(hits), len(hits)))
    n_clusters, cluster = connected_components(adj, directed=False)
    sizes = np.bincount(cluster, minlength=n_clusters)

    distance = np.linalg.norm(hits[["fX", "fY", "fZ"]].to_numpy() / 100, axis=1)
    time = hits.fTime.to_numpy()
    if (time <= 0).any():
        raise ValueError("Nonpositive time needs an explicit BmnRoot Fill policy")
    speed = distance / (0.299792458 * time)
    order = np.lexsort((hits.Instance.to_numpy(), -speed, cluster))
    _, starts = np.unique(cluster[order], return_index=True)
    head = order[starts]
    beta = speed[head]
    physical = np.isfinite(beta) & (beta > 0) & (beta < 1)
    energy = np.full(n_clusters, np.nan)
    energy[physical] = 0.93955 * (1 / np.sqrt(1 - beta[physical] ** 2) - 1)

    front = np.zeros(n_clusters, dtype=bool)
    # CSV layers 1/2 correspond to physical front layers 0/1 in the talk.
    np.logical_or.at(front, cluster, position[:, 0] <= 2)
    deep_multi = ~front & (sizes >= 2)
    # The published low-energy cut suppresses slow background. The
    # shared upper edge follows the approximate HGND ToF design reach.
    # Physical beta is checked before calculating either working point.
    positive = deep_multi & physical & (energy > .3)
    selected = positive & (energy < 4)
    literal = positive & (beta < .9)
    ledger = np.array([
        n_clusters, front.sum(), (~front & (sizes < 2)).sum(),
        (deep_multi & ~physical).sum(), (deep_multi & physical & ~positive).sum(),
        (positive & ~selected).sum(), selected.sum(),
    ], dtype=np.int64)
    if ledger[1:].sum() != n_clusters:
        raise ValueError("Rule rejection ledger does not conserve clusters")
    return cluster, row[head], energy, selected, literal, ledger
