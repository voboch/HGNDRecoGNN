"""Candidate metadata joins and frozen-threshold accounting fixtures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from HGNDRecoGNN.scripts.diagnose_student_pipeline import metadata, measure


def test_metadata_preserves_local_components_and_both_sides():
    p = {k: torch.tensor(v) for k, v in dict(cluster_ptr=[0, 2, 3, 4],
         node_ptr=[0, 3, 4, 6], row=[10, 10, 20], side=[0, 1, 0],
         cluster=[0, 0, 1, 0, 0, 0], label=[1, 0, 0, 1], energy=[1., 0., 0., 2.]).items()}
    f = metadata(p)
    assert f.ClusterID.tolist() == [0, 1, 0, 0]
    assert f['size'].tolist() == [2, 1, 1, 2]
    assert f.multiplicity.tolist() == [3, 3, 3, 1]
    assert f.cl_istop.tolist() == [0, 0, 1, 0]


def test_threshold_and_energy_populations_are_distinct():
    f = pd.DataFrame(dict(cl_label=[1, 1, 0], cl_score=[.5, .9, .8],
                          e_pred=[1.1, 1.8, 5.], e_true=[1., 2., 0.]))
    r = measure(f, .5)
    assert r['selected_signal'] == 1 and r['selected'] == 2
    assert r['efficiency'] == .5 and r['purity'] == .5
    assert np.isclose(r['bias_GeV'], -.05)
    assert np.isclose(r['selected_bias_GeV'], -.2)
