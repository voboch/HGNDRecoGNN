"""Describe the singleton contribution to existing DCM AUC; no fitting or inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.diagnose_student_pipeline import metadata, KEYS  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(inputs, output):
    manifest_path = inputs/'diagnostic_export_manifest_20261021.json'
    manifest = json.loads(manifest_path.read_text())
    paths = ['test_evaluation/test_pack.pt'] + [f'test_evaluation/student_sage_{s}/predictions.csv' for s in [42,123,456]]
    for p in paths:
        if sha(inputs/p) != manifest[p]:
            raise ValueError('Previously evaluated DCM input changed')
    meta = metadata(torch.load(inputs/paths[0], weights_only=True, map_location='cpu'))
    rows = []
    # One hit versus multiple hits follows the student's singleton exclusion.
    # This diagnoses a population difference; it selects no future threshold.
    for seed, path in zip([42,123,456], paths[1:]):
        pred = pd.read_csv(inputs/path)
        f = pred.merge(meta, on=KEYS, suffixes=('', '_pack'), validate='one_to_one')
        assert len(f) == len(meta) == 222251
        assert np.array_equal(f.cl_label, f.cl_label_pack)
        y = f.cl_label.to_numpy() == 1
        score, single = f.cl_score.to_numpy(), f['size'].to_numpy() == 1
        cells = []
        for a in [True,False]:
            positive = score[y & (single == a)]
            for b in [True,False]:
                negative = np.sort(score[~y & (single == b)])
                auc = float(np.mean((np.searchsorted(negative, positive, side='left')
                          + np.searchsorted(negative, positive, side='right'))/(2.*len(negative))))
                weight = len(positive)*len(negative)/(int(y.sum())*int((~y).sum()))
                cells.append(dict(signal_singleton=a, background_singleton=b,
                                  n_signal=len(positive), n_background=len(negative),
                                  pair_weight=weight, pair_auc=auc, auc_deficit_contribution=weight*(1-auc)))
        auc = float(roc_auc_score(y,score))
        assert np.isclose(sum(c['pair_weight']*c['pair_auc'] for c in cells),auc,rtol=0,atol=1e-12)
        rows.append(dict(seed=seed, all_auc=auc, all_n=len(f), all_signal=int(y.sum()),
                         singleton_auc=float(roc_auc_score(y[single],score[single])), singleton_n=int(single.sum()),
                         singleton_signal=int((single & y).sum()),
                         multiple_hit_auc=float(roc_auc_score(y[~single],score[~single])),
                         multiple_hit_n=int((~single).sum()), multiple_hit_signal=int((~single & y).sum()),
                         without_singletons_minus_all_auc=float(roc_auc_score(y[~single],score[~single]))-auc,
                         pair_decomposition=cells))
    result = dict(scope='Post-hoc descriptive population diagnostic on already evaluated DCM source988976; no new inference, tuning or independence claim',
                  uncertainty='sample SD across three fixed DCM training seeds; not MC/source error',
                  inputs={p:sha(inputs/p) for p in paths}, manifest_sha256=sha(manifest_path),
                  script_sha256=sha(Path(__file__)), per_seed=rows,
                  summary={k:dict(mean=float(np.mean([r[k] for r in rows])), sample_sd=float(np.std([r[k] for r in rows],ddof=1)))
                           for k in ['all_auc','singleton_auc','multiple_hit_auc','without_singletons_minus_all_auc']})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs,args.output)
