"""CPU end-to-end: prepare, GPU-smoke logic, training with exact mirror resume, barrier, evaluation, analysis."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pytest
import torch

from HGNDRecoGNN import device, models
from HGNDRecoGNN.data.student_pipeline import NativePackDataset
from HGNDRecoGNN.scripts import smash_train as st
from HGNDRecoGNN.scripts import smash_train_analysis as analysis
from HGNDRecoGNN.scripts import smash_train_prepare as prep
from HGNDRecoGNN.scripts import smash_transfer_infer
from HGNDRecoGNN.scripts.publish_smash_directory import publish
from HGNDRecoGNN.training.checkpoint import load_checkpoint
from HGNDRecoGNN.tests.test_smash_train_fixtures import (
    MEMBERS, fake_dcm, write_audit_bundles, write_raw, write_smoke_split, write_split)

IDENTITY = {'sums_sha256': 'pkg', 'code_digest': 'code', 'files': 0}


def fake_load_dcm(native, seed, dataset, device_name):
    """A fixed, untrained student_sage stands in for the frozen DCM weights."""
    torch.manual_seed(1000 + int(seed))
    model, spec = models.get(st.MODEL, dataset, **st.recipe(st.MODEL, int(seed))['architecture'])
    plan = device.plan_for_spec(model, device_name, spec.default_cpu_pinned)
    device.to_device(model, plan)
    model.eval()
    return model, spec, plan, {'cut': .4, 'checkpoint_sha256': f'fake-{seed}', 'selected_epoch': 0}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(smash_transfer_infer, 'frozen_dcm', lambda args: fake_dcm())
    monkeypatch.setattr(st, 'load_dcm', fake_load_dcm)
    monkeypatch.setattr(st, 'EPOCHS', 2)
    # Two optimizer steps on a toy sample cannot reach the inherited 0.70; the
    # plumbing, not the working point, is under test here.
    monkeypatch.setattr(st, 'PURITY_TARGET', .5)
    # The untrained stand-in DCM has no energy response to invert; fall back to
    # the identity there. The calibration itself is tested in test_smash_train_selection.
    real = st.fit_calibration

    def calibration(e_true, e_pred):
        try:
            return real(e_true, e_pred)
        except ValueError:
            return {'x_e_pred': [1., 2.], 'y_e_true': [1., 2.], 'identity_fallback_in_test': True}
    monkeypatch.setattr(st, 'fit_calibration', calibration)


def run_prepare(split, output, work, audit_index=None, keep=True):
    args = argparse.Namespace(split=split, output=output, work=work, dcm_prepared=None, native_campaign=None,
                              audit_index=audit_index, source_sums=None, keep_observations=keep, workers=1)
    return prep.prepare(args)


def transfer_reference(tmp_path, store, ids):
    """Stand-in for the stored transfer CSVs, written exactly as the transfer did."""
    root = tmp_path / 'transfer'
    pins = {}
    first = NativePackDataset(store.load(ids[0], 'A'))
    for seed in st.SEEDS:
        model, spec, plan, _ = fake_load_dcm(None, seed, first, 'cpu')
        for cid in ids:
            tag, member = cid.split('/', 1)
            frame = st.score_pack(model, spec, plan, store.load(cid, 'A'))
            path = root / tag / f'seed_{seed}' / (member + '.csv')
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, float_format='%.9g')
            pins[f'{tag}/seed_{seed}/{member}.csv'] = prep.sha(path)
    (root / 'pins.json').write_text(json.dumps(pins))
    return root


def test_smoke_checks_pass_on_cpu_and_resume_is_exact(tmp_path, patched):
    raw, hashes = write_raw(tmp_path, events=40)
    index = write_audit_bundles(tmp_path, raw)
    split = write_smoke_split(tmp_path, raw, hashes)
    out = tmp_path / 'output'
    run_prepare(split, out / 'prepare', tmp_path / 'work', index)
    store = st.PackStore.from_directory(out / 'prepare')
    ids = sorted(prep.load_split(split)['contents'])
    reference = transfer_reference(tmp_path, store, ids)
    args = argparse.Namespace(output=out, split=split, prepared=out / 'prepare', transfer_reference=reference,
                              native_campaign=None, device='cpu', mirror=tmp_path / 'mirror', smoke_epochs=2,
                              throughput_batches=3, project_train_files=292, project_val_files=98)
    st.smoke(args, IDENTITY)
    summary = json.loads((out / 'smoke/summary.json').read_text())
    assert summary['all_checks_passed'] and summary['test_opened'] is False
    checks = summary['checks']
    assert checks['observations_bit_identical_to_audit']['files'] == 6
    assert checks['a0_parity_with_transfer']['comparisons'] == 18
    assert checks['a0_parity_with_transfer']['csv_bytes_identical'] == 18
    # On CPU the interrupted-and-resumed run reproduces the uninterrupted weights exactly.
    assert checks['resume_from_mirror']['final_weights_max_abs_diff_vs_uninterrupted'] == 0
    projection = summary['projection']['from_measured_graphs_per_file']
    assert projection['recommended_time_seconds'] >= 1.5 * projection['projected_run_seconds'] + 1800
    complete = json.loads((out / 'smoke/run/complete.json').read_text())
    assert set(complete['cuts']) == set(st.ARMS) and complete['cuts']['A0'] == .4
    assert complete['cuts']['B0'] == complete['cuts']['B1']
    for e in range(2):
        assert load_checkpoint(str(out / f'smoke/run/checkpoints/epoch_{e:03d}.pt')).epoch == e


def test_training_barrier_evaluation_and_analysis(tmp_path, patched):
    raw, hashes = write_raw(tmp_path, events=40)
    split = write_split(tmp_path, raw, hashes, test=(MEMBERS[2],))
    run_prepare(split, tmp_path / 'prep_src/prepare', tmp_path / 'prep_work', keep=False)
    publish(tmp_path / 'prep_src', tmp_path / 'prep_bundle_work', tmp_path / 'prep_bundle', tmp_path / 'prep_receipt.json')
    manifest_sha = prep.sha(tmp_path / 'prep_bundle/manifest.json')
    smoke = tmp_path / 'smoke_summary.json'
    smoke.write_text(json.dumps({'state': 'complete', 'all_checks_passed': True, 'identity': IDENTITY}))
    common = dict(split=split, prepare_bundle=tmp_path / 'prep_bundle', prepare_manifest_sha256=manifest_sha,
                  native_campaign=None, device='cpu')
    tasks = []
    for t in range(3):
        args = argparse.Namespace(**common, task=t, smoke_summary=smoke, work=tmp_path / f'work_{t}',
                                  output=tmp_path / f'train_{t}', mirror=tmp_path / f'mirror/task_{t}', resume_mirror=None)
        st.train(args, IDENTITY)
        receipt = tmp_path / f'train_receipt_{t}.json'
        publish(args.output, tmp_path / f'train_bundle_work_{t}', tmp_path / f'train_bundles/task_{t}', receipt)
        tasks.append({'task': t, 'receipt': str(receipt), 'receipt_sha256': prep.sha(receipt)})
    index = tmp_path / 'train_index.json'
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    eval_args = argparse.Namespace(**common, fold=0, train_index=index, campaign=campaign,
                                   work=tmp_path / 'eval_work', output=tmp_path / 'eval')
    # Barrier: an incomplete fold never opens its test packs.
    index.write_text(json.dumps({'tasks': tasks[:2]}))
    with pytest.raises(ValueError, match='lacks a completed seed'):
        st.evaluate(eval_args, IDENTITY)
    assert not (campaign / 'evaluation_fold_0.opened').exists() and not (tmp_path / 'eval_work/prepare').exists()
    index.write_text(json.dumps({'tasks': tasks}))
    eval_args.work = tmp_path / 'eval_work_2'  # every job starts from a fresh node-local directory
    st.evaluate(eval_args, IDENTITY)
    summary = json.loads((tmp_path / 'eval/evaluation/summary.json').read_text())
    tests = sorted(prep.load_split(split)['folds'][0]['test'])
    assert summary['test_files'] == tests and summary['test_threshold_tuning'] is False
    for arm in st.ARMS:
        for seed in st.SEEDS:
            assert sorted(summary['predictions'][arm][str(seed)]) == tests
    cid = tests[0]
    read = {arm: pd.read_csv(tmp_path / 'eval/evaluation' / summary['predictions'][arm]['42'][cid]['path'])
            for arm in st.ARMS}
    assert list(read['A0'].columns) == st.PREDICTION_COLUMNS
    assert read['A0'].cl_score.equals(read['A1'].cl_score) and read['B0'].cl_score.equals(read['B1'].cl_score)
    assert read['A0'][['Row', 'ClusterID', 'cl_label', 'e_true']].equals(read['B1'][['Row', 'ClusterID', 'cl_label', 'e_true']])
    complete = json.loads((tmp_path / 'train_0/train/complete.json').read_text())
    expected = st.apply_calibration(complete['calibrations']['B1'], read['B1'].e_pred_raw.to_numpy())
    assert np.allclose(read['B1'].e_pred.to_numpy(), expected, rtol=1e-7)
    assert np.array_equal(read['B0'].e_pred.to_numpy(), read['B0'].e_pred_raw.to_numpy())
    # Each fold's test is opened exactly once, across jobs.
    with pytest.raises(FileExistsError):
        st.evaluate(argparse.Namespace(**{**vars(eval_args), 'output': tmp_path / 'eval2', 'work': tmp_path / 'w2'}),
                    IDENTITY)
    result = analysis.run([tmp_path / 'eval/evaluation'], [tmp_path / f'train_{t}/train' for t in range(3)],
                          tmp_path / 'analysis', transfer_root=None)
    assert set(result['arms']) == set(st.ARMS)
    assert result['selection']['fold_0/seed_42']['e_cls'] in (0, 1)
    figures = {p.name for p in (tmp_path / 'analysis/figures').iterdir()}
    assert {'head_histories.pdf', 'threshold_scans.pdf', 'calibration_response.pdf', 'delta_bias.pdf'} <= figures
