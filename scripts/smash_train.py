"""SMASH-trained GraphSAGE on deduplicated corrected-SMASH files, three-fold cross-fitting.

Protocol: reports/smash_train_protocol_20261028.md. One variable separates the
arms, the data that produced the weights; graph view, label, features and the
student_sage recipe are shared and imported, never copied.

  smoke              engineering checks on a small prepared split (GPU, test partition)
  train --task t     fold t//3, seed SEEDS[t%3]; every epoch kept; per-head selection;
                     validation cut and energy calibration frozen in complete.json
  evaluate --fold k  barrier on that fold's three complete.json, then the fold's test
                     packs are opened once; per-file predictions for A0, A1, B0, B1, B-comp

Arms. A0 frozen DCM weights, DCM cut, uncalibrated energy. A1 frozen DCM weights,
cut and calibration from fold validation. B0 SMASH weights (scores from e_cls,
energies from e_en), validation cut, uncalibrated. B1 as B0 plus calibration.
B-comp SMASH weights of the composite-selected epoch, validation cut and calibration.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN import device, models  # noqa: E402
from HGNDRecoGNN.data.student_pipeline import NativePackDataset  # noqa: E402
from HGNDRecoGNN.scripts.diagnose_student_pipeline import measure  # noqa: E402
from HGNDRecoGNN.scripts.reco_performance import E_BINS, threshold_scan  # noqa: E402
from HGNDRecoGNN.scripts.smash_train_prepare import (  # noqa: E402
    PACK_PREFIX, TAGS, clean, extract_members, fit_fold_scalers,
    load_split, same_candidates, sha, write_json)
from HGNDRecoGNN.scripts.scratch_access import ACCESS_RETRIES, with_access_retry  # noqa: E402
from HGNDRecoGNN.scripts.smash_transfer_infer import check_prediction  # noqa: E402
from HGNDRecoGNN.scripts.student_pipeline import check_isolation, recipe  # noqa: E402
from HGNDRecoGNN.scripts.student_reproduce import environment  # noqa: E402
from HGNDRecoGNN.training.losses import HeteroClusterWeights  # noqa: E402
from HGNDRecoGNN.training.checkpoint import load_checkpoint  # noqa: E402
from HGNDRecoGNN.training.eval import predict  # noqa: E402
from HGNDRecoGNN.training.train import (  # noqa: E402
    RESUME_NAME, TrainConfig, epoch_checkpoint_name, fit, seed_everything)

MODEL = 'student_sage'
SEEDS = (42, 123, 456)  # Inherited training seeds of the native and DCM campaigns.
N_FOLDS = 3
EPOCHS = 30  # Inherited budget (student_pipeline recipe); milestones 10/15/25 all fire.
TASKS = [(fold, seed) for fold in range(N_FOLDS) for seed in SEEDS]  # fold-major: t -> (t//3, SEEDS[t%3])
# Inherited comparison convention (student_pipeline.py:146-148), not optimised here:
# the first of 401 cuts reaching count purity 0.70; stricter cuts trade efficiency.
THRESHOLDS = np.linspace(.001, .999, 401)
PURITY_TARGET = .70
# Calibration range: E_BINS (reco_performance.py:24) above the 0.3 GeV native label
# floor, where MC-positive clusters exist. 25 per bin is the inherited minimum of
# energy_performance; a bin below it carries no median.
CAL_BINS = E_BINS[E_BINS >= .3 - 1e-12]
CAL_MIN_COUNT = 25
ARMS = ('A0', 'A1', 'B0', 'B1', 'Bcomp')
PREDICTION_COLUMNS = ['Row', 'ClusterID', 'cl_score', 'e_pred', 'cl_istop', 'cl_label', 'e_true',
                      'cluster_size', 'e_pred_raw']
DEFAULT_NATIVE = Path('/scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206')
NATIVE_SOURCE_SHA = '56bcc48206186e32e182a801410d42865b7f25df656d85166e7a64f67a67fe96'


# ── pure selection, cut and calibration ──────────────────────────────────

def select_epochs(history):
    """Per-head selection on population-aggregated validation losses.

    e_cls minimises validation BCE, e_en the SmoothL1 on MC-positive clusters,
    e_comp the inherited composite (BCE + 0.5 SmoothL1). np.argmin returns the
    first minimum, so the earliest epoch wins a tie.
    """
    if not history or any(h['validation']['aggregation'] != 'population' for h in history):
        raise ValueError('Population validation aggregation was not applied')
    if [h['epoch'] for h in history] != list(range(len(history))):
        raise ValueError('History is not one entry per epoch from 0')
    cls = np.array([h['validation']['components']['cluster_cls'] for h in history], float)
    energy = np.array([h['validation']['components']['cluster_energy'] for h in history], float)
    composite = np.array([h['validation']['total'] for h in history], float)
    if not (np.isfinite(cls).all() and np.isfinite(energy).all() and np.isfinite(composite).all()):
        raise ValueError('Nonfinite validation loss')
    chosen = {'e_cls': int(np.argmin(cls)), 'e_en': int(np.argmin(energy)), 'e_comp': int(np.argmin(composite))}
    last = len(history) - 1
    return {**chosen,
            'validation_cluster_cls': cls.tolist(), 'validation_cluster_energy': energy.tolist(),
            'validation_composite': composite.tolist(),
            # A head minimum at the last epoch means the budget limited it; flag, never extend.
            'budget_limited': {k: v == last for k, v in chosen.items()},
            'e_cls_at_epoch_0': chosen['e_cls'] == 0}


def choose_cut(frame):
    scan = threshold_scan(frame, THRESHOLDS)
    eligible = scan.loc[scan.purity_count >= PURITY_TARGET]
    return (float(eligible.iloc[0].threshold) if len(eligible) else None), scan


def _isotonic(values, weights):
    """Pool-adjacent-violators: weighted nondecreasing fit, returned with its blocks."""
    blocks = []
    for i, (v, w) in enumerate(zip(values, weights)):
        blocks.append([float(v), float(w), [i]])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, i2 = blocks.pop()
            v1, w1, i1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, i1 + i2])
    return blocks


def fit_calibration(e_true, e_pred):
    """Inverse median response, one monotone function for all potentials.

    Per E_true bin of CAL_BINS: the median reconstructed energy m_k and the
    median true energy t_k of MC-positive clusters. The map m -> t is the
    numerical inverse of the median response. A non-increasing m_k (a
    saturating response) is pooled by weighted isotonic regression and the
    pooled bins merge into one point, so the inverse stays single-valued.
    Between points: linear; outside: constant ratio t/m of the end point.
    """
    e_true, e_pred = np.asarray(e_true, float), np.asarray(e_pred, float)
    rows = []
    for lo, hi in zip(CAL_BINS[:-1], CAL_BINS[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() < CAL_MIN_COUNT:
            continue
        rows.append({'e_lo': float(lo), 'e_hi': float(hi), 'n': int(m.sum()),
                     'median_e_true': float(np.median(e_true[m])),
                     'median_e_pred': float(np.median(e_pred[m]))})
    if len(rows) < 2:
        raise ValueError('Fewer than two populated calibration bins')
    blocks = _isotonic([r['median_e_pred'] for r in rows], [r['n'] for r in rows])
    x, y = [], []
    for value, weight, members in blocks:
        x.append(value)
        y.append(sum(rows[i]['median_e_true'] * rows[i]['n'] for i in members) / weight)
    x, y = np.array(x), np.array(y)
    if len(x) < 2 or np.any(np.diff(x) <= 0) or x[0] <= 0 or np.any(np.diff(y) <= 0):
        raise ValueError('Median response cannot be inverted monotonically')
    return {'method': 'inverse median response in E_true bins, PAVA-monotone, linear interpolation, '
                      'constant end ratios', 'bins': rows, 'x_e_pred': x.tolist(), 'y_e_true': y.tolist(),
            'pooled_bins': int(len(rows) - len(x)), 'min_count': CAL_MIN_COUNT, 'edges_GeV': CAL_BINS.tolist()}


def apply_calibration(calibration, e_pred):
    x, y = np.asarray(calibration['x_e_pred']), np.asarray(calibration['y_e_true'])
    e = np.asarray(e_pred, float)
    out = np.interp(e, x, y)
    out = np.where(e < x[0], e * y[0] / x[0], out)
    return np.where(e > x[-1], e * y[-1] / x[-1], out)


# ── source, packs and models ─────────────────────────────────────────────

# Executed outside this process (publication) or by the launcher; part of the code identity.
CODE_EXTRAS = ('HGNDRecoGNN/scripts/smash_train.py', 'HGNDRecoGNN/scripts/smash_train_prepare.py',
               'HGNDRecoGNN/scripts/publish_smash_directory.py', 'HGNDRecoGNN/scripts/smash_input_bundle.py',
               'slurm/smash_train.hpc.sbatch', 'slurm/smash_train_prepare.hpc.sbatch')


def source_identity(sums):
    """Verify every file of the immutable package and digest the code this process runs.

    The code digest covers every package module loaded at this point plus
    CODE_EXTRAS, so a change to inputs/ or to an unrelated script leaves it
    unchanged, while any change to executed code requires a new GPU smoke.
    """
    sums = Path(sums)
    root = sums.parent.resolve()
    entries = {}
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.strip().lstrip('*')
        if sha(sums.parent / name) != digest:
            raise ValueError('Source package file changed: ' + name)
        entries[name] = digest
    if Path(__file__).resolve() != (root / 'HGNDRecoGNN/scripts/smash_train.py').resolve():
        raise ValueError('Running code is not the verified package')
    loaded = set()
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path or not Path(path).is_absolute() or not Path(path).is_file():
            continue  # frozen, namespace or synthetic modules carry no file of ours
        if root in Path(path).resolve().parents:
            loaded.add(Path(path).resolve().relative_to(root).as_posix())
    names = sorted(loaded | set(CODE_EXTRAS))
    unverified = [n for n in names if n not in entries]
    if unverified:
        raise ValueError(f'Executed code outside the verified package: {unverified[:5]}')
    import hashlib
    text = ''.join(f'{entries[n]}  {n}\n' for n in names).encode()
    return {'sums_sha256': sha(sums), 'code_digest': hashlib.sha256(text).hexdigest(), 'files': len(entries),
            'code_files': names}


def concat_packs(packs):
    """One pack from per-file packs; edges and cluster ids are graph-local already."""
    out = {k: torch.cat([p[k] for p in packs]) for k in ('x', 'pe', 'cluster', 'label', 'energy', 'row', 'side', 'index')}
    out['edges'] = torch.cat([p['edges'] for p in packs], dim=1)
    for key in ('node_ptr', 'edge_ptr', 'cluster_ptr'):
        parts, offset = [torch.zeros(1, dtype=torch.long)], 0
        for p in packs:
            parts.append(p[key][1:] + offset)
            offset += int(p[key][-1])
        out[key] = torch.cat(parts)
    return out


class PackStore:
    """Hash-verified per-file packs, from a prepare directory or a published bundle."""

    def __init__(self, receipt_path, root):
        self.receipt_path, self.root = Path(receipt_path), Path(root)
        self.receipt = json.loads(self.receipt_path.read_text())

    @classmethod
    def from_directory(cls, directory):
        return cls(Path(directory) / 'summary.json', directory)

    @classmethod
    def from_bundle(cls, bundle, manifest_sha256, work):
        member = f'{PACK_PREFIX}/summary.json'
        path = extract_members(bundle, manifest_sha256, [member], Path(work) / 'receipt')[member]
        store = cls(path, Path(work) / 'packs' / PACK_PREFIX)
        store.bundle, store.manifest_sha256, store.work = Path(bundle), manifest_sha256, Path(work) / 'packs'
        return store  # the archive was hashed by the receipt extraction above

    def key(self, cid, arm, fold=None):
        return 'A' if arm == 'A' else f'B/fold_{fold}'

    def entries(self, cids, arm, fold=None):
        out = []
        for cid in cids:
            item = self.receipt['files'][cid]
            if item['empty_after_cuts']:
                continue
            out.append((cid, item['packs'][self.key(cid, arm, fold)]))
        return out

    def fetch(self, cids, arm, fold=None):
        """Extract (bundle mode) only these packs; return {cid: path}."""
        entries = self.entries(cids, arm, fold)
        if hasattr(self, 'bundle'):
            members = [f"{PACK_PREFIX}/{e['path']}" for _, e in entries if not (self.root / e['path']).exists()]
            if members:
                extract_members(self.bundle, self.manifest_sha256, members, self.work, check_archive=False)
        return {cid: self.root / e['path'] for cid, e in entries}

    def load(self, cid, arm, fold=None):
        entry = self.receipt['files'][cid]['packs'][self.key(cid, arm, fold)]
        path = self.root / entry['path']
        if sha(path) != entry['sha256']:
            raise ValueError('Pack changed after prepare: ' + str(path))
        return torch.load(path, weights_only=True, map_location='cpu')


def load_packs(store, cids, arm, fold=None):
    store.fetch(cids, arm, fold)
    return {cid: store.load(cid, arm, fold) for cid, _ in store.entries(cids, arm, fold)}


def build_model(seed, dataset, device_name):
    """student_pipeline.build with an explicit device; the recipe is imported."""
    seed_everything(seed)
    config = recipe(MODEL, seed)
    model, spec = models.get(MODEL, dataset, **config['architecture'])
    plan = device.plan_for_spec(model, device_name, spec.default_cpu_pinned)
    device.to_device(model, plan)
    return config, model, spec, plan


def load_dcm(native, seed, dataset, device_name):
    """Frozen DCM GraphSAGE of one seed; its files sit on /scratch (RA-09 retry)."""
    return with_access_retry(_load_dcm, native, seed, dataset, device_name)


def _load_dcm(native, seed, dataset, device_name):
    folder = Path(native) / f'student_sage_{seed}'
    complete = json.loads((folder / 'complete.json').read_text())
    if complete['config']['source_sha256'] != NATIVE_SOURCE_SHA:
        raise ValueError('DCM campaign source differs')
    if complete['config']['prepared_sha256'] != sha(Path(native) / 'prepared.json'):
        raise ValueError('DCM prepared receipt differs')
    path = folder / 'checkpoints/model.pt'
    if sha(path) != complete['files']['checkpoints/model.pt']:
        raise ValueError('DCM checkpoint changed')
    checkpoint = load_checkpoint(str(path))
    if checkpoint.epoch != complete['selected_epoch'] or checkpoint.arch_name != MODEL:
        raise ValueError('DCM checkpoint is not the selected student_sage epoch')
    model, spec = models.get(MODEL, dataset, **complete['config']['architecture'])
    model.load_state_dict(checkpoint.state_dict)
    plan = device.plan_for_spec(model, device_name, spec.default_cpu_pinned)
    device.to_device(model, plan)
    return model, spec, plan, {'cut': complete['validation_threshold'], 'checkpoint_sha256': sha(path),
                               'selected_epoch': complete['selected_epoch']}


def score_pack(model, spec, plan, pack):
    """Per-file scoring exactly as the frozen transfer: batch 256, file order."""
    loader = DataLoader(NativePackDataset(pack), batch_size=256, shuffle=False)
    frame = predict(model, loader, plan=plan, forward_fn=spec.forward_fn, verbose=False)['clusters']
    check_prediction(pack, frame)
    cluster = pack['cluster'].numpy() + np.repeat(pack['cluster_ptr'][:-1].numpy(), np.diff(pack['node_ptr'].numpy()))
    frame['cluster_size'] = np.bincount(cluster, minlength=len(frame))
    return frame


def score_files(model, spec, plan, packs):
    return {cid: score_pack(model, spec, plan, pack) for cid, pack in packs.items()}


def load_epoch(model, checkpoint_dir, epoch, expected_sha=None):
    path = Path(checkpoint_dir) / epoch_checkpoint_name(epoch)
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError('Epoch checkpoint changed: ' + str(path))
    checkpoint = load_checkpoint(str(path))
    if checkpoint.epoch != epoch or checkpoint.arch_name != MODEL:
        raise ValueError('Epoch checkpoint metadata differs: ' + str(path))
    model.load_state_dict(checkpoint.state_dict)
    return model


def combine(score_frames, energy_frames):
    """Scores from one checkpoint, energies from another, on identical candidates."""
    keys = ['Row', 'ClusterID', 'cl_istop', 'cl_label', 'e_true', 'cluster_size']
    out = {}
    for cid, frame in score_frames.items():
        energy = energy_frames[cid]
        if not frame[keys].equals(energy[keys]):
            raise ValueError('Score and energy checkpoints scored different candidates: ' + cid)
        f = frame.copy()
        f['e_pred'] = energy.e_pred.to_numpy()
        out[cid] = f
    return out


def pooled(frames):
    parts = []
    for cid, f in frames.items():
        g = f.copy()
        g['content'] = cid
        g['tag'] = cid.split('/', 1)[0]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def positives(frame):
    m = (frame.cl_label.to_numpy() == 1) & (frame.e_true.to_numpy() > 0)
    return frame.e_true.to_numpy()[m], frame.e_pred.to_numpy()[m]


def arm_metrics(frame, cut):
    result = {'pooled': measure(frame, cut if cut is not None else np.inf)}
    for tag in TAGS:
        part = frame.loc[frame.tag == tag]
        if len(part):
            result[tag] = measure(part, cut if cut is not None else np.inf)
    return clean(result)


# ── mirror and resume ────────────────────────────────────────────────────

class Mirror:
    """Non-fatal copy of each committed epoch to scratch; resume.pt copies are versioned.

    state.json is written last and names the newest complete generation, so a
    copy interrupted mid-epoch leaves the previous generation intact.
    """

    def __init__(self, local, remote, static=()):
        self.local, self.remote, self.failures, self.hashes = Path(local), Path(remote), [], {}
        try:
            self.remote.mkdir(parents=True, exist_ok=True)
            for path in static:
                self._copy(Path(path), self.remote / Path(path).name)
        except Exception as exc:  # noqa: BLE001 - the mirror must never stop training
            self._fail(-1, exc)

    def _copy(self, source, target):
        digest = sha(source)
        tmp = target.with_name(target.name + '.tmp')
        shutil.copyfile(source, tmp)
        if sha(tmp) != digest:
            raise OSError('Mirror copy differs from its source: ' + str(source))
        os.replace(tmp, target)
        return digest

    def _fail(self, epoch, exc):
        self.failures.append({'epoch': epoch, 'error': repr(exc)})
        print(f'WARNING mirror failed at epoch {epoch}: {exc!r}', file=sys.stderr, flush=True)

    def __call__(self, epoch, history):
        try:
            name = epoch_checkpoint_name(epoch)
            self.hashes[name] = self._copy(self.local / name, self.remote / name)
            resume = f'resume_{epoch:03d}.pt'
            self.hashes[resume] = self._copy(self.local / RESUME_NAME, self.remote / resume)
            missing = [epoch_checkpoint_name(e) for e in range(epoch + 1) if epoch_checkpoint_name(e) not in self.hashes]
            if missing:  # an earlier failed copy: retry it before naming this generation
                for item in missing:
                    self.hashes[item] = self._copy(self.local / item, self.remote / item)
            state = {'next_epoch': epoch + 1, 'resume': resume,
                     'sha256': {epoch_checkpoint_name(e): self.hashes[epoch_checkpoint_name(e)] for e in range(epoch + 1)}
                     | {resume: self.hashes[resume]},
                     'job': os.getenv('SLURM_JOB_ID'), 'array_task': os.getenv('SLURM_ARRAY_TASK_ID')}
            write_json(self.remote / 'state.json', state)
        except Exception as exc:  # noqa: BLE001
            self._fail(epoch, exc)


def restore_from_mirror(mirror, checkpoint_dir):
    """Copy the newest complete mirrored generation into a fresh node-local directory."""
    mirror, checkpoint_dir = Path(mirror), Path(checkpoint_dir)
    state = json.loads(with_access_retry((mirror / 'state.json').read_text))
    n = int(state['next_epoch'])
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for name, digest in state['sha256'].items():
        target = checkpoint_dir / (RESUME_NAME if name == state['resume'] else name)
        with_access_retry(shutil.copyfile, mirror / name, target)
        if sha(target) != digest:
            raise ValueError('Mirrored checkpoint differs from its state: ' + name)
    blob = torch.load(checkpoint_dir / RESUME_NAME, map_location='cpu', weights_only=False)
    if int(blob['next_epoch']) != n or len(blob['history']) != n:
        raise ValueError('Mirrored resume state disagrees with state.json')
    # model.pt is rewritten only on a composite improvement; restore it from the
    # same epoch's weights so a resumed run ends with the file it would have had.
    shutil.copyfile(checkpoint_dir / epoch_checkpoint_name(int(blob['best_epoch'])), checkpoint_dir / 'model.pt')
    return n, state, sha(mirror / 'state.json')


# ── training ─────────────────────────────────────────────────────────────

def train_config(config, seed, spec, checkpoint_dir, epochs, start_epoch=0):
    return TrainConfig(epochs=epochs, start_epoch=start_epoch, batch_size=config['batch_size'], seed=seed,
                       arch_name=MODEL, arch_kwargs=config['architecture'], lr=config['lr'],
                       weight_decay=config['weight_decay'], loss_weights=spec.default_loss_weights,
                       checkpoint_dir=str(checkpoint_dir), scheduler_milestones=tuple(config['milestones']),
                       scheduler_gamma=config['gamma'], save_best_on='val',
                       validation_aggregation='population', epoch_checkpoints=True, verbose=False)


class _Interrupted(Exception):
    """Smoke-only stand-in for a job killed after an epoch was committed."""


def train_run(out, seed, train_ds, val_ds, device_name, epochs, identity, mirror=None, resume_mirror=None,
              stop_after=None):
    """Train one seed on fixed packs; returns (model, spec, plan, selection, checkpoint hashes, run info).

    `stop_after` (smoke only) ends the run after that epoch's files and mirror are
    committed, as a scheduler kill would, and returns None.
    """
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = out / 'checkpoints'
    start, resumed = 0, None
    if resume_mirror is not None:
        start, state, state_sha = restore_from_mirror(resume_mirror, checkpoints)
        previous = json.loads((Path(resume_mirror) / 'effective_config.json').read_text())
        resumed = {'mirror': str(resume_mirror), 'state_sha256': state_sha, 'next_epoch': start,
                   'interrupted_job': state.get('job'), 'previous_identity': previous.get('identity')}
    config, model, spec, plan = build_model(seed, train_ds, device_name)
    config.update(epochs=epochs, identity=identity, parameters=sum(p.numel() for p in model.parameters()),
                  train_graphs=len(train_ds), validation_graphs=len(val_ds), resumed=resumed)
    if resumed is not None:
        keys = ('model', 'seed', 'architecture', 'epochs', 'batch_size', 'optimizer', 'lr', 'weight_decay',
                'milestones', 'gamma', 'identity', 'train_graphs', 'validation_graphs')
        if any(previous.get(k) != json.loads(json.dumps(clean(config)))[k] for k in keys):
            raise ValueError('Resumed run differs from the interrupted configuration')
    write_json(out / 'effective_config.json', config)
    write_json(out / 'environment.json', environment())
    cfg = train_config(config, seed, spec, checkpoints, epochs, start)
    write_json(out / 'train_config.json', asdict(cfg))
    checkpoints.mkdir(exist_ok=True)
    mirrored = None
    if mirror is not None:
        mirrored = Mirror(checkpoints, mirror, static=(out / 'effective_config.json', out / 'train_config.json'))

    def callback(epoch, history):
        if mirrored is not None:
            mirrored(epoch, history)
        if stop_after is not None and epoch == stop_after:
            raise _Interrupted
    loaders = (DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
               DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False))
    print(f'Training seed {seed} from epoch {start}: {len(train_ds)} train / {len(val_ds)} validation graphs, '
          f"{config['parameters']} parameters", flush=True)
    started = time.monotonic()
    try:
        fit(model, *loaders, cfg, plan, forward_fn=spec.forward_fn, loss_fn=spec.loss_fn, epoch_callback=callback)
    except _Interrupted:
        return None
    seconds = time.monotonic() - started
    resume = torch.load(checkpoints / RESUME_NAME, weights_only=False, map_location='cpu')
    if resume['next_epoch'] != epochs or len(resume['history']) != epochs:
        raise ValueError('Incomplete training budget')
    selection = select_epochs(resume['history'])
    composite = load_checkpoint(str(checkpoints / 'model.pt'))
    if composite.epoch != selection['e_comp'] or resume['best_epoch'] != selection['e_comp']:
        raise ValueError('Composite checkpoint is not the first validation minimum')
    hashes = {}
    for epoch in range(epochs):
        path = checkpoints / epoch_checkpoint_name(epoch)
        if load_checkpoint(str(path)).epoch != epoch:
            raise ValueError('Epoch checkpoint metadata differs: ' + path.name)
        hashes[f'checkpoints/{path.name}'] = sha(path)
    last = load_checkpoint(str(checkpoints / epoch_checkpoint_name(epochs - 1)))
    for key, value in last.state_dict.items():
        if not torch.equal(value, resume['model'][key]):
            raise ValueError('Last epoch checkpoint differs from the atomic resume state')
    write_json(out / 'history.json', resume['history'])
    for name in ('checkpoints/model.pt', f'checkpoints/{RESUME_NAME}', 'history.json'):
        hashes[name] = sha(out / name)
    mirror_record = None if mirrored is None else {'remote': str(mirrored.remote), 'failures': mirrored.failures}
    return model, spec, plan, selection, hashes, {'train_seconds': seconds, 'mirror': mirror_record}


def validate_arms(model, spec, plan, checkpoints, selection, hashes, val_b, val_a, dcm, out):
    """Score fold validation for every arm; choose cuts and calibrations; save evidence."""
    out.mkdir(exist_ok=False)
    frames = {}
    for head in ('e_cls', 'e_en', 'e_comp'):
        name = epoch_checkpoint_name(selection[head])
        load_epoch(model, checkpoints, selection[head], hashes[f'checkpoints/{name}'])
        frames[head] = score_files(model, spec, plan, val_b)
    dcm_model, dcm_spec, dcm_plan, dcm_info = dcm
    arm_frames = {'A': pooled(score_files(dcm_model, dcm_spec, dcm_plan, val_a)),
                  'B': pooled(combine(frames['e_cls'], frames['e_en'])),
                  'Bcomp': pooled(frames['e_comp'])}
    a, b = arm_frames['A'], arm_frames['B']
    keys = ['content', 'Row', 'ClusterID', 'cl_istop', 'cl_label', 'e_true', 'cluster_size']
    if not a[keys].equals(b[keys]):
        raise ValueError('A and B validation candidates differ')
    cuts, calibrations, errors, metrics = {'A0': dcm_info['cut']}, {}, {}, {}
    for arm, frame in arm_frames.items():
        cut, scan = choose_cut(frame)
        cuts['A1' if arm == 'A' else arm] = cut
        scan.to_csv(out / f'threshold_scan_{arm}.csv', index=False, float_format='%.9g')
        cal_name = {'A': 'A1', 'B': 'B1', 'Bcomp': 'Bcomp'}[arm]
        try:
            calibrations[cal_name] = fit_calibration(*positives(frame))
        except ValueError as exc:
            calibrations[cal_name], errors[cal_name] = None, repr(exc)
        np.savez_compressed(out / f'predictions_{arm}.npz', tag=frame.tag.to_numpy().astype('U11'),
                            content=frame.content.to_numpy().astype(str), cl_score=frame.cl_score.to_numpy(np.float32),
                            e_pred=frame.e_pred.to_numpy(np.float32), e_true=frame.e_true.to_numpy(np.float32),
                            cl_label=frame.cl_label.to_numpy(np.int8), cluster_size=frame.cluster_size.to_numpy(np.int32))
    for arm in ARMS:
        frame = arm_frames['A' if arm.startswith('A') else ('Bcomp' if arm == 'Bcomp' else 'B')].copy()
        cal = {'A1': calibrations.get('A1'), 'B1': calibrations.get('B1'), 'Bcomp': calibrations.get('Bcomp')}.get(arm)
        if cal is not None:
            frame['e_pred'] = apply_calibration(cal, frame.e_pred.to_numpy())
        metrics[arm] = arm_metrics(frame, cuts['B'] if arm in ('B0', 'B1') else cuts[arm])
    cuts['B0'] = cuts['B1'] = cuts.pop('B')
    write_json(out / 'cuts_and_calibration.json', {'cuts': cuts, 'calibrations': calibrations, 'errors': errors})
    write_json(out / 'metrics.json', metrics)
    files = {f'validation/{p.name}': sha(p) for p in sorted(out.iterdir())}
    return cuts, calibrations, errors, metrics, files


def task_partitions(split, fold):
    f = split['folds'][fold]
    return sorted(f['train']), sorted(f['validation'])


def run_task(args, identity, store, split, fold, seed, out, epochs, mirror=None, resume_mirror=None, dcm=None):
    train_ids, val_ids = task_partitions(split, fold)
    train_ds = NativePackDataset(concat_packs(list(load_packs(store, train_ids, 'B', fold).values())))
    val_b = load_packs(store, val_ids, 'B', fold)
    val_a = load_packs(store, val_ids, 'A')
    if list(val_a) != list(val_b) or any(not same_candidates(val_a[c], val_b[c]) for c in val_a):
        raise ValueError('A and B validation packs differ in candidates')
    val_ds = NativePackDataset(concat_packs(list(val_b.values())))
    identity = {**identity, 'fold': fold, 'seed': seed, 'train_files': train_ids, 'validation_files': val_ids}
    model, spec, plan, selection, hashes, run_info = train_run(
        out, seed, train_ds, val_ds, args.device, epochs, identity, mirror=mirror, resume_mirror=resume_mirror)
    write_json(out / 'selection.json', selection)
    if dcm is None:
        dcm = load_dcm(args.native_campaign, seed, val_ds, args.device)
    cuts, calibrations, errors, metrics, val_files = validate_arms(
        model, spec, plan, out / 'checkpoints', selection, hashes, val_b, val_a, dcm, out / 'validation')
    evaluable = all(cuts[a] is not None for a in ARMS) and not errors
    receipt = {'state': 'complete', 'evaluable': evaluable, 'fold': fold, 'seed': seed, 'epochs': epochs,
               'model': MODEL, 'identity': identity, 'selection': selection, 'cuts': cuts,
               'calibrations': calibrations, 'calibration_errors': errors, 'validation_metrics': metrics,
               'dcm': dcm[3], 'files': {**hashes, **val_files, 'selection.json': sha(out / 'selection.json'),
                                        'effective_config.json': sha(out / 'effective_config.json')},
               'run': run_info, 'test_opened': False,
               'frozen_before_test': 'cuts and calibrations above are fixed before any test pack of this fold is opened'}
    write_json(out / 'complete.json', receipt)
    return receipt, (model, spec, plan), (train_ds, val_ds, val_a, val_b)


def train(args, identity):
    fold, seed = TASKS[args.task]
    smoke = json.loads(with_access_retry(args.smoke_summary.read_text))
    if smoke.get('state') != 'complete' or not smoke.get('all_checks_passed') \
            or smoke['identity']['code_digest'] != identity['code_digest']:
        raise ValueError('No passed GPU smoke for this exact code')
    split = load_split(args.split)
    store = PackStore.from_bundle(args.prepare_bundle, args.prepare_manifest_sha256, args.work / 'prepare')
    if store.receipt['split_sha256'] != sha(args.split) or store.receipt['state'] != 'complete':
        raise ValueError('Prepared packs belong to another split')
    identity = {**identity, 'split_sha256': sha(args.split), 'prepare_manifest_sha256': args.prepare_manifest_sha256,
                'prepare_receipt_sha256': sha(store.receipt_path),
                'smoke_summary_sha256': sha(args.smoke_summary), 'task': args.task}
    out = args.output / 'train'
    receipt, _, _ = run_task(args, identity, store, split, fold, seed, out, EPOCHS, mirror=args.mirror,
                             resume_mirror=args.resume_mirror)
    if any(receipt['selection']['budget_limited'].values()):
        print('WARNING head validation minimum at the last epoch: budget-limited, not extended', flush=True)
    write_json(args.output / 'summary.json', {'state': 'complete', 'task': args.task, 'fold': fold, 'seed': seed,
                                              'complete_sha256': sha(out / 'complete.json'),
                                              'evaluable': receipt['evaluable'],
                                              'access_retries': list(ACCESS_RETRIES)})


# ── evaluation ───────────────────────────────────────────────────────────

def require_fold(args, fold, work):
    """Barrier BEFORE any test member is read: all three seeds complete and evaluable."""
    index = json.loads(args.train_index.read_text())
    found = {}
    for item in index['tasks']:
        receipt = json.loads(Path(item['receipt']).read_text())
        bundle = Path(receipt['bundle_dir'])
        if receipt['state'] != 'complete' or sha(Path(item['receipt'])) != item['receipt_sha256']:
            raise ValueError('Training publication receipt changed or incomplete')
        t = int(item['task'])
        if TASKS[t][0] != fold:
            continue
        target = work / f'task_{t}'
        paths = extract_members(bundle, receipt['manifest_sha256'], ['train/complete.json'], target)
        complete = json.loads(paths['train/complete.json'].read_text())
        if (complete['fold'], complete['seed']) != TASKS[t] or not complete['evaluable'] or complete['test_opened']:
            raise ValueError(f'Task {t} is not an evaluable completed run of fold {fold}')
        if complete['epochs'] != EPOCHS or complete['identity']['split_sha256'] != sha(args.split) \
                or complete['identity']['prepare_manifest_sha256'] != args.prepare_manifest_sha256:
            raise ValueError(f'Task {t} was trained on other inputs')
        heads = {complete['selection'][h] for h in ('e_cls', 'e_en', 'e_comp')}
        members = [f'train/checkpoints/{epoch_checkpoint_name(e)}' for e in sorted(heads)]
        paths = extract_members(bundle, receipt['manifest_sha256'], members, target)
        for member in members:
            if sha(paths[member]) != complete['files'][member.removeprefix('train/')]:
                raise ValueError('Selected checkpoint differs from complete.json: ' + member)
        found[complete['seed']] = {'task': t, 'complete': complete, 'checkpoints': target / 'train/checkpoints',
                                   'complete_sha256': sha(target / 'train/complete.json')}
    if sorted(found) != sorted(SEEDS):
        raise ValueError(f'Fold {fold} lacks a completed seed: {sorted(found)}')
    return found


def evaluate(args, identity):
    fold = args.fold
    work = args.work
    split = load_split(args.split)
    runs = require_fold(args, fold, work / 'training')
    marker = args.campaign / f'evaluation_fold_{fold}.opened'
    # Each fold's test is opened exactly once, across jobs. EACCES fails the
    # lookup before the file exists, so retrying the exclusive create is safe.
    with with_access_retry(marker.open, 'x') as stream:
        stream.write(json.dumps({'job': os.getenv('SLURM_JOB_ID'), 'time': time.time()}) + '\n')
    out = args.output / 'evaluation'
    out.mkdir(parents=True, exist_ok=False)
    store = PackStore.from_bundle(args.prepare_bundle, args.prepare_manifest_sha256, work / 'prepare')
    if store.receipt['split_sha256'] != sha(args.split):
        raise ValueError('Prepared packs belong to another split')
    test_ids = sorted(split['folds'][fold]['test'])
    test_a, test_b = load_packs(store, test_ids, 'A'), load_packs(store, test_ids, 'B', fold)
    if list(test_a) != list(test_b) or any(not same_candidates(test_a[c], test_b[c]) for c in test_a):
        raise ValueError('A and B test packs differ in candidates')
    any_pack = NativePackDataset(next(iter(test_b.values())))
    records, metrics = {}, {}
    for seed in SEEDS:
        run = runs[seed]
        complete = run['complete']
        dcm_model, dcm_spec, dcm_plan, dcm_info = load_dcm(args.native_campaign, seed, any_pack, args.device)
        if dcm_info['checkpoint_sha256'] != complete['dcm']['checkpoint_sha256']:
            raise ValueError('DCM checkpoint differs from the one validated')
        _, model, spec, plan = build_model(seed, any_pack, args.device)
        heads = {}
        for head in ('e_cls', 'e_en', 'e_comp'):
            epoch = complete['selection'][head]
            name = f'checkpoints/{epoch_checkpoint_name(epoch)}'
            load_epoch(model, run['checkpoints'], epoch, complete['files'][name])
            heads[head] = score_files(model, spec, plan, test_b)
        a_frames = score_files(dcm_model, dcm_spec, dcm_plan, test_a)
        b_frames = combine(heads['e_cls'], heads['e_en'])
        cal = complete['calibrations']
        arms = {'A0': (a_frames, None), 'A1': (a_frames, cal['A1']), 'B0': (b_frames, None),
                'B1': (b_frames, cal['B1']), 'Bcomp': (heads['e_comp'], cal['Bcomp'])}
        for arm, (frames, calibration) in arms.items():
            per_file = {}
            for cid, frame in frames.items():
                f = frame.copy()
                f['e_pred_raw'] = f.e_pred.to_numpy()
                if calibration is not None:
                    f['e_pred'] = apply_calibration(calibration, f.e_pred_raw.to_numpy())
                path = out / 'predictions' / arm / f'seed_{seed}' / (cid + '.csv')
                path.parent.mkdir(parents=True, exist_ok=True)
                f[PREDICTION_COLUMNS].to_csv(path, index=False, float_format='%.9g')
                records.setdefault(arm, {}).setdefault(str(seed), {})[cid] = {
                    'path': str(path.relative_to(out)), 'sha256': sha(path)}
                per_file[cid] = f
            metrics.setdefault(arm, {})[str(seed)] = {'cut': complete['cuts'][arm],
                                                       **arm_metrics(pooled(per_file), complete['cuts'][arm])}
        del model, dcm_model
        torch.cuda.empty_cache()
    summary = {'state': 'complete', 'fold': fold, 'test_files': test_ids,
               'empty_after_cuts': [c for c in test_ids if store.receipt['files'][c]['empty_after_cuts']],
               'identity': {**identity, 'split_sha256': sha(args.split),
                            'prepare_manifest_sha256': args.prepare_manifest_sha256},
               'training': {str(s): {'task': r['task'], 'complete_sha256': r['complete_sha256'],
                                     'cuts': r['complete']['cuts'], 'selection': {
                                         h: r['complete']['selection'][h] for h in ('e_cls', 'e_en', 'e_comp')}}
                            for s, r in runs.items()},
               'predictions': records, 'metrics': metrics,
               'scope': 'each file scored only by models that never saw it; previously scored by the frozen '
                        'transfer, so not a blind holdout',
               'test_threshold_tuning': False}
    write_json(out / 'summary.json', summary)
    write_json(args.output / 'summary.json', {'state': 'complete', 'fold': fold,
                                              'evaluation_sha256': sha(out / 'summary.json'),
                                              'access_retries': list(ACCESS_RETRIES)})


# ── smoke ────────────────────────────────────────────────────────────────

def compare_to_transfer(frame, reference_path):
    """A0 parity against a stored frozen-transfer CSV: keys exact, floats float32 assert_close."""
    ref = pd.read_csv(reference_path)
    cols = ['Row', 'ClusterID', 'cl_score', 'e_pred', 'cl_istop', 'cl_label', 'e_true', 'cluster_size']
    if list(ref.columns) != cols or len(ref) != len(frame):
        raise ValueError('Stored transfer CSV has another layout: ' + str(reference_path))
    for col in ('Row', 'ClusterID', 'cl_istop', 'cl_label', 'cluster_size'):
        if not np.array_equal(ref[col].to_numpy(), frame[col].to_numpy()):
            raise ValueError(f'A0 candidate key {col} differs from the transfer')
    if not np.array_equal(ref.e_true.to_numpy(np.float32), frame.e_true.to_numpy(np.float32)):
        raise ValueError('A0 true energies differ from the transfer')
    diffs = {}
    for col in ('cl_score', 'e_pred'):
        mine = torch.tensor(frame[col].to_numpy(np.float32))
        theirs = torch.tensor(ref[col].to_numpy(np.float32))
        torch.testing.assert_close(mine, theirs)  # float32 defaults: rtol 1.3e-6, atol 1e-5
        diffs[col] = float((mine - theirs).abs().max())
    rendered = frame[cols].to_csv(index=False, float_format='%.9g').encode()
    import hashlib
    return {'max_abs_diff': diffs, 'csv_bytes_identical': hashlib.sha256(rendered).hexdigest() == sha(reference_path)}


def throughput(model, spec, plan, train_ds, val_ds, n_batches, batch_size=256):
    """Time n training steps and n validation batches, loader included, on a model copy."""
    trial = copy.deepcopy(model)
    optimizer = torch.optim.Adam(trial.parameters(), lr=1e-3, weight_decay=1e-5)
    weights = HeteroClusterWeights()

    def batches(dataset, shuffle):
        while True:
            yield from DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def sync():
        if plan.main.type == 'cuda':
            torch.cuda.synchronize()

    result = {}
    for phase, dataset in (('train', train_ds), ('validation', val_ds)):
        stream = batches(dataset, phase == 'train')
        trial.train(phase == 'train')
        graphs = 0
        for i in range(n_batches + 5):  # five warm-up batches are not timed
            if i == 5:
                sync()
                started = time.perf_counter()
            graph = next(stream).to(plan.main)
            if phase == 'train':
                optimizer.zero_grad()
                loss = spec.loss_fn(spec.forward_fn(trial, graph), graph, weights)['total']
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    spec.loss_fn(spec.forward_fn(trial, graph), graph, weights)
            if i >= 5:
                graphs += int(graph.num_graphs)
        sync()
        seconds = time.perf_counter() - started
        result[phase] = {'batches': n_batches, 'graphs': graphs, 'seconds': seconds,
                         'seconds_per_graph': seconds / graphs}
    if plan.main.type == 'cuda':
        result['max_memory_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
    return result


def project(rates, train_graphs, val_graphs, epochs=30, load_seconds=600):
    """Per-run wall time: epochs x (train + validation) + four validation scoring passes + loading."""
    epoch = train_graphs * rates['train']['seconds_per_graph'] + val_graphs * rates['validation']['seconds_per_graph']
    scoring = 4 * val_graphs * rates['validation']['seconds_per_graph']
    total = epochs * epoch + scoring + load_seconds
    # Inherited margin rule of the plan: 1.5 x projection + 0.5 h, rounded up to 5 min.
    limit = int(np.ceil((1.5 * total + 1800) / 300) * 300)
    return {'train_graphs': train_graphs, 'validation_graphs': val_graphs, 'epoch_seconds': epoch,
            'post_training_scoring_seconds': scoring, 'load_seconds_assumed': load_seconds,
            'projected_run_seconds': total, 'recommended_time_seconds': limit,
            'recommended_time': f'{limit // 3600:02d}:{limit % 3600 // 60:02d}:00'}


def smoke(args, identity):
    started = time.monotonic()
    out = args.output / 'smoke'
    out.mkdir(parents=True, exist_ok=False)
    split = load_split(args.split)
    store = PackStore.from_directory(args.prepared)
    receipt = store.receipt
    if receipt['state'] != 'complete' or receipt['split_sha256'] != sha(args.split):
        raise ValueError('Prepared smoke packs belong to another split')
    checks = {}
    fold = 0
    train_ids, val_ids = task_partitions(split, fold)
    bound = [c for c in train_ids + val_ids if split['contents'][c]['truth_bound']]
    refs = {c: receipt['files'][c]['observation']['audit_reference'] for c in bound}
    checks['observations_bit_identical_to_audit'] = {
        'passed': all(r is not None and r['observation_bit_identical'] for r in refs.values()) and len(refs) == len(bound),
        'files': len(refs), 'file_sha256_identical': sum(bool(r and r['file_sha256_identical']) for r in refs.values())}
    same = {c: same_candidates(store.load(c, 'A'), store.load(c, 'B', fold)) for c in train_ids + val_ids
            if not receipt['files'][c]['empty_after_cuts']}
    checks['a_b_candidates_identical'] = {'passed': all(same.values()), 'files': len(same)}
    # Scaler: refit from the training observations only and require equality.
    observations = {c: args.prepared / receipt['files'][c]['observation']['path'] for c in train_ids}
    geometry = receipt['dcm_inputs']['geometry_bounds']
    refit = fit_fold_scalers(train_ids, observations, geometry)
    stored = receipt['scalers'][f'fold_{fold}']
    equal = all(np.array_equal(np.asarray(stored['sides'][name][k]), getattr(s, k + '_'))
                for name, s in zip(('bot', 'top'), refit) for k in ('mean', 'var', 'scale'))
    checks['scaler_train_only'] = {
        'passed': bool(equal and stored['files'] == sorted(train_ids) and not set(stored['files']) & set(val_ids)),
        'files': stored['files'], 'refit_equal': bool(equal)}
    # Two epochs with epoch checkpoints, mirror, per-head selection and validation freezing.
    dcm_val = load_packs(store, val_ids, 'A')
    first = NativePackDataset(next(iter(dcm_val.values())))
    dcm = load_dcm(args.native_campaign, SEEDS[0], first, args.device)
    mirror = args.mirror / 'uninterrupted'
    task, (model, spec, plan), (train_ds, val_ds, val_a, val_b) = run_task(
        args, identity, store, split, fold, SEEDS[0], out / 'run', args.smoke_epochs, mirror=mirror, dcm=dcm)
    checks['epoch_checkpoints_and_selector'] = {
        'passed': all(f'checkpoints/{epoch_checkpoint_name(e)}' in task['files'] for e in range(args.smoke_epochs))
        and (mirror / 'state.json').exists(),
        'selection': {k: task['selection'][k] for k in ('e_cls', 'e_en', 'e_comp')},
        'cuts': task['cuts'], 'evaluable': task['evaluable'], 'calibration_errors': task['calibration_errors'],
        'mirror_failures': task['run']['mirror']['failures']}
    # Interrupted after one epoch, resumed from the scratch mirror in a fresh directory.
    interrupted = args.mirror / 'interrupted'
    train_run(out / 'interrupted', SEEDS[0], train_ds, val_ds, args.device, args.smoke_epochs, task['identity'],
              mirror=interrupted, stop_after=0)
    model_b, _, _, selection_b, hashes_b, _ = train_run(
        out / 'resumed', SEEDS[0], train_ds, val_ds, args.device, args.smoke_epochs, task['identity'],
        mirror=args.mirror / 'resumed', resume_mirror=interrupted)
    reference = load_checkpoint(str(out / 'run/checkpoints' / epoch_checkpoint_name(args.smoke_epochs - 1))).state_dict
    resumed = load_checkpoint(str(out / 'resumed/checkpoints' / epoch_checkpoint_name(args.smoke_epochs - 1))).state_dict
    diff = max(float((reference[k] - resumed[k]).abs().max()) for k in reference)
    copied = sha(out / 'resumed/checkpoints' / epoch_checkpoint_name(0)) == sha(out / 'interrupted/checkpoints' / epoch_checkpoint_name(0))
    checks['resume_from_mirror'] = {'passed': bool(copied), 'epoch_000_copied_bit_identical': bool(copied),
                                    'final_weights_max_abs_diff_vs_uninterrupted': diff,
                                    'note': 'GPU scatter order is nondeterministic; the difference is reported, not gated'}
    del model_b
    # Candidate isolation and no truth in the forward pass, B model and DCM model.
    check_isolation(model, spec, val_ds, plan)
    check_isolation(dcm[0], dcm[1], NativePackDataset(concat_packs(list(val_a.values()))), dcm[2])
    checks['isolation_and_no_truth_in_forward'] = {'passed': True, 'models': ['B trained', 'A frozen DCM']}
    # A0 parity with the stored frozen-transfer CSVs, every smoke file and DCM seed.
    parity = {}
    all_a = load_packs(store, train_ids + val_ids, 'A')
    pins = json.loads((args.transfer_reference / 'pins.json').read_text())
    for seed in SEEDS:
        dmodel, dspec, dplan, _ = load_dcm(args.native_campaign, seed, first, args.device)
        for cid, pack in all_a.items():
            tag, member = cid.split('/', 1)
            path = args.transfer_reference / tag / f'seed_{seed}' / (member + '.csv')
            if sha(path) != pins[f'{tag}/seed_{seed}/{member}.csv']:
                raise ValueError('Stored transfer CSV differs from its pin: ' + str(path))
            parity[f'{cid}/seed_{seed}'] = compare_to_transfer(score_pack(dmodel, dspec, dplan, pack), path)
        del dmodel
    checks['a0_parity_with_transfer'] = {'passed': True, 'comparisons': len(parity),
                                         'csv_bytes_identical': sum(v['csv_bytes_identical'] for v in parity.values()),
                                         'max_abs_diff': {c: max(v['max_abs_diff'][c] for v in parity.values())
                                                          for c in ('cl_score', 'e_pred')}}
    write_json(out / 'a0_parity.json', parity)
    rates = throughput(model, spec, plan, train_ds, val_ds, args.throughput_batches)
    graphs_per_file = float(np.mean([receipt['files'][c]['population']['graphs'] for c in train_ids + val_ids
                                     if not receipt['files'][c]['empty_after_cuts']]))
    projection = {'measured_smoke_graphs_per_file': graphs_per_file,
                  'from_measured_graphs_per_file': project(rates, args.project_train_files * graphs_per_file,
                                                           args.project_val_files * graphs_per_file),
                  'from_plan_4115_graphs_per_file': project(rates, args.project_train_files * 4115,
                                                            args.project_val_files * 4115),
                  'files_assumed': {'train': args.project_train_files, 'validation': args.project_val_files}}
    checks['throughput'] = {'passed': True, **rates}
    packs = [e['bytes'] for c in train_ids + val_ids if not receipt['files'][c]['empty_after_cuts']
             for e in receipt['files'][c]['packs'].values()]
    projection['pack_bytes_mean'] = float(np.mean(packs))
    projection['prepare_archive_bytes_for_584_files_A_plus_3B'] = float(np.mean(packs) * 584 * 4)
    passed = all(v['passed'] for v in checks.values())
    summary = {'state': 'complete', 'all_checks_passed': passed, 'checks': checks, 'projection': projection,
               'identity': identity, 'prepare_receipt_sha256': sha(args.prepared / 'summary.json'),
               'split_sha256': sha(args.split), 'test_opened': False, 'seconds': time.monotonic() - started,
               'access_retries': list(ACCESS_RETRIES)}
    write_json(out / 'summary.json', summary)
    if not passed:
        raise ValueError('GPU smoke check failed: ' + ', '.join(k for k, v in checks.items() if not v['passed']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['smoke', 'train', 'evaluate'])
    p.add_argument('--source-sums', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Node-local output root (published once)')
    p.add_argument('--work', type=Path, help='Node-local work directory')
    p.add_argument('--native-campaign', type=Path, default=DEFAULT_NATIVE)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--prepared', type=Path, help='smoke: node-local prepare output')
    p.add_argument('--transfer-reference', type=Path, help='smoke: pinned stored transfer CSVs')
    p.add_argument('--smoke-epochs', type=int, default=2)
    p.add_argument('--throughput-batches', type=int, default=200)
    # Projection inputs: half of ~584 unique contents train, one sixth validate (plan section 1/3).
    p.add_argument('--project-train-files', type=int, default=292)
    p.add_argument('--project-val-files', type=int, default=98)
    p.add_argument('--prepare-bundle', type=Path)
    p.add_argument('--prepare-manifest-sha256')
    p.add_argument('--smoke-summary', type=Path, help='train: verified passed GPU smoke summary')
    p.add_argument('--task', type=int, choices=range(len(TASKS)))
    p.add_argument('--mirror', type=Path, help='Scratch mirror for epoch checkpoints')
    p.add_argument('--resume-mirror', type=Path, help='Pre-approved infrastructure retry: resume from this mirror')
    p.add_argument('--fold', type=int, choices=range(N_FOLDS))
    p.add_argument('--train-index', type=Path, help='evaluate: publication receipts of the training tasks')
    p.add_argument('--campaign', type=Path, help='evaluate: scratch campaign root for the once-only marker')
    args = p.parse_args()
    torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '4')))
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('This campaign must execute on an allocated CUDA GPU')
    identity = source_identity(args.source_sums)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == 'smoke':
            smoke(args, identity)
        elif args.mode == 'train':
            train(args, identity)
        else:
            evaluate(args, identity)
    except BaseException as exc:
        write_json(args.output / f"failure_{args.mode}_{os.getenv('SLURM_JOB_ID', 'local')}.json",
                   {'error': repr(exc), 'traceback': traceback.format_exc(), 'identity': identity})
        raise


if __name__ == '__main__':
    main()
