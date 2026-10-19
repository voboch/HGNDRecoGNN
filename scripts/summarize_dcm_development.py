"""Export the finite DCM development campaign, including its incomplete runs.

Reads verified small artifacts only. No inference, fitting or job submission.
The predeclared three-seed contrasts remain unavailable if a member is missing.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from HGNDRecoGNN.scripts.plot_b_full import apply_style, assert_contained


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/dcm_development_20261017'
CURRENT = ROOT / 'current'
OUT = ROOT / 'closure'
SEEDS = (42, 123, 456)  # Predeclared paired training seeds, not an optimized set.
RECIPES = ('reference', 'layernorm', 'pool', 'low_lr')
HEADS = ('hit', 'link', 'cluster_cls', 'cluster_energy', 'cluster_link')
COUNTS = {
    'train': dict(zip(HEADS, (591740, 3072406, 159085, 26244, 6416))),
    'validation': dict(zip(HEADS, (362359, 1812177, 101107, 16877, 3690))),
}
INPUTS = {}


def read_bytes(path):
    data = path.read_bytes()
    INPUTS[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
    return data


def read(path):
    return json.loads(read_bytes(path))


def validate():
    proof = read(CURRENT / 'completed_artifacts_verified.json')
    # Freeze the terminal observation used for this export. The live monitor
    # may subsequently record the heartbeat pause without changing this input.
    terminal = OUT / 'terminal_status.json'
    if not terminal.exists():
        terminal.write_bytes((ROOT / 'launch_status.json').read_bytes())
    status = read(terminal)
    frozen = read(CURRENT / 'full_population_verified.json')
    data_hash = hashlib.sha256(read_bytes(CURRENT / 'data_manifest.json')).hexdigest()
    source_hash = hashlib.sha256(read_bytes(CURRENT / 'source_manifest.json')).hexdigest()
    assert data_hash == proof['data_manifest_sha256']
    states = {x.split('|')[0]: x.split('|')[1:] for x in
              status['slurm_accounting'].splitlines()[1:] if '|' in x}
    assert len(states) == 12 and all(x[0] in ('COMPLETED', 'FAILED') for x in states.values())
    assert status['test_evaluation'] is False
    rows, histories, scans = [], {}, {}
    for seed in SEEDS:
        for recipe in RECIPES:
            name = f'{recipe}_{seed}'
            run = CURRENT / name
            task = SEEDS.index(seed) * len(RECIPES) + RECIPES.index(recipe)
            slurm = states[f'4413514_{task}']
            cfg = read(run / 'effective_config.json')
            tc = read(run / 'train_config.json')
            model = read(run / 'model_manifest.json')
            population = read(run / 'population_verified.json')
            h = read(run / 'checkpoints/history.json')
            raw_status = read(run / 'status.json')
            assert cfg['recipe'] == recipe and cfg['seed'] == tc['seed'] == seed
            assert cfg['test_evaluation'] is raw_status['test_opened_by_this_campaign'] is False
            assert cfg['data_manifest_sha256'] == data_hash
            assert cfg['source_manifest_sha256'] == source_hash
            assert cfg['split_manifest_sha256'] == '1ab9285e3315dc0f043b7d5116255f1459978477dc9a9b4d95e046649163278b'
            expected_arch = {'hidden_channels': 512, 'num_layers': 8,
                             'hit_normalization': 'layer' if recipe == 'layernorm' else 'batch',
                             'cluster_pooling': 'mean_max_sum' if recipe == 'pool' else 'mean',
                             'cluster_size_feature': False, 'hit_residual': False}
            assert cfg['architecture'] == tc['arch_kwargs'] == model['architecture'] == expected_arch
            params = 16673477 if recipe == 'pool' else 12479173
            assert model['parameters'] == params and population == frozen
            initial = 1e-4 if recipe == 'low_lr' else 1e-3
            for c in (cfg, tc):
                assert c['epochs'] == 30 and c['batch_size'] == 256
                assert c['lr'] == initial and c['weight_decay'] == 1e-5
                assert c['scheduler'] == 'multistep' and c['scheduler_milestones'] == [10, 15, 25]
                assert c['scheduler_gamma'] == .1 and c['warmup_epochs'] == 0
                assert c['grad_clip_norm'] is None and c['save_best_on'] == 'val'
                assert c['validation_aggregation'] == 'population'
            assert tc['loss_weights']['energy_flatten'] is False
            assert all(tc['loss_weights'][key] == 1 for key in HEADS)
            for i, epoch in enumerate(h):
                assert epoch['epoch'] == i and epoch['validation_aggregation'] == 'population'
                assert math.isclose(epoch['lr_used'], initial * .1 ** sum(i >= k for k in (10, 15, 25)))
                assert math.isclose(epoch['lr_next'], initial * .1 ** sum(i + 1 >= k for k in (10, 15, 25)))
                for split in COUNTS:
                    part = epoch[split]
                    assert part['counts'] == COUNTS[split]
                    assert math.isfinite(part['total']) and all(math.isfinite(v) for v in part['components'].values())
                assert math.isclose(sum(epoch['validation']['components'].values()), epoch['validation']['total'])
            histories[name] = h
            row = {'name': name, 'task': task, 'recipe': recipe, 'training_seed': seed,
                   'state': slurm[0], 'epochs_completed': len(h), 'parameters': params,
                   'validation_roc_auc': None, 'selected_epoch_zero_based': None,
                   'validation_threshold': None, 'validation_working_point': None,
                   'checkpoint_sha256': None, 'scored_on': 'reused validation 988974; development only'}
            if slurm[0] == 'COMPLETED':
                assert slurm[2] == '0:0' and len(h) == 30
                verified = proof['completed_runs'][name]
                for rel, expected in verified['sha256'].items():
                    if rel.endswith(('.pt', '.pkl')):
                        assert not (run / rel).exists(), 'Do not fetch weights or predictions'
                        continue
                    assert hashlib.sha256(read_bytes(run / rel)).hexdigest() == expected
                complete = read(run / 'complete.json')
                perf = read(run / 'validation/performance.json')
                assert complete['test_opened_by_this_campaign'] is False and complete['smoke'] is False
                assert perf['test_opened_by_this_campaign'] is False
                assert perf['n_clusters'] == 101107 and perf['n_signal_clusters'] == 16877
                best = min(range(len(h)), key=lambda i: h[i]['validation']['total'])
                assert complete['selected_epoch'] == verified['selected_epoch_zero_based'] == best
                assert complete['checkpoint_sha256'] == verified['sha256']['checkpoints/model.pt']
                assert perf['roc_auc'] == verified['validation_roc_auc']
                # CSV leaves ratios undefined when no cluster survives. Keep
                # gaps in the plotted curve rather than inventing zero purity.
                scan = [{k: float(v) if v.strip() else float('nan') for k, v in item.items()} for item in
                        csv.DictReader(io.StringIO(read_bytes(run / 'validation/threshold_scan.csv').decode()))]
                assert len(scan) == 401
                for i, point in enumerate(scan):
                    assert math.isclose(point['threshold'], .001 + i * (.999 - .001) / 400, abs_tol=1e-12)
                    assert 0 <= point['n_selected_signal'] <= point['n_selected'] <= 101107
                    if point['n_selected']:
                        assert math.isclose(point['purity_count'], point['n_selected_signal'] / point['n_selected'])
                    else:
                        assert math.isnan(point['purity_count'])
                    assert math.isclose(point['efficiency_count'], point['n_selected_signal'] / 16877, abs_tol=1e-15)
                    if i:
                        assert point['n_selected'] <= scan[i - 1]['n_selected']
                # This inherited target is conventional. The full scan shows
                # the purity/efficiency trade-off; no EoS optimum is implied.
                first = next((x for x in scan if x['purity_count'] >= .70), None)
                if first is None:
                    assert perf['working_threshold'] is None
                else:
                    assert math.isclose(first['threshold'], perf['working_threshold'], abs_tol=1e-12)
                    for key, value in perf['working_point'].items():
                        assert math.isclose(first[key], value, abs_tol=1e-12)
                scans[name] = scan
                row.update(validation_roc_auc=perf['roc_auc'], selected_epoch_zero_based=best,
                           validation_threshold=perf['working_threshold'],
                           validation_working_point=perf['working_point'],
                           checkpoint_sha256=complete['checkpoint_sha256'])
            else:
                assert name in ('pool_42', 'reference_123') and len(h) in (13, 16)
                assert name not in proof['completed_runs'] and not (run / 'complete.json').exists()
                row['failure'] = 'checkpoint I/O EACCES; stale raw status overridden by Slurm'
            rows.append(row)
    assert sum(r['state'] == 'COMPLETED' for r in rows) == len(proof['completed_runs']) == 10
    return rows, histories, scans, status


def contrasts(rows):
    by_name = {r['name']: r for r in rows}
    result = {}
    for candidate in RECIPES[1:]:
        paired, missing = [], []
        for seed in SEEDS:
            a, b = by_name[f'{candidate}_{seed}'], by_name[f'reference_{seed}']
            if a['state'] == b['state'] == 'COMPLETED':
                paired.append({'training_seed': seed,
                               'candidate_minus_reference_auc': a['validation_roc_auc'] - b['validation_roc_auc']})
            else:
                missing.append({'training_seed': seed, 'missing_runs': [r['name'] for r in (a, b) if r['state'] != 'COMPLETED']})
        item = {'role': 'predeclared primary' if candidate == 'layernorm' else 'exploratory secondary',
                'available_descriptive_pairs': paired, 'missing_pairs': missing,
                'required_training_seeds': list(SEEDS), 'difference_mean': None,
                'difference_se': None, 't': None, 'df': None,
                'status': 'unavailable: incomplete predeclared cohort'}
        if not missing:
            values = [p['candidate_minus_reference_auc'] for p in paired]
            mean = statistics.mean(values); se = statistics.stdev(values) / math.sqrt(3)
            item.update(difference_mean=mean, difference_se=se, t=mean/se, df=2,
                        tcrit_two_sided_005=4.303, status='complete three-seed comparison')
        result[candidate] = item
    return result


def finish_plot(fig, path, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, ax in enumerate(np.asarray(axes).flat):
        assert_contained(ax, f'{path.name}:{i}')
        frame = ax.get_window_extent(renderer)
        for annotation in ax.texts:
            box = annotation.get_window_extent(renderer)
            assert frame.x0 <= box.x0 and box.x1 <= frame.x1, (path.name, annotation.get_text())
            assert frame.y0 <= box.y0 and box.y1 <= frame.y1, (path.name, annotation.get_text())
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix('.' + ext), dpi=200)
    plt.close(fig)


def plot_histories(rows, histories):
    folder = OUT / 'histories'; folder.mkdir(exist_ok=True)
    keys = ('population_sum',) + HEADS
    titles = ('Sum of per-head means', 'Hit', 'Hit link', 'Cluster class', 'Signal-cluster energy', 'Cluster link')
    for row in rows:
        h = histories[row['name']]; x = np.arange(len(h))
        fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.8), layout='constrained')
        for ax, key, title in zip(axes.flat, keys, titles):
            for split, colour, ls in (('train', '#2b2b2b', '-'), ('validation', '#1f77b4', '--')):
                # Plot the same population aggregation in both traces. The
                # training gradient objective and logged train.total remain
                # the historical minibatch objective, not this diagnostic sum.
                y = [sum(e[split]['components'].values()) if key == 'population_sum'
                     else e[split]['components'][key] for e in h]
                ax.plot(x, y, color=colour, ls=ls)
            if row['state'] == 'COMPLETED':
                ax.axvline(row['selected_epoch_zero_based'], color='.5', ls=':', lw=.7,
                           ymax=.65 if key == 'population_sum' else 1)
            ax.set(title=title, xlabel='Epoch (zero-based)', ylabel='Loss (dimensionless)')
            values = np.concatenate([line.get_ydata() for line in ax.lines if line.get_transform() is ax.transData])
            if values.min() > 0 and values.max() / values.min() > 50:
                ax.set_yscale('log')
            else:
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.margins(x=.035, y=.10)
        # Reserve headroom for provenance and the legend; neither may cover a
        # loss trajectory, including the isolated reference validation spike.
        top = axes[0, 0]; low, high = top.get_ylim()
        if top.get_yscale() == 'log':
            top.set_ylim(low, math.exp(math.log(high) + .60 * (math.log(high) - math.log(low))))
        else:
            top.set_ylim(low, high + .60 * (high - low))
        top.text(.03, .97, 'DCM simulation; setup unverified\nreco + MC truth; b unavailable', transform=top.transAxes,
                 va='top', fontsize=7)
        top.legend(handles=[Line2D([], [], color='#2b2b2b', label='Train'),
                            Line2D([], [], color='#1f77b4', ls='--', label='Validation')],
                   loc='upper right', bbox_to_anchor=(.98, .80), ncol=2,
                   borderaxespad=0, columnspacing=1)
        state = 'COMPLETE' if row['state'] == 'COMPLETED' else 'INCOMPLETE: checkpoint I/O failure'
        fig.suptitle(f"DCM simulation | reconstructed + MC truth | {row['recipe']}, seed {row['training_seed']}\n"
                     f"{len(h)}/30 epochs; {state} | individual trajectories; uncertainty not estimated", fontsize=8)
        # Collision energy and b settings are not verified for this cache.
        # Do not reuse the primary-truth Xe+Cs energy annotation here.
        finish_plot(fig, folder / row['name'], axes)


def plot_scans(rows, scans):
    folder = OUT / 'thresholds'; folder.mkdir(exist_ok=True)
    by_name = {r['name']: r for r in rows}
    for recipe in RECIPES:
        fig, axes = plt.subplots(2, 3, figsize=(6.84, 4.7), layout='constrained', sharex=True, sharey='row')
        for column, seed in enumerate(SEEDS):
            name = f'{recipe}_{seed}'; row = by_name[name]
            axes[0, column].set_title(f'Training seed {seed}')
            if name not in scans:
                for ax in axes[:, column]:
                    ax.text(.5, .5, 'INCOMPLETE\nNo validation evaluation', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8)
                continue
            scan = scans[name]; x = [v['threshold'] for v in scan]
            for ax, key in zip(axes[:, column], ('purity_count', 'efficiency_count')):
                ax.plot(x, [v[key] for v in scan], color='#1f77b4')
                if row['validation_threshold'] is not None:
                    ax.axvline(row['validation_threshold'], color='#2b2b2b', ls='--', lw=.8,
                               ymax=.80 if key == 'purity_count' else 1)
                else:
                    ax.text(.5, .5, 'Target unreachable', transform=ax.transAxes, ha='center')
                ax.margins(x=.025, y=.06)
                ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
            axes[0, column].axhline(.70, color='.4', ls=':', lw=.8)
            axes[1, column].text(.97, .93, f"t = {row['validation_threshold']:.6f}" if row['validation_threshold'] is not None else 't = null',
                                 transform=axes[1, column].transAxes, ha='right', va='top', fontsize=7)
        for ax in axes[0]: ax.set_ylabel('Count purity (dimensionless)')
        for ax in axes[1]: ax.set(xlabel='Score threshold (dimensionless)', ylabel='Signal efficiency (dimensionless)')
        # The shared purity range includes a label band above all finite scan
        # values; the band is layout space, not an extrapolation of the scan.
        low, high = axes[0, 0].get_ylim()
        axes[0, 0].set_ylim(low, high + .25 * (high - low))
        axes[0, 0].text(.04, .97, 'DCM simulation | reco + MC labels',
                        transform=axes[0, 0].transAxes, va='top', fontsize=7)
        fig.suptitle(f'DCM simulation | reconstructed score + MC truth labels | {recipe}\n'
                     'Reused validation 988974: 101107 clusters, 16877 signal; no uncertainty estimated', fontsize=8)
        finish_plot(fig, folder / recipe, axes)


def write_report(rows, summary):
    lines = ['# DCM development: итог разрешённой серии, 2026-10-17', '',
             'Массив `4413514` завершён: 10 из 12 runs прошли 30 epochs и полную проверку; '
             '`pool_42` и `reference_123` остановились при checkpoint I/O после 16 и 13 '
             'committed epochs. Их partial checkpoints сохранены, но не считаются результатами '
             '30-epoch recipes. Дополнительные recovery jobs не отправлены: требуется ранее '
             'запрошенное явное разрешение пользователя.', '',
             'Это закрытие разрешённого бюджета с неполным cohort. Primary LayerNorm minus '
             'reference по seeds 42/123/456 недоступен из-за reference_123; exploratory pool '
             'и low-LR contrasts также неполны. Mean/SE/t на 2 df оставлены null. Неполные '
             'или непарные cohorts не усреднялись. Доступные отдельные paired differences '
             'сохранены в JSON как descriptive values без статистического вывода.', '',
             '## Population и fitted provenance', '',
             'DCM frozen detector cache `ndet_dataset_dcm_xecs_own_tw1p0`, schema 2. '
             'Фактические collision settings и связь с primary-truth / corrected-detector '
             'generations не установлены. Selection: reconstructable detector clusters; '
             'impact-parameter selection отсутствует, b unavailable. Нового test evaluation '
             'нет; ранее открытый source 988976 не оценивался.', '',
             'Train source 988975: 92252 arm graphs / 64138 physical events / 159085 clusters, '
             '26244 signal. Reused validation 988974: 59973 / 41808 / 101107, 16877 signal. '
             'Scalers fitted только на восстановленных float32 train-hit features; graph '
             'edges, labels, timing noise и indices заморожены. Все arms заново обучались '
             'с population-weighted per-head validation selector. Эти development metrics '
             'не являются independent-test generalisation. Training-seed SE, если cohort '
             'будет восстановлен, не оценит between-source variance.', '',
             '<!-- table-provenance: all metrics scored on reused validation source 988974, '
             '101107 reconstructed clusters/16877 MC signal labels, no b selection (b unavailable); '
             'model weights trained on source 988975, affine scalers fitted on train hits only, '
             'DCM frozen detector cache schema 2; collision settings/generation lineage unverified. -->', '',
             '## Все запланированные seeds', '',
             'Reconstructed cluster ROC AUC, dimensionless, higher is better. Каждая строка — '
             'один training seed; per-row uncertainty не оценена. Threshold dimensionless: '
             'первая из 401 точек [0.001,0.999] с validation count purity >=0.70. Это inherited '
             'convention: более строгий cut меняет purity/efficiency, EoS optimum не определялся. '
             'Недостижимый target сохранялся бы как null. «—» у failed run означает отсутствие '
             'полной evaluation, а не недостижимый target. Epoch выбранного checkpoint zero-based; '
             'число параметров не является quality score.', '',
             '| Recipe | Training seed | Status / epochs | Parameters (count) | AUC ↑ | Selected epoch | Threshold |',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        auc = f"{r['validation_roc_auc']:.5f}" if r['validation_roc_auc'] is not None else '—'
        threshold = f"{r['validation_threshold']:.6f}" if r['validation_threshold'] is not None else ('null' if r['state'] == 'COMPLETED' else '—')
        epoch = str(r['selected_epoch_zero_based']) if r['selected_epoch_zero_based'] is not None else '—'
        lines.append(f"| {r['recipe']} | {r['training_seed']} | {r['state']} / {r['epochs_completed']} | {r['parameters']} | {auc} | {epoch} | {threshold} |")
    lines += ['', '## Что проверено и что остаётся открытым', '',
              'Для всех 10 complete runs проверены 30 sequential histories, finite components, '
              'per-head counts, actual model kwargs/parameter counts, LR used/next, common '
              'split и shard manifest hashes, первый минимум validation composite loss, '
              'selected checkpoint hash и 401-point scans. Все 10 достигли conventional '
              'purity target. Selected epochs лежат от 1 до 10; поздние training losses '
              'не заменяют выбранные validation checkpoints. Это наблюдение не изолирует '
              'причину ранних minima.', '',
              'RA-01–RA-08 и их проверки сохранены в [pipeline audit](pipeline_reaudit_20261017.md). '
              'RA-09 остаётся OPEN: временная недоступность пути / EACCES при сохранении '
              'checkpoint и затем failure status; точная Lustre/node причина не доказана. '
              'Оба failures были на cn-008, но pool_123 там завершился успешно. Slurm status '
              'имеет приоритет над оставшимся `training` в двух raw run-status files.', '',
              'Проверенные coherent atomic resumes и отдельные backups оставлены без изменений. '
              'Proposal pool_42: 14 epochs до total 30; reference_123: 17 epochs до total 30. '
              'Раздельные last/optim files на эпоху впереди atomic resume и не могут быть '
              'смешаны с ним; orphan temporary resume сохранён без promotion. Предложения '
              'возобновления не равны отправленным jobs. Healthy runs не перезапускались.', '',
              'Pooling увеличил capacity с 12479173 до 16673477 parameters при выключенном '
              'size feature; pure fixed-capacity pooling gain этим дизайном не измеряется. '
              'LayerNorm сохраняет affine parameter count. Low-LR меняет только initial '
              'rate на одну декаду при том же 30-epoch budget. Это конечные perturbations, '
              'а не найденные optima. Recipes не менялись после запуска.', '',
              '## Графики и воспроизведение', '',
              'Каждый run имеет шесть panels с суммой population head means и пятью heads. '
              'Train curves — population means измеренных minibatch losses во время изменения '
              'weights; validation curves — epoch-end evaluation на другом source run. '
              'Они являются trajectories двух явно разных populations, а не performance '
              'ratio. Поэтому ratio panel для этих learning diagnostics не строится. '
              'Train population sum в figure вычислен из сохранённых components; исходный '
              'train.total остаётся minibatch-mean objective и не переписан. Vertical dotted '
              'line — selected validation epoch только у complete run. Partial histories '
              'обрываются и подписаны incomplete; никаких extrapolated epochs нет.', '',
              'Threshold plots показывают все 401 points отдельно для каждого training seed; '
              'пустые panels двух failed runs явно обозначены. Dotted horizontal line = '
              'conventional purity 0.70; dashed vertical = первый достигающий target cut. '
              'Изменение cut обычно повышает purity за счёт efficiency, весь trade-off '
              'виден без выбора лучшего seed. Curves используют reconstructed score и '
              'MC truth labels. Error bands/uncertainty не оценены; отсутствие bands '
              'не означает нулевую неопределённость. Source settings не переносятся с '
              'других MC generations.', '',
              'Файлы — `results/dcm_development_20261017/closure/histories/` (12 PDF/PNG), '
              '`closure/thresholds/` (4 PDF/PNG), `closure/summary.json`, `closure/per_seed.csv`. '
              'Каждый PDF векторный с embedded TrueType, PNG 200 dpi; limits derived from '
              'plotted data. Source hashes и визуальная проверка записываются в closure.', '',
              'Воспроизведение из small artifacts: `PYTHONPATH=.. python -m '
              'HGNDRecoGNN.scripts.summarize_dcm_development` из HGNDRecoGNN. Скрипт не '
              'загружает weights/prediction pickles, не выполняет inference и не отправляет jobs.', '',
              'Manuscript сохраняет прежний inconclusive cosine test и archived four-epoch '
              'SMASH result. Development table в статью не добавлена: primary cohort неполон. '
              'Фактический historical minibatch-mean selector уже исправлен в Methods и '
              'проверенном PDF; его эффект отдельно не измерен. Claims о pure scaler '
              'ablation, independent legacy holdout, физическом unfolded spectrum или '
              'neutron-track response не добавлялись.']
    lines += ['', 'Полные recipes и обоснования фиксированных параметров: '
              '[preregistered protocol](dcm_development_protocol_20261017.md). '
              'Thresholds напечатаны до шести decimals, чтобы сохранить исходные grid points.', '',
              '## Просмотр графиков', '', 'Per-head histories (PDF; по шесть panels на run):', '']
    for row in rows:
        lines.append(f"- [{row['name']}: {row['epochs_completed']}/30 epochs](../results/dcm_development_20261017/closure/histories/{row['name']}.pdf)")
    lines += ['', 'Полные threshold scans; каждый panel соответствует одному training seed:', '']
    for recipe in RECIPES:
        lines += [f'![{recipe}: reconstructed score and MC truth labels, reused validation](../results/dcm_development_20261017/closure/thresholds/{recipe}.png)', '']
    (REPO / 'reports/dcm_development_results_20261017.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    apply_style()
    rows, histories, scans, status = validate()
    summary = {'campaign_state': 'closed authorized budget; incomplete cohort',
               'train_array_job': '4413514', 'completed_runs': 10, 'incomplete_runs': 2,
               'source_sha256': status['source_sha256'], 'scored_on': 'reused validation 988974; development only',
               'test_evaluated': False, 'rows': rows, 'contrasts': contrasts(rows),
               'expected_per_head_counts': COUNTS,
               'uncertainty': 'No full paired three-seed SE/t is available; no imputation or unbalanced-cohort means.',
               'recovery_authorization': 'pending explicit human authorization; proposals not submitted'}
    plot_histories(rows, histories)
    plot_scans(rows, scans)
    write_report(rows, summary)
    summary['source_files_sha256'] = INPUTS
    summary['export_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    fields = ['name', 'task', 'recipe', 'training_seed', 'state', 'epochs_completed', 'parameters',
              'validation_roc_auc', 'selected_epoch_zero_based', 'validation_threshold', 'checkpoint_sha256', 'scored_on']
    with (OUT / 'per_seed.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'complete': 10, 'incomplete': 2, 'history_figures': len(histories),
                      'threshold_figures': len(RECIPES), 'contrast_status': {k:v['status'] for k,v in summary['contrasts'].items()}}, indent=2))


if __name__ == '__main__':
    main()
