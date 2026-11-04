"""Render the finite no-floor closure from accepted saved reductions."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = json.loads((HERE/'results.json').read_text())
DATA = ROOT/'results/label_ablation_20261103/closure_20261104/extracted'
ACCEPT = json.loads((DATA/'acceptance.json').read_text())
rows = []
for line in ACCEPT['sacct'].splitlines():
    v=line.split('|')
    if '.' not in v[0] and '[' not in v[0] and len(v)>5:
        assert v[1]=='COMPLETED' and v[2]=='0:0'
        h,m,s=map(int,v[3].split(':'))
        rows.append({'job':v[0],'seconds':3600*h+60*m+s,'start_moscow':v[4],'end_moscow':v[5]})
assert len(rows)==18
events=[]
for v in rows:
    events.extend([(v['start_moscow'],1),(v['end_moscow'],-1)])
active=peak=0
for stamp,change in sorted(events):
    active+=change;peak=max(peak,active)
assert active==0 and peak<=2
cost={'main_slurm_jobs':rows,'allocated_GPU_seconds':sum(v['seconds'] for v in rows),'peak_simultaneous_tasks':peak,
      'complete_models':9,'epochs_per_model':30,'segments_per_model':2,'failed_jobs':0}
cost['allocated_GPU_hours']=cost['allocated_GPU_seconds']/3600
(HERE/'execution_cost.json').write_text(json.dumps(cost,indent=2)+'\n')


def fmt_error(mean, error, signed=False):
    # At most two significant uncertainty digits; central value rounded to
    # the same place. Large MeV uncertainties use an explicit x10^3 scale.
    factor = 1000 if abs(error) >= 100 else 1
    m,e = mean/factor,error/factor
    places = 1-int(math.floor(math.log10(abs(e)))) if e else 6
    m,e = round(m,places),round(e,places)
    decimals=max(0,places)
    sign='+' if signed else ''
    text=f'{m:{sign}.{decimals}f} ± {e:.{decimals}f}'
    return '('+text+') ×10³' if factor==1000 else text


def pair(key,arm,scale=1,places=6):
    v=R['macro_by_seed'][key][arm]
    return fmt_error(v['mean']*scale,v['seed_SD']*scale)


out=['# No-floor MC-label ablation: nine complete SMASH models',
     '', '04.11.2026. Все девять моделей завершили по 30 эпох, 15+15, без сбоев.',
     'Последняя задача закончилась в 08:38:30 МСК. Архивы, каждый member, coherent',
     'resume и все 262 read-only source-файла проверены. Продолжение сохранило',
     'историю и checkpoints первых 15 эпох. Новых model calls при анализе нет.', '',
     '**Итог:** снятие energy clause улучшило AUC/AP на общей расширенной',
     'разметке disabled и регрессию добавленных мягких targets. На общей native',
     'разметке AUC/AP снизились. На common positives изменение raw MAE не',
     'разрешено парным сравнением трёх seeds; physical Rₙ здесь не оценивался.', '',
     '## Популяция и приёмка', '',
     f"Исходные fold-validation роли: **{R['groups']} групп, {R['n_unique_candidates']:,} кластеров**.",
     f"Native positives **{R['reference_counts']['native']:,}**, disabled **{R['reference_counts']['disabled']:,}**;",
     f"добавлено **{R['reference_counts']['disabled']-R['reference_counts']['native']:,}** мягких объектов.",
     'Четыре unbound detector-labelled группы сохранены. Это популяция validation,',
     'отличная от truth-bound S286 census с 5480 добавленными объектами.',
     'Наблюдения, графы, scalers, split 848ebf49, recipe и seeds 42/123/456 прежние.',
     'Менялось энергетическое условие MC label/target; Edep/Side/ToF не изменялись.', '',
     'Baseline — raw B0 на тех же validation candidates, не прежняя test-сводка',
     'с AUC≈0,9547. Новый arm и baseline оцениваются против каждой из **двух',
     'одинаковых reference-разметок**. Энергия — initial MCTrack max-target кластера,',
     'не deposited energy и не независимо восстановленная энергия нейтронного трека.',
     'MC-negative e_true=0 не интерпретируется как физическая нулевая энергия.', '',
     'Старый `predictions_B.npz` не содержит Row/side/ClusterID. Их порядок внутри',
     'каждого content связан с native pack через actual score_pack/check_prediction',
     'и digest candidate_keys принятого patch. Digest новых CSV совпал с patch;',
     'native pack SHA каждого B/fold совпал с prepare4465983. Native float32',
     'labels/targets, cluster_size и counts воспроизвелись точно. Затем выполнен',
     'join 1:1 по (content, Row, side, ClusterID), без intersection/drop fallback.', '',
     '## Общие задачи классификации', '',
     'Ниже равновесное среднее трёх fold-метрик внутри seed, затем mean ± sample',
     'SD трёх training seeds. У каждого arm epochs выбирались на этой validation;',
     'baseline cuts также были настроены на ней. Это selected-validation диагностика',
     'на одной продукции, без fresh holdout и без новой F-оценки. Fold×seed не',
     'образуют девять независимых повторов; pooled cross-fold AUC не используется.', '',
     '| Общая MC reference | Метрика, единица; направление | Native baseline B0 | No energy clause |',
     '| --- | --- | --- | --- |']
for ref,label in (('native','Native Ekin>0.3 GeV'),('disabled','Disabled energy clause')):
    for metric,name,scale,places in (('auc','ROC-AUC [1], ↑',1,6),('ap','AP [1], ↑',1,6),('purity','Purity [%], ↑',100,3),('efficiency','Cluster recall [%], ↑',100,3)):
        key=ref+'/'+metric
        out.append(f'| {label} | {name} | {pair(key,"baseline",scale,places)} | {pair(key,"no_floor",scale,places)} |')
out += ['', '![Common-reference ROC and PR](01_common_reference_curves.png)', '',
        'Reconstructed scores против MC labels, SMASH Xe+Cs, 2,87 ГэВ/нуклон;',
        'отбора по B нет. Curves сначала усредняются по folds, затем по seeds;',
        'полосы — seed SD. Grid 201 точка [0;1] — только display interpolation,',
        'не новый подбор cut. AP вычислен непосредственно из каждого fold,',
        'не как площадь под усреднённой нарисованной кривой.', '',
        '## Raw energy и численный operating point', '',
        'Soft calibration не подгонялась. Для каждого fold/seed повторно применён',
        'его прежний численный B0 cut. Это проверка переноса cut, а не сравнение',
        'при гарантированно одинаковой purity. Новый cut не выбирался.',
        'Исторический baseline выбирал первый из 401 cuts [0,001;0,999],',
        'достигавший cluster-count purity≥0,70 на validation. Этот выбор связан',
        'с прежним operating point, а не с оптимальностью порога энергии 0,3 ГэВ.', '',
        '| Фиксированная MC популяция | Метрика, единица; направление | Native B0 | No energy clause |',
        '| --- | --- | --- | --- |']
for pop,label in (('common_positive','Common positives, N=326651'),('added_soft','Added soft, N=5564')):
    for metric,name in (('raw_mae_GeV','Raw MAE [MeV], ↓'),('raw_bias_GeV','Raw bias [MeV], ближе к 0')):
        out.append(f'| {label} | {name} | {pair(pop+"/"+metric,"baseline",1000,3)} | {pair(pop+"/"+metric,"no_floor",1000,3)} |')
out.append(f'| Added soft, N=5564 | Selected fraction [%], ↑ | {pair("added_soft/selected_fraction","baseline",100,3)} | {pair("added_soft/selected_fraction","no_floor",100,3)} |')
out += ['', 'Нулевых positive-energy targets нет. Common-positive targets прежние;',
        'все добавленные targets ниже 0,3 ГэВ. На added-soft популяции baseline',
        'energy head работала вне своей исходной regression training target:',
        'эти объекты раньше были native-negative. Поэтому большой raw MAE baseline',
        'на них не описывает её прежнюю signal-energy resolution.', '',
        'При прежних cuts новый arm выбирает только около 3,3% добавленных мягких',
        'кластеров. Уменьшение их MAE на всех candidates не является измерением',
        'нейтронной detection efficiency, primary spectrum или physical Rₙ.', '',
        '![Paired seed metrics](02_paired_metrics.png)', '',
        'Каждая линия связывает один training seed после одинакового macro по трём',
        'folds. Разметка и candidates внутри каждой показанной задачи общие.',
        'Небольшой рост common-positive MAE не разрешён на трёх seeds.', '',
        '![Raw energy response](03_raw_energy_response.png)', '',
        'Reconstructed raw energy против MC initial matched-track target отдельно',
        'на common positives и added soft. Внутри display bins объединены candidates',
        'трёх folds; точки стоят в фактическом mean target, бары — SD трёх seeds.',
        'Это описательная response-проекция, не primary fold-macro estimator таблиц.',
        'Все target bins сохранены; binning унаследован от report/census и не',
        'используется для выбора label, окна или cut. Средняя response не является',
        'полной resolution distribution.', '',
        '## Парное сравнение и пределы ошибки', '',
        'Случай разных обученных весов: Δ=no-floor−baseline на одной reference.',
        'В таблице mean ± SE среднего трёх парных training-seed разностей;',
        't имеет 2 df, двухсторонний tcrit(0,05)=4,303. Это условная проверка',
        'инициализации на фиксированных данных и recipe. Она не включает',
        'file/domain/production variance, selection bias или новую test-проверку.',
        'Показанные endpoints множественны; p-значения в JSON не имеют поправки',
        'на множественные сравнения и не являются подтверждением generalisation.', '',
        '| Метрика Δ, единица | Δ seeds 42 / 123 / 456 | Mean ± SE | t, df=2 |',
        '| --- | --- | --- | --- |']
for key,label,scale,places in [('native/auc','Native ROC-AUC [1]',1,6),('disabled/auc','Disabled ROC-AUC [1]',1,6),('native/ap','Native AP [1]',1,6),('disabled/ap','Disabled AP [1]',1,6),('common_positive/raw_mae_GeV','Common-positive MAE [MeV]',1000,3),('added_soft/raw_mae_GeV','Added-soft MAE [MeV]',1000,3)]:
    d=R['paired_differences'][key]
    decimals=1-int(math.floor(math.log10(abs(d['SE_mean_3_seeds']*scale))))
    vals=' / '.join(f'{round(x*scale,decimals):+.{max(0,decimals)}f}' for x in d['by_seed_42_123_456'])
    out.append(f'| {label} | {vals} | {fmt_error(d["mean"]*scale,d["SE_mean_3_seeds"]*scale,True)} | {d["t"]:.3g} |')
out += ['', '## Полнота обучения и selectors', '',
        'Все строки — полный 30-epoch seed. e_cls/e_en в таблице имеют **нулевую',
        'индексацию**; на рисунке эпохи показаны как 1–30. У четырёх новых моделей',
        'e_cls=29: классификационный selector упирается в конец бюджета.',
        'Обучение завершено по протоколу; сходимость за пределами 30 эпох не проверялась.', '',
        '| Fold | Seed | Epochs | B0 e_cls/e_en | No-floor e_cls/e_en | Frozen B0 cut |',
        '| --- | --- | --- | --- | --- | --- |']
for r in R['records']:
    b,n=r['selection_baseline'],r['selection_no_floor']
    out.append(f'| {r["fold"]} | {r["seed"]} | 30 | {b["e_cls"]}/{b["e_en"]} | {n["e_cls"]}/{n["e_en"]} | {r["cut"]:.6f} |')
out += ['', '![Validation histories](04_validation_histories.png)', '',
        'BCE и energy-component histories относятся к собственным training targets',
        'каждого arm. Сравнивать их абсолютные уровни как выигрыш общей задачи нельзя.',
        'Отмечены per-head minima и единственная запланированная граница после 15 эпох.', '',
        f"Выделенное GPU-время: **{cost['allocated_GPU_seconds']:,} с = {cost['allocated_GPU_hours']:.2f} GPU·ч**;",
        'сумма 18 main Slurm records, без повторного счёта batch/extern steps.',
        'Это allocation с I/O/validation, а не pure fit. Одновременно не более',
        'двух campaign tasks, каждый segment в пределах 2:45. Новых test jobs',
        'или дополнительных unit tests не запускалось по прямому human waiver;',
        'source/input/native-label/data assertions сохранены. Ошибка первого',
        'локального collector lookup для отсутствующей fold-role исправлена через',
        'явный .get; исходный traceback сохранён, научные jobs не повторялись.', '',
        '## Вывод и граница конечного опыта', '',
        'No-floor подтвердил обучаемость добавленных soft targets и улучшение',
        'расширенной MC-label задачи. Он меняет компромисс с native-задачей;',
        'из этого опыта не следует, что 0,3 ГэВ — оптимальная физическая граница',
        'или что no-floor уже следует заменить основной pipeline.',
        'Nonprimary qualifying MCTrack не означает автоматически material background;',
        'полная genealogy отсутствует. Потери до retained hits этим опытом не оценены.', '',
        'Эта конечная кампания завершена. F scoring, primary truth/Rₙ comparison,',
        'новый cut fit, soft calibration, второй floor и скан 148 окон не выполнялись.',
        'Ранее принятые baseline-контрасты 18/0, 90/0, 90/18 в',
        '[статусе](../status_20261103.md) сохраняются; новые числа их не заменяют.',
        'Для следующего physical transfer потребуются отдельно зафиксированные',
        'operating policy, исходный primary truth и применимая авторизация.', '',
        '## Происхождение и воспроизведение', '',
        '- Source commit `721beff`; sums `921e04a4e6d1faec8e8b1a757e2a08df2b02f34d2a2f35e9d4e34c0ef750eb3c`.',
        '- Accepted target preparation4482805; immutable patch summary SHA в `input_pins.json`.',
        '- Archive acceptance: `acceptance.json`; compact selected-member hashes: `compact_manifest.json`.',
        '- Arrays4482848/4482849, whole-array afterok, registered 15+15; actual accounting in `execution_cost.json`.',
        '- [Все метрики и certified candidate keys](results.json); [data/collector pins](input_pins.json); [paired curve/response arrays](plot_data.npz).',
        '- Подробные fold/seed таблицы: [классификация](per_fold_classification.csv), [raw energy](per_fold_energy.csv).',
        '- Model-free replay: `python reports/label_ablation_results_20261104/collect.py`, затем `plot.py` и `build_report.py` в той же директории.',
        '- Collector требует локальный hash-verified compact archive; raw данные, новые checkpoint forward passes или cluster jobs не нужны.',
        '- Дополнительные unit tests не запускались. Реальные export/native/reference/resume/SHA assertions выполнены на сохранённых данных.',
        '- [Read-only review](review.md) и [visual/font/hash validation](validation.json).', '']
(HERE/'README.md').write_text('\n'.join(out))
(HERE/'acceptance.json').write_text(json.dumps(ACCEPT,indent=2)+'\n')
manifest=(DATA/'compact_manifest.json').read_bytes()
(HERE/'compact_manifest.json').write_bytes(manifest)
for name,pops in (('per_fold_classification.csv',('native','disabled')),('per_fold_energy.csv',('common_positive','added_soft'))):
    records=[]
    for r in R['records']:
        for arm in ('baseline','no_floor'):
            for pop in pops:
                records.append({'fold':r['fold'],'seed':r['seed'],'arm':arm,'population':pop,
                                'cut':r['cut'],**r['metrics'][arm][pop]})
    with (HERE/name).open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
print('Wrote finite no-floor report, acceptance and cost')
