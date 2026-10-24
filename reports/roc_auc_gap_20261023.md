# Откуда разница ROC-AUC

23 октября 2026. Разрыв student≈0.98 → native≈0.91 возникает уже на DCM.
Его нельзя приписать переносу на SMASH. Веса студента отсутствовали; обучение
с нуля на восстановленном student event split дало≈0.98 у четырёх моделей
без GATv2 по трём seeds. Это воспроизведённое качество конкретной задачи.

Основной pipeline использует source-run split988975/988974/988976,
нативные radius3.6/time1.0ns кандидаты с одиночными хитами, десять признаков,
существующий frozen detector timing, train-only scaling и Ekin>0.3GeV в
определении сигнала. Студент использовал random event split одного pickle,
радиус5/time2.5ns, исключение singletons, шесть признаков, fTime_nonoise и
сигнал без этого energy floor. Source lineage pickle не отождествлён с98897x.

Student residual backbone/pooling/heads перенесены, но input projection
обучалась заново. Меняется и обучение: native BCE+0.5·signal SmoothL1,
30эпох, checkpoint по validation composite loss; student BCE+L1+event-ranking,
до100эпох с early stopping по separation. Поэтому общий contrast не выделяет
чистый эффект шума, source split, objective или кластеризации. В native
серии11/12 выбранных checkpoints относятся к первой эпохе; validation BCE
позднее ухудшается. Это наблюдение training trajectory, а не доказанная
причина всего AUC gap. Saved histories и threshold scans сохранены в
`student_pipeline_20261020/diagnostics_20261021`.

## Проверенный вклад состава кандидатов

Новая post-hoc диагностика использует только уже сохранённые предсказания
GraphSAGE seeds42/123/456 на reused DCM test988976. Нового inference, fit,
выбора порогов или recipe по test нет. Все входные hashes сверены с прежним
export manifest. Сигнал — MC cluster label, score — реконструированная оценка.
Число clusters и positives одинаково для трёх seeds. AUC безразмерен,
больше — лучше; spread ниже — sample SD трёх training seeds, не MC/source error.
Scored on: reused held-out DCM source988976; fitted on train988975, checkpoint
and score cut selected on validation988974. Higher ROC-AUC is better.

| Популяция | Clusters (count) | MC positive clusters (count) | ROC-AUC mean (dimensionless) | Training seed SD (dimensionless) |
|---|---:|---:|---:|---:|
| Все native DCM кандидаты | 222251 | 36713 | 0.9080 | 0.0077 |
| Одиночные хиты | 85816 | 4703 | 0.8085 | 0.0125 |
| Более одного хита | 136435 | 32010 | 0.9137 | 0.0054 |

Исключение singletons на фиксированных весах повышает AUC лишь на
0.0057±0.0031 (paired mean±sample SD трёх training seeds);
оно не восстанавливает0.98. Это изменение условной популяции, не retraining
и не чистая абляция preprocessing студента. Сам AUC не является средним
AUC подгрупп: он включает пары signal/background из разных size-групп.
Сохранённое разложение всех таких пар воспроизводит общий AUC с точностью
1e-12. Числа и hashes: `results/smash_full_transfer_20261023/dcm_auc_population_diagnostic.json`.

## Что известно о SMASH

Обновление24октября: полный frozen transfer завершён и проверен. Три
training seeds дают zeroSpot AUC0.8999±0.0073 на1220632clusters/218410positives,
defaultSpot0.8993±0.0074 на1160155/208101, bigSpot0.9002±0.0072 на322049/57959.
Это mean±sample SD трёх DCM training seeds, без MC/source uncertainty.
Scored on retrospective common-file corrected-SMASH development; fitted on
DCM988975, checkpoint/cuts selected on validation988974; fitting на SMASH нет.
Разные sample populations не дают из этого контраста causal attribution или
значимость domain gap. Canonical inputs, size strata, energy targets и QA:
`smash_full_transfer_results_20261024.md`.

Историческая first-file проверка:
First-file probe4435668, на тех же frozen DCM GraphSAGE weights/cuts/scalers:
zeroSpot0.8990±0.0074 (6233clusters/1116positives),
defaultSpot0.9089±0.0065 (1611/329), bigSpot0.8995±0.0066 (6302/1137).
Spread — training seed SD. Это разные populations и размеры, без оценки
MC/source uncertainty; статистическую значимость разности с DCM не заявляем.
Потенциалы не являются физически парными событиями. Full source binding и
conditional transfer завершены по `smash_full_transfer_protocol_20261023.md`.
Новый SMASH training требует separate source grouping/split; fresh holdout нет.
