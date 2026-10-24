# Полный corrected-SMASH transfer: проверенные результаты

24 октября 2026. Gate сохранённых предсказаний и cluster metrics пройден.
Все три GPU jobs завершились COMPLETED/exit0; проверены SHA256 всех трёх
архивов, 1296 prediction CSV, полное membership и неизменные checkpoint/cut/scaler
inputs. Метрики независимо пересчитаны из CSV. SMASH training и fitting не было.

Числа таблиц ниже сгенерированы из одного источника:
`results/smash_full_transfer_20261023/full_metrics_verified.json`.
Сводки jobs и input gates: `full_transfer_artifacts_verified.json` и
`full_transfer_summary_bundle.json` в том же каталоге.

## Популяция и модель

Использованы три native DCM GraphSAGE checkpoints, seeds42/123/456. Веса и
нормировки обучены на DCM988975; checkpoint и score cut выбраны на DCM988974.
Cut по seed:0.465070 /0.357785 /0.397705. Основание cut — сохранённый полный
validation scan native DCM, см. `student_pipeline_diagnostics_20261021.md`.
На SMASH пороги не подбирались. Данные — corrected hardSkyrme Xe+Cs2.87GeV/A,
U_sym0/18/90MeV, frozen common-file retrospective development population.
Fresh holdout и независимость production jobs не установлены.

Native graph сохраняет r3.6cells/time1.0ns, одиночные хиты,10 input features,
путь-зависимый jitter sigma0.15ns/seed42 и train-only DCM affine/scaling.
Это прежние conventions для сопоставимости, без новой оптимизации. Сигнал —
native MC cluster label с energy floor0.3GeV и прежним ToF coincidence.
MC truth используется для labels/targets и binding, forward получает hits.

В original ledger сохранены568 members:432 уникальных непустых pairs,
100 header-only pairs и36 exact byte aliases bigSpot. В header-only CSV нет
экспортированных строк или candidates; aliases исключены из повторного scoring. Число
processed ROOT events, включая zero-response, остаётся неизвестным.

Scored on: retrospective corrected-SMASH development; fitted on DCM train988975,
checkpoint/cut selected on validation988974. No direction is better for these counts.

| Potential | Original members (count) | Scored nonempty files (count) | Clusters (count) | MC-positive clusters (count) | Events with candidates (count) |
|---|---:|---:|---:|---:|---:|
| zeroSpot | 195 | 195 | 1 220 632 | 218 410 | 594 190 |
| defaultSpot | 200 | 186 | 1 160 155 | 208 101 | 563 287 |
| bigSpot | 173 | 51 | 322 049 | 57 959 | 156 757 |

Event key: canonical detector member + Row; обе detector arms остаются
вместе. Events with candidates не являются числом всех processed/generated events.

## Классификация

Для каждого potential и seed использована собственная ROC; seeds не
складываются в одну population. Все ± — sample SD трёх DCM training seeds
на одних и тех же SMASH candidates. Это не MC/source error и не SE среднего.
Purity — доля native MC-positive среди выбранных clusters; efficiency —
выбранная доля native MC-positive candidates, без generated-neutron denominator.
Scored on: retrospective SMASH development; fitted on DCM train988975,
cut/checkpoint selected on validation988974. Higher AUC/AP/purity/efficiency is better.

| Potential | ROC-AUC mean ± training seed SD (1) | AP mean ± training seed SD (1) | Purity mean ± training seed SD (%) | Conditional efficiency mean ± training seed SD (%) |
|---|---:|---:|---:|---:|
| zeroSpot | 0.8999 ± 0.0073 | 0.7013 ± 0.0086 | 69.97 ± 0.30 | 54.99 ± 1.03 |
| defaultSpot | 0.8993 ± 0.0074 | 0.6992 ± 0.0092 | 69.85 ± 0.33 | 54.71 ± 1.08 |
| bigSpot | 0.9002 ± 0.0072 | 0.7024 ± 0.0087 | 70.05 ± 0.30 | 55.07 ± 0.95 |

Per-seed ROC-AUC приведён отдельно, без усреднения scores.
Scored on: те же SMASH candidates; fitted on DCM train988975/validation988974.
Higher ROC-AUC is better; столбцы — разные DCM training seeds.

| Potential | Seed42 ROC-AUC (1) | Seed123 ROC-AUC (1) | Seed456 ROC-AUC (1) |
|---|---:|---:|---:|
| zeroSpot | 0.9054 | 0.9027 | 0.8916 |
| defaultSpot | 0.9047 | 0.9022 | 0.8909 |
| bigSpot | 0.9056 | 0.9029 | 0.8920 |

Близкие ROC при трёх potentials описывают качество signal/background
classification. Чувствительность к U_sym требует отдельной проверки
физического observable, его energy response и closure. Значимость различий
между potentials и с reused DCM test здесь не оценивается.

## Размер кластера

Границы1,2–3,4–7,≥8hits зафиксированы в DCM diagnostics до этих SMASH scores.
Это диагностические size bands; удаление группы или новый selection по ним
не выполнялись. Все n одинаковы по трём seeds в каждой строке.
Scored on: те же retrospective SMASH candidates; fitted on DCM train988975,
checkpoint/cut selected on validation988974. Higher ROC-AUC is better;
± — sample SD трёх training seeds, без MC/source uncertainty.

| Potential | Cluster size (hits) | Clusters (count) | MC-positive clusters (count) | ROC-AUC mean ± training seed SD (1) |
|---|---:|---:|---:|---:|
| zeroSpot | 1 | 472 953 | 33 845 | 0.8078 ± 0.0105 |
| zeroSpot | 2–3 | 348 509 | 66 606 | 0.8572 ± 0.0109 |
| zeroSpot | 4–7 | 274 811 | 78 373 | 0.9282 ± 0.0035 |
| zeroSpot | ≥8 | 124 359 | 39 586 | 0.9356 ± 0.0030 |
| defaultSpot | 1 | 450 560 | 32 149 | 0.8076 ± 0.0107 |
| defaultSpot | 2–3 | 330 414 | 63 762 | 0.8560 ± 0.0110 |
| defaultSpot | 4–7 | 261 134 | 74 767 | 0.9268 ± 0.0034 |
| defaultSpot | ≥8 | 118 047 | 37 423 | 0.9362 ± 0.0035 |
| bigSpot | 1 | 124 489 | 8 956 | 0.8068 ± 0.0104 |
| bigSpot | 2–3 | 92 060 | 17 733 | 0.8574 ± 0.0107 |
| bigSpot | 4–7 | 72 580 | 20 783 | 0.9276 ± 0.0032 |
| bigSpot | ≥8 | 32 920 | 10 487 | 0.9387 ± 0.0036 |

Однохитовые candidates имеют наименьший AUC во всех трёх samples.
Это описательная стратификация на фиксированных весах. Она не выделяет
причину всего student0.98 → native DCM0.91 contrast; контролируемый DCM
population diagnostic сохранён в `roc_auc_gap_20261023.md`.

## Energy target

Ошибки относятся к reconstructed E_pred относительно максимальной энергии
qualifying MC hit в native-positive cluster. Для них n — MC-positive counts
в первой таблице; score cut не применяется. Selected-target metrics и их
per-seed n доступны в canonical JSON. Эти ошибки не являются разрешением
энергии отдельных neutron tracks или physical unfolded spectrum.
Scored on: native-positive retrospective SMASH clusters; fitted on DCM988975,
checkpoint selected on validation988974. MAE lower is better, bias closer to zero is better;
± — sample SD трёх training seeds, не MC/source uncertainty.

| Potential | Bias mean ± training seed SD (MeV) | MAE mean ± training seed SD (MeV) |
|---|---:|---:|
| zeroSpot | -48.3 ± 53.2 | 167.7 ± 14.4 |
| defaultSpot | -48.6 ± 53.0 | 168.5 ± 14.7 |
| bigSpot | -52.6 ± 54.4 | 169.7 ± 15.5 |

## Figures и проверка

ROC и полные CDF energy residuals сохранены в
`results/smash_full_transfer_20261023/figures/roc.png` и
`results/smash_full_transfer_20261023/figures/energy_residual_cdf.png`.
ROC использует reconstructed score и native MC label; residual CDF —
reconstructed energy minus native MC max-hit target. Все три seeds показаны
отдельно; подписи содержат MC provenance и n. Хвосты не обрезаны.
Figures визуально проверены; data/annotation bounds проверены программно.
PNG200dpi — report artifacts; manuscript не менялся.

Архивы выгружены без нового compute job. При первом SFTP чтении zeroSpot
получен EACCES, bigSpot glob lookup вернул отсутствие каталога. Последующая
проверка metadata увидела оба файла с прежними owner/modes. Read-only SSH
stream через один descriptor получил исходные bytes; все publisher SHA256
совпали. Причина нестабильной доступности не установлена, RA-09 остаётся OPEN.
Отказ и успешный экспорт сохранены в `artifact_export_failures_20261024.json`
и `artifact_stream_retrieval_20261024.json`. Нового retry расчётов не было.

## Следующий scientific gate

Текущий конечный audit/transfer завершён. Для SMASH training или физической
оценки разделения потенциалов требуется отдельный протокол и исходный ledger:

1. Связать generator job/seed и parent ROOT с каждым CSV, восстановить
   обработанные entry ranges и zero-response accounting. Content binding
   initial MCTracks уже проверен; он не восстанавливает production lineage.
2. По независимым source groups зафиксировать train/development/holdout split
   до обучения. Текущие открытые samples остаются development; fresh holdout нет.
3. На общей популяции и одинаковой кластеризации проверить native/relaxed
   labels, neutron-object matching, energy migration и response выбранного
   physical observable. Новое training требует отдельной авторизации и GPU smoke.

Полный audit label census использовал time1.5ns и дал добавленные soft MC
clusters при снятии energy floor; full model transfer использует time1.0ns
и native labels. Из этих разных graphs нельзя вычислить эффект student
label на чувствительность к U_sym. Такой эффект здесь не измерен.
