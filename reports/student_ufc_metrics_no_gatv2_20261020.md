# Тестовые метрики студенческих моделей без GATv2 — 2026-10-20

<!-- table-provenance: Scored on held-out student test events; weights fitted on 137918 train events, checkpoints selected on 29554 validation events; same frozen Jan26 input and preprocessing for all rows. -->

Обучение и оценка завершены. Продолжение 4423612 и оценка 4423613: Slurm COMPLETED 0:0. Все 18 runs имеют полные receipts; хеши best/resume/history и 18 CSV проверены. По запросу пользователя здесь исключены baseline и spectral-gated GATv2; остаются 12 обученных replicas.

Scored on: held-out student event split of hits_Jan_26.pkl. Fitted on 137918 train events; checkpoint selected on 29554 validation events. Test: 29555 events, 24811 with UFC candidates, 36343 candidates, 9857 MC-positive. No source-run holdout is established; the input has no source-run manifest.

Reconstructed candidate scores against MC truth labels. Higher is better. Each entry is mean ± sample SD over training seeds 42/7/21, n=3; SD is not SE, a confidence interval or between-source uncertainty. PR-AUC denotes average precision. No seed pooling or threshold selection.

| Модель | Test ROC-AUC | Test PR-AUC (AP) |
|---|---:|---:|
| GraphSAGE | 0.98210 ± 0.00013 | 0.95476 ± 0.00073 |
| Spectral-gated GraphSAGE | 0.98112 ± 0.00040 | 0.95318 ± 0.00185 |
| DynamicEdgeConv | 0.98338 ± 0.00048 | 0.95768 ± 0.00163 |
| Spectral-gated DynamicEdgeConv | 0.98336 ± 0.00068 | 0.95715 ± 0.00077 |

Reconstructed energy against MC target on 9857 MC-positive test candidates with target E>0, without a classifier score cut. Lower is better for MAE/RMSE; absolute bias closer to zero is better. Same mean ± training-seed SD, n=3.

| Модель | Energy MAE (MeV) | Energy RMSE (MeV) | Energy bias (MeV) |
|---|---:|---:|---:|
| GraphSAGE | 160.7 ± 14.5 | 290.8 ± 16.8 | 10.2 ± 12.4 |
| Spectral-gated GraphSAGE | 170.9 ± 5.0 | 301.1 ± 4.6 | 11.1 ± 17.3 |
| DynamicEdgeConv | 154.6 ± 11.5 | 281.0 ± 8.6 | -50.8 ± 31.4 |
| Spectral-gated DynamicEdgeConv | 158.3 ± 5.5 | 288.4 ± 3.9 | 1.2 ± 70.4 |

Энергетическая мишень — максимум истинной кинетической энергии подходящих neutron hits внутри UFC-компоненты, как при обучении. Эти ошибки не являются разрешением энергии уникального нейтрона или ошибкой восстановленного физического спектра. Эффективность обнаружения отдельных нейтронов здесь не измеряется.

Параметры preprocessing унаследованы из тетрадок: fTime_nonoise без дополнительного размытия, прежние UFC и MC labels. Эти метрики не переносятся напрямую на другой HGND/DCM benchmark.

Канонический источник чисел: results/student_ufc_reproduce_20261019/no_gatv2_metrics_20261020.json. Скрипт: scripts/summarize_student_metrics.py. Полный архив сохраняет результаты всех шести моделей; их обучение и исходные предсказания не изменены.
