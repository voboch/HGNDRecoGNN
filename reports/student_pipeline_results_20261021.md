# Результат native HGND проверки студенческих архитектур

Проверено 21 октября 2026. Все12 обучений4424084 и test4424085 завершены, exit0. Каждый run прошёл30epochs; выбранные validation checkpoints и thresholds сохранены без изменений. Проверены hashes всех12 prediction CSV, weights, atomic resumes и histories. Повторной inference не было.

<!-- table-provenance: reconstructed native HGND clusters scored against MC truth on reused test source988976, 222251 clusters /36713 signal /91081 physical events; trained on988975 with train-only scalers and selected on validation988974; DCM frozen detector cache schema2, collision settings and generation lineage unverified, no b selection (b unavailable). -->

ROC-AUC и average precision безразмерны, higher is better. Energy MAE в MeV, lower is better. Среднее ± sample SD по трём training seeds42/123/456; это не SE, не CI и не between-source variance. Energy population: все36713 MC-positive clusters с e_true>0, без score cut. Energy target — native maximum qualifying hit energy; это не разрешение нейтронных tracks.

| Модель | ROC-AUC ↑ (dimensionless) | Average precision ↑ (dimensionless) | Energy MAE ↓ (MeV) |
|---|---:|---:|---:|
| GraphSAGE | 0.9080 ± 0.0077 | 0.701 ± 0.010 | 249 ± 26 |
| DynamicEdgeConv | 0.9095 ± 0.0060 | 0.709 ± 0.014 | 261 ± 19 |
| Spectral-gated GraphSAGE | 0.9060 ± 0.0043 | 0.693 ± 0.010 | 244 ± 6 |
| Spectral-gated DynamicEdgeConv | 0.9062 ± 0.0054 | 0.708 ± 0.010 | 256 ± 15 |

Это проверка четырёх архитектур в другом task/recipe, а не single-factor ablation исходной студенческой задачи. Разницу между ROC-AUC≈0.98 в student event split и≈0.91 здесь нельзя приписывать одному split, метке или time noise. Source988976 уже был открыт ранее; новые веса/пороги по его результатам не выбираются. Победитель модели и преимущество spectral gate по этой таблице не объявляются.

Canonical inputs: `results/student_pipeline_20261020/evaluation_summary_20261021.json` и `evaluation_per_seed_20261021.json`. Проверка: `completion_validation_20261021.json`. Scope, optimizer, train objective и source/data hashes зафиксированы в `student_pipeline_protocol_20261020.md`. Диагностика и визуальная проверка12 histories,12 validation scans и4 energy-response figures завершены21октября: `student_pipeline_diagnostics_20261021.md`. Per-seed cuts и model recipe не менялись.
