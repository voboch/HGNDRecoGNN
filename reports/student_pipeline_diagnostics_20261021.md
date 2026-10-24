# Диагностика native DCM после переноса студенческих архитектур

21 октября 2026. Разобраны готовые предсказания всех 12 обучений 4424084; нового обучения и inference не было. Сверены 75 входных SHA256, включая tensor pack,12 CSV, validation scans,12 histories и completion receipts. Ключи Row/side/local ClusterID и MC targets совпадают с native pack. ROC-AUC и energy MAE воспроизведены из CSV.

Популяция: reused test source 988976,222251 candidates,36713 MC-positive clusters,91081 events с хотя бы одним native candidate. Это не full generated-event inventory. Train 988975; checkpoint и score cut выбраны только на validation 988974. Для DCM collision-generation lineage не установлена; b selection отсутствует, b unavailable. Test уже открывался ранее; этот пакет даёт диагностику, а не независимое подтверждение или выбор новой модели.

<!-- table-provenance: reconstructed native HGND clusters against MC truth on reused DCM test source988976, 222251 candidates /36713 positive /91081 candidate-bearing events; trained on988975, scored on reused test988976 (held-out source, previously opened), frozen selector and cuts validation988974; collision generation lineage unverified; no b selection (b unavailable). -->

Среднее ± sample SD по training seeds42/123/456. Все fractions безразмерны; higher is better. Это training variation на одном source, не SE/CI и не between-source uncertainty. Порог каждого run — первый сохранённый validation cut с count purity≥0.70; test cuts не менялись.

| Модель | Count purity ↑ (dimensionless) | Count efficiency ↑ (dimensionless) |
|---|---:|---:|
| GraphSAGE | 0.6819 ± 0.0064 | 0.5815 ± 0.0188 |
| DynamicEdgeConv | 0.6840 ± 0.0029 | 0.5853 ± 0.0483 |
| Spectral GraphSAGE | 0.6842 ± 0.0021 | 0.5637 ± 0.0265 |
| Spectral DynamicEdgeConv | 0.6883 ± 0.0040 | 0.6030 ± 0.0064 |

Validation target 0.70 не переносится на reused test точно: здесь mean purity 0.682–0.688. Это измеренный сдвиг; новый cut на test не подбирался.

## Что ограничивает качество

11 из 12 selected checkpoints имеют epoch 0, то есть конец первой эпохи; DynamicEdgeConv seed 456 имеет epoch 1. К эпохе 29 validation composite loss 0.527–0.579 против выбранных минимумов 0.321–0.369, train loss 0.037–0.040. Основное ухудшение validation идёт по cluster BCE. Train curves используют minibatch mean, validation — population aggregation; их разность не объявляется отдельной оценкой generalisation error. Рост validation loss во времени показывает, что увеличение числа эпох в этом recipe не решает проблему. Разделить source shift, capacity и прочие причины этим наблюдением нельзя.

Одиночных кандидатов 85816, из них 4703 MC-positive. При frozen cuts mean efficiency по моделям 0.098–0.113, purity 0.241–0.263; для кластеров 4–7 hits efficiency 0.729–0.774. Эти интервалы — диапазоны четырёх per-model means, не uncertainty bands. Удалять singleton candidates по открытому test нельзя: это изменило бы acceptance и denominator. Нужна отдельная development-проверка образования кандидатов и их энергетического response.

Энергозависимая efficiency мала в мягкой области: при native MC target 0.3–0.5 GeV четыре per-model means составляют 0.124–0.186. Для 2.6–3.0 GeV —0.716–0.745. Энергетическая эффективность относится к уже образованным MC-positive clusters, не ко всем прибывшим или generated neutrons.

На всех 496 положительных кластерах с native target 5.0–6.5 GeV per-model mean bias составляет−1.61…−1.39 GeV. Все seeds/strata сохранены; на графике показаны mean±sample SD по трём seeds. В этом хвосте нужна проверка energy migration перед использованием hardness response. Target — максимум qualifying hit Ekin в компоненте, а не энергия однозначно сопоставленного neutron track.

Multiplicity задана количеством всех кандидатов в physical event, суммарно по обеим сторонам. Для 1 candidate per event mean AUC0.926–0.932, для≥8 —0.876–0.877; это descriptive source-specific slices. Side-specific и reconstructed-energy slices также сохранены. Truth-energy slices содержат только positive clusters: AUC/AP/purity там null; по ним считаются efficiency и energy errors. Фоновые energy slices заданы через predicted energy и содержат обе labels.

## Графики и воспроизведение

`results/student_pipeline_20261020/diagnostics_20261021/figures/` содержит12 histories с отдельными classification/energy losses,12 полных validation threshold scans и4 energy-response figures. Все28 PDF отрендерены и визуально просмотрены; fonts embedded TrueType, data/annotation/legend bounds проверены. На curves не оценены MC-statistical intervals; energy bands показывают только training sample SD. Небольшие bins и открытый energy tail сохранены в JSON, численные physics bands не выбраны.

Canonical per-run numbers: `results/student_pipeline_20261020/diagnostics_20261021/per_seed.json`; mean/SD: `summary.json`; hashes и QA: `validation.json`, `render_validation.json` в том же каталоге. Исходный archive SHA256:2443b56046cd657ea36c49b9585d75bd77474f393cf3df4b47d20f9966f8b284.

Команда из HGNDRecoGNN с новым output directory:

```bash
python scripts/diagnose_student_pipeline.py \
  --inputs results/student_pipeline_20261020/diagnostic_inputs_20261021 \
  --completed results/student_pipeline_20261020/completed_20261021 \
  --output /path/to/new-diagnostic-directory
```

Перед следующим training recipe требуются development controls для source dependence, singleton losses и энергетического хвоста. Переход к оценке U_sym остаётся условным: corrected SMASH provenance, общий neutron matching и finite energy bands должны пройти отдельный аудит.
