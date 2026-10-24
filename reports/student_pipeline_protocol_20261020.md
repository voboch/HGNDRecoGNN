# Проверка студенческих моделей в основном HGND pipeline

2026-10-20. Зафиксировано до нового обучения и оценки. Разрешение пользователя:
«нужно протестировать в основном пайплайне на утвержденном разбиении».
Новая конечная серия, без обоих вариантов GATv2. Закрытые задания и исходные
студенческие runs не возобновляются. Основной default checkpoint не заменяется.

## Вопрос и ожидаемый результат

Проверяем качество четырёх архитектур на штатных DCM кандидатах: GraphSAGE,
DynamicEdgeConv и их spectral-gated варианты. Primary metric — cluster ROC-AUC;
secondary — average precision и signal-cluster energy MAE/RMSE/bias, затем
count purity/efficiency на замороженном validation cut. Это реконструированные
кластерные оценки относительно MC truth, не эффективность нейтронных tracks.

На исходном event split студента воспроизведён ROC-AUC 0.9811–0.9834. Здесь
поменялись timing, признаки, кластеризация, labels, разделение по source runs
и training objective. Рабочее ожидание — снижение AUC относительно 0.98;
грубый диагностический диапазон 0.85–0.98 взят из прежних native development
runs и студенческой репликации. Он не является критерием успешности и не
разрешает менять рецепт после теста. Эффект spectral gate заранее неизвестен;
gain и чистая архитектурная причина не предполагаются. Различие задач
исключает прямой статистический contrast с исходными студенческими числами.

## Утверждённые данные

Manifest `dcm_source_run_holdout_v1`, SHA256
`1ab9285e3315dc0f043b7d5116255f1459978477dc9a9b4d95e046649163278b`:
train 988975, validation 988974, test 988976. Обе detector arms одного
physical event остаются вместе. Source/Row/side ownership, индексы, полное
покрытие и неизменность scalers проверяются до подготовки.

Train: 92252 arm graphs / 64138 physical events / 159085 clusters / 26244 signal.
Validation: 59973 / 41808 / 101107 / 16877. Test: 131173 / 91081 / 222251 clusters.
Это население frozen detector cache schema 2; b selection отсутствует,
collision settings и связь с corrected-detector/primary-truth generations
по-прежнему не установлены. Один source run на partition не позволяет
оценить between-source variance.

Используется существующий cache `ndet_dataset_dcm_xecs_own_tw1p0` и его
277 shard hashes, inventory SHA256
`1722523f8708a10d3b6eac43c303fd4780ea31e4df671daf25d37ca69c7a7c29`.
Индексы относятся к конкатенации фактических длин shards, включая короткий
shard137. Признаки, frozen detector timing/noise, edges, все кандидаты,
включая singleton, native cllabel и clenergy сохраняются. Старое scaling
обращается и применяется уже зафиксированный train-only affine remap;
нового fit scalers нет. Дополнительного time jitter нет.

**988976 уже был открыт 17 октября.** Явное новое поручение разрешает эту
оценку на принятом разбиении; она обозначается как reused test. Утверждать
новое независимое подтверждение или подбирать по нему recipes нельзя.
Подготовка и GPU smoke используют только train/validation. Test tensors
строятся отдельной фазой после завершения всего cohort.

## Архитектура и обучение

Registry `student_sage`, `student_dynedge`, `student_spectral_sage`,
`student_spectral_dynedge` использует точные student residual blocks,
attention/mean/max/sum pooling, log1p multiplicity в regression head и обе
головы. Input projection обучается заново на десяти штатных признаках,
поэтому six-feature notebook weights не переиспользуются.

H=128, четыре слоя, dropout=0.1, DEC k=8; spectral PE=4, gate hidden=64,
init std=0.001 — унаследованные настройки проверенных notebooks, не optima.
kNN ограничен отдельным кандидатом, даже внутри одного события. PE зависит
только от замороженной топологии; truth columns в forward не читаются.
Hit sorting для kNN не меняет output cluster-node order. CPU tensor packs
меняют хранение, но сохраняют порядок и float32 values affine remap.

Общий штатный fit: Adam lr=1e-3, weight_decay=1e-5, batch256, workers0,
30 эпох без early stopping, MultiStepLR milestones10/15/25 gamma0.1;
seeds42/123/456. Эти budget/schedule наследуют main reference; больший budget
мог бы изменить convergence, он здесь не оптимизируется. Все 12 runs имеют
одинаковый бюджет и population validation selector. Checkpoint — первый
минимум validation composite loss; test не участвует.

Штатная cluster-only loss: BCEWithLogits + 0.5·signal SmoothL1, без pos_weight,
EventRank и flattening. Коэффициент0.5 унаследован от native cluster-only
ModelSpec; он фиксирует компромисс между классификацией и энергией. Сравнение
со студентом меняет весь task/recipe и не выделяет отдельный causal factor.
Неподдерживаемые hit/link heads не получают вымышленных predictions.

Cut — первое из401 значений [0.001,0.999], где validation count purity≥0.70.
Это прежняя comparison convention: повышение cut обычно уменьшает efficiency
ради purity; EoS optimum не заявляется. Полный scan сохраняется в CSV/PDF/PNG.
Если цель недостижима, threshold и selected metrics остаются null; test cut
не подбирается. Energy errors без cut сохраняются отдельно для всех
MC-positive clusters с e_true>0 и относятся к native max-hit energy target.

## Проверки, расходы и test barrier

Локальные tests: точный roundtrip native pack/remap; отсутствие truth inputs;
изоляция событий и кандидатов; interleaved membership; singleton; нативный
prediction export; запрет test до всех complete receipts. Перед длинной
серией обязателен partition=test GPU smoke всех четырёх моделей с настоящим
CUDA kNN, backward, native fit, checkpoint save/load и batch isolation.
512 train /256 val arm graphs ограничивают инженерную проверку; quality
числа из smoke не используются. Подготовка всех train/val packs и smoke
ограничены30минутами, без test scoring.

12 runs × максимум3часа на V100, максимум два одновременно. Test job
максимум2часа, после afterok всего train array плюс программной проверки
12 complete receipts и hashes source/data/config/checkpoints/histories.
Это верхние allocation bounds, не оценка elapsed time. Точные минуты и
GPU environment записываются после выполнения. Automatic retry/requeue нет.
cn007/cn008 исключаются из размещения после прежних EACCES как эксплуатационная
предосторожность; инфраструктурная причина не считается установленной.
--mem=0 сохраняется из-за cluster placeholder RealMemory=1MB.

Итог — каждый seed и mean±sample SD для полного cohort из трёх seeds.
SD характеризует training variance на одном source run. Незавершённый cohort
не усредняется; pooled-seed ROC и альтернативные significance tests не
подставляются. Разрешения на manuscript update или publication здесь нет.
