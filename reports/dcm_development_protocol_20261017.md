# DCM development: архитектура и training, 2026-10-17

План записан до запуска новой серии. Пользователь поручил продолжить
эксперименты после повторного аудита pipeline. Серия конечная: 4 recipes,
3 training seeds (42, 123, 456), 12 trainings. Одновременно не более двух
GPU jobs. Длинная серия отправляется только после успешного test-partition
GPU smoke всех четырёх recipes; test evaluations этой серией не отправляются.

## Данные и estimator

Используется frozen DCM cache `ndet_dataset_dcm_xecs_own_tw1p0` (schema 2,
277 shards, 283398 arm graphs). Prepared split/scalers берутся из завершённой
source-run series с manifest SHA-256
`1ab9285e3315dc0f043b7d5116255f1459978477dc9a9b4d95e046649163278b`.
Train source run 988975: 92252 graphs / 64138 physical events;
validation 988974: 59973 / 41808; validation scoring: 101107 clusters.
Обе половины события и все файлы source run остаются вместе. Scalers fitted
на восстановленных float32 train-hit features. Новый scaler fit, графы,
labels, edges и timing noise не создаются.

Старый test 988976 уже открыт предыдущим опытом и не используется новой
серией. Loader держит только train/validation graphs; manifests сверяют
disjoint indices. До smoke фиксируются content SHA-256 всех читаемых shards
и cache metadata; каждый training loader проверяет те же hashes. Boundary
shard может физически содержать дополнительные графы, но в dataset попадают
только разрешённые indices. Source code исполняется из нового immutable
snapshot, outputs/checkpoints имеют новый campaign root.

Primary development metric — cluster ROC AUC на validation, dimensionless,
higher is better. Это reused validation с checkpoint selection и последующим
сравнением recipes; independent generalisation не заявляется. Все 12 outputs
сохраняются, включая неудачные или не достигшие conventional purity.

## Зафиксированные recipes и ожидания

Общий reference: `net_default`, H512/L8, mean pooling, BatchNorm в четырёх
SAGE layers, без cluster-size feature и residual. Adam, initial lr 1e-3,
weight decay 1e-5, batch 256, 30 epochs, MultiStepLR 10/15/25 с gamma 0.1,
без warmup/clipping, все пять loss weights равны единице. Это унаследованный
reference для сравнения, не найденный optimum. Новый selector — минимум
population-weighted composite validation loss (RA-01); baseline тоже
переобучается с ним. Старые checkpoints не используются как initial weights.

- `reference`: общий control.
- `layernorm`: только BatchNorm → per-hit LayerNorm, с теми же affine
  parameter counts. Проверяется зависимость от running batch statistics;
  предполагается небольшой положительный либо нулевой эффект AUC, ориентир
  0–0.01. Это гипотеза, а не установленная причина train/validation расхождения.
- `pool`: только mean → mean/max/sum; size feature выключен. Мотив — более
  полное представление распределения hit embeddings. Ожидается положительный
  либо нулевой эффект порядка 0–0.01 AUC; увеличение capacity измеряется и
  сообщается, pure pooling gain при фиксированной capacity не заявляется.
- `low_lr`: только initial lr 1e-3 → 1e-4. One-decade perturbation проверяет
  чувствительность к начальному шагу; не выбирается лучшая точка sweep.
  Ожидание 0–0.01 AUC по validation с риском недообучения за те же 30 epochs.

Ориентиры величины эффектов задают масштаб проверки, а не power guarantee.
Ранее validation loss достигал минимума рано; это мотив проверять training
устойчивость, но не доказательство конкретного механизма. Plot использованных
LR schedules и исходных loss trajectories сохраняется в dated results.
Ожидаемый расход — около 20–30 GPU hours суммарно, максимум два V100, до
5 часов на task. Smoke: один V100, 4 CPU, до 20 минут; training: 8 CPU.
Slurm публикует RealMemory=1 MB на всех узлах; фиксированный GB request
отвергается. Используется `--mem=0`, как unlimited partition default.
Это настройка scheduler, не оценка peak RSS; память проверяется по runtime.

## Сравнение, пороги и ограничения

Primary contrast — LayerNorm minus reference, paired по трём training seeds.
Сохраняются per-seed AUC, mean difference, SE = sample SD / sqrt(3), t на
2 df и critical 4.303 для двустороннего 0.05. Pooling и low-lr contrasts
вторичные exploratory; одновременного confirmatory заявления по трём тестам
нет. Seed SE не подменяет between-source variance или cluster bootstrap.

Operating threshold — первая из 401 точек [0.001,0.999], достигающая count
purity 0.70 на validation. Это inherited convention: более строгая purity
обычно уменьшает signal efficiency, а не гарантирует лучшую EoS sensitivity.
Сохраняются полный scan и его plot. Если target недостижим, threshold=None
и этот исход сообщается; recipe/seed не исключается из AUC summary.
Energy metrics вторичны и относятся к selected signal clusters, не к
нейтронным tracks или unfolded physical spectrum.

Каждый run сохраняет effective config, actual model kwargs/parameter count,
component losses/denominators и LR каждой эпохи, best/last/atomic resume,
predictions и validation metrics, completion/status receipts. Seed ставится
до materialisation модели; CUDA RNG restore проверяется реальным smoke.
GPU smoke также сравнивает отдельную и batched inference двух событий с
настоящим kNN и проверяет save/reload выбранного checkpoint.

Не меняем recipes после старта. При failure сначала logs, затем не более
одного обоснованного исправления/retry в новом snapshot; completed/active
tasks не дублируются. Никаких автоматических новых sweeps по результату.
После завершения этих 12 tasks сохраняются все результаты и ограничения;
новый untouched source run нужен до independent-test заявления в статье.

## Отправка и immutable source

Smoke job `4413427`, partition `test`, один V100/type_a, 4 CPU, 20 минут.
На момент отправки ожидает Resources; прогноз Slurm — 11:08 MSK.
Train array ещё не отправлен: HPC.md требует passed GPU smoke до sbatch
длинного обучения. После успешной проверки четырёх complete receipts и
`smoke_complete.json` отправляются только 12 заданных tasks `0-11%2`.

Execution snapshot:
`/home/vbocharnikov/hgnd_experiments/dcm_dev_20261017_c055b8517a`.
Source SHA-256:
`c055b8517afbd8bf0ffa726882550a014e99bacb6c3ed6ffb8f54390539e7d1a`.
Campaign:
`/scratch/vbocharnikov/hgnd/experiments/dcm_dev_20261017_c055b8517a`.
Prepared artifacts читаются из старого campaign только для frozen split/scalers.
Source tree read-only; все artifacts сохраняются в новом campaign.

Предыдущая попытка отправки с fixed-GB memory request отклонена до создания
job (Slurm RealMemory placeholder). Snapshot `f0a3c379f3` сохранён как
superseded submission attempt, не исполнялся. Настройки architecture/training
не менялись. Новый `--mem=0` проверен через Slurm test-only перед отправкой.
Подробные квитанции и план массива: `results/dcm_development_20261017/`.

## Passed smoke и отправленный массив, 2026-10-17 11:05 MSK

`4413427` COMPLETED, exit 0, 3:52. Проверены все четыре complete receipts,
effective configs, component histories, выбранные checkpoint hashes,
CUDA RNG replay и real-kNN batch invariance. Snapshot сверён по 180 file
hashes. Проверка сохранена в `failed_c055b8517a/validated_smoke.json`
(каталог архивирован после RA-08).

После этого принят training array `4413448`, `0-11%2`. Mapping:
`task//4` выбирает seed из 42/123/456; `task%4` выбирает reference,
layernorm, pool, low_lr. 12 задач, максимум две одновременно, V100/type_a,
8 CPU, до 5 часов, `--mem=0`, partition rocky. Полный argv и квитанция
сохранены в `failed_c055b8517a/submissions.json`. Новых test jobs нет.
Текущее состояние — `results/dcm_development_20261017/launch_status.json`.
Этот массив уже отправлен: перед любой коррекцией сверять Slurm и receipts,
активные и завершённые tasks не дублировать.

## Единственное исправление загрузчика, 2026-10-17 12:46 MSK

Массив `4413448` остановлен: шесть tasks упали при загрузке до первой эпохи,
оставшиеся шесть отменены. Ни один из 12 trainings не завершился; все
первоначальные логи сохранены в `failed_c055b8517a/`. Причина и ограничение
первого smoke описаны как RA-08 в `pipeline_reaudit_20261017.md`.

Исправленный selective loader сохраняет flat-preload semantics по cumulative
actual shard lengths. Новый test-partition smoke сначала проверяет полный
train/validation population и source membership, включая 101107 validation
clusters, затем прежние четыре коротких trainings, RNG и real-kNN проверки.
Сохраняются hashes и длины всех shards, ordered index/Row/side hashes каждого
раздела. Длинные trainings сверяют те же receipts до первой эпохи.

Предусмотренный один bounded correction/retry касается только загрузки и
проверки покрытия. Все четыре recipes, три seeds, 30 epochs, frozen scalers,
partition и validation-only estimator остаются прежними. После passed нового
GPU smoke допустим только replacement `0-11%2` в новом campaign. Старый
`c055b8517a` snapshot и его checkpoints не изменяются. Test jobs запрещены.

Corrected execution snapshot:
`/home/vbocharnikov/hgnd_experiments/dcm_dev_retry_20261017_7dcd849c18`,
source SHA-256
`7dcd849c18f8b496eedb99d7a4abb0c4c63faabd4e578d63583c07c0f7da0bb7`.
Новый campaign:
`/scratch/vbocharnikov/hgnd/experiments/dcm_dev_retry_20261017_7dcd849c18`.
Smoke `4413503` принят в `test` в 12:50:55 MSK, один V100, 4 CPU,
`--mem=0`. Лимит 30 минут покрывает полный inventory и повторную загрузку
точным training loader; это максимальная длительность test-partition job.
Его `smoke_complete.json` должен подтверждать `full_training_loader_replay`
и `full_train_validation_coverage`, hash data manifest, прежние CUDA/kNN
проверки и четыре complete receipts. Полные populations сохраняются отдельно
в `full_population_verified.json`. На 12:51 MSK smoke ожидает Resources,
replacement array ещё не отправлен. Квитанции нового campaign — `current/`.

## Исправленный smoke passed; replacement принят, 2026-10-17 13:25 MSK

`4413503` COMPLETED, exit 0, 11:13. Подтверждены полный train/validation
population, source-run membership и replay точного training loader; обе
детекторные половины остались в прежних разделах. Измерены short shards:
137 — 807 graphs, 276 — 991; после первого из них fixed-size offsets
расходились с flat-preload на 217. Hashes всех 277 shards и actual lengths
зафиксированы в `current/data_manifest.json`, population/identity receipts —
в `current/full_population_verified.json`. Все четыре models/configs,
one-epoch histories, selected checkpoint hashes, CUDA RNG и real-kNN checks
проверены; итоговая квитанция — `current/validated_smoke.json`.

Replacement array `4413514` отправлен в 13:25:28 MSK из immutable
`7dcd849c18`, ровно `0-11%2`. Прежние четыре recipes и seeds 42/123/456,
population validation selector, 30 epochs, rocky/type_a, один GPU, 8 CPU,
`--mem=0`, 5 часов на task. Source/campaign/prepared paths приведены выше;
argv и mapping сохранены в `current/submissions.json`. Этот массив уже
принят: повторно его не отправлять. Train steps начнутся только после
per-task сверки полного population с passed smoke. Test evaluation запрещён;
второго automatic correction/retry и новых sweeps в этом протоколе нет.

## Одна задача остановилась на I/O, 2026-10-17 16:26 MSK

`4413514_2` (`pool_42`) FAILED при записи atomic resume после 16 сохранённых
эпох; остальные healthy tasks продолжаются. Текущий `resume.pt` прочитан
и проверен, `next_epoch=16`, optimizer/scheduler/RNG согласованы. Отдельные
last/optim files опережают его и для продолжения не используются. Создана
проверенная резервная копия 467152443 bytes в отдельном recovery directory.
Подробный разбор — RA-09 в `pipeline_reaudit_20261017.md`.

Предложение одного продолжения оставшихся 14 epochs при исходном total 30,
без смены recipe/source/split, сохранено в `current/pool_42_resume_proposal.json`.
Оно не отправлено: дальнейший automatic retry запрещён текущими heartbeat
instructions. Требуется отдельное разрешение пользователя. Зависимость
`afterany:4413514` в предложении предотвращает третью одновременную задачу;
90 минут покрывают remaining budget с запасом относительно 16 эпох за 52:31.
Без разрешённого resume `pool_42` остаётся incomplete, и paired three-seed
pooling comparison недоступен; результат 16 эпох не заменяет 30-epoch seed.

## Закрытие разрешённых запусков, 2026-10-17 23:53 MSK

Все tasks массива `4413514` terminal. Завершены и проверены десять 30-epoch
runs; `pool_42` и `reference_123` остаются incomplete после checkpoint I/O
failures. Дополнительные resume proposals не отправлены без явного
разрешения пользователя; исходный ban на further automatic retry сохранён.
Новых trainings, smokes, test evaluations и sweeps не назначено.

Итоговый отчёт `dcm_development_results_20261017.md` и canonical
`results/dcm_development_20261017/closure/summary.json` сохраняют все planned
seeds, completion status, точные per-seed AUC, thresholds и provenance.
Плановая статистика primary contrast на 2 df недоступна без reference_123;
missing runs не исключены ради нового двух-seed теста и не заменены partial
weights. Secondary comparisons также не имеют полного planned cohort.
SE/t/mean differences остаются null; available single-pair differences
имеют только descriptive status. Полные histories/scans сохранены в closure.

Разрешённый конечный бюджет и correction checks завершаются этим отчётом;
heartbeat после финальной проверки приостанавливается. Предложения двух
coherent atomic resumes остаются доступными для отдельного решения
пользователя. Это закрытие выполненных разрешённых jobs, а не завершение
всех 12 научных recipes или доказательство отсутствия архитектурного эффекта.
