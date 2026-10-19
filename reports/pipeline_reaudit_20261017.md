# Повторная проверка pipeline, 2026-10-17

Поручение пользователя: повторить аудит pipeline и продолжить архитектурные
и training experiments. Проверены текущие working-tree paths; прежние
immutable snapshots, checkpoints и опубликованные в draft числа не изменены.
Исходная локальная проверка перед правками: 108 Python tests passed.

## Исправленные механизмы

**RA-01, P2: validation loss зависит от размера последнего батча.**
`training/train.py::eval_epoch` усреднял `total` losses батчей с одинаковым
весом. Но hit/link/cluster/energy heads имеют разные числа объектов, и
signal-only energy term бывает пустым. Например, cluster losses 1 и 9 на
2 и 18 clusters дают batch mean 5 вместо population mean 8.2; energy loss
пустого батча дополнительно уменьшал epoch mean. Это дефект статистики
выбора checkpoint, а не обнаруженная ошибка test AUC. Изменение выбранной
эпохи в старой серии этим аудитом не измерено.

Теперь native loss functions явно задают denominator каждого head; epoch
validation сначала суммирует weighted numerators/denominators, затем применяет
loss weights. `validation_aggregation=population` — новый default;
`batch_mean` сохраняет возможность воспроизвести исторический selector.
Mini-batch gradient objective не меняется. Test фиксирует инвариантность
population aggregation к объединению батчей и обработку пустого energy mask.

**RA-02, P2: CUDA RNG сохранён, но не восстановлен.** Atomic resume сохранял
CUDA RNG states, а loader восстанавливал только torch CPU и numpy. Python
random также отсутствовал. Теперь восстанавливаются Python, numpy, torch
CPU и все CUDA RNG states. Проверены локальный replay и сам вызов CUDA restore;
реальный CUDA replay включён в GPU smoke. Это важно для stochastic layers;
величина влияния на старый default Net без dropout не установлена. Bitwise
детерминизм CUDA kernels этим не обещается.

**RA-03, P2: caller мог задать epoch, не совпадающий с atomic resume.**
Проверка `start_epoch == resume.next_epoch` отсутствовала. Теперь mismatch
и смена architecture metadata отвергаются до продолжения обучения.
Новая серия дополнительно сверяет полную recipe/input identity и использует
kernel file lock: duplicate active task не может писать в тот же каталог.

**RA-04, P1 для CLI `--energy-flatten`: occupancy weights fitted до split.**
`scripts/train.py` передавал `_with_energy_flattening` весь dataset до
`build_loaders`, включая evaluation labels. Вызов перенесён после split и
получает только `train_loader.dataset`. Тест проходит через настоящий CLI
и проверяет полученный объект training subset. В source-run schedule серии
этот флаг выключен; исправление не меняет её уже вычисленные AUC. Новая
архитектурная серия также не использует flattening.

**RA-05, P1 для cluster-only architectures: prediction export ожидал 6 outputs.**
Registry HeteroGNN возвращает `(cls_logits, energy)`, а `predict` распаковывал
шесть выходов default Net. Добавлен двухвыходный путь с sigmoid и сохранением
cluster truth/Row/side. Hit/edge tables остаются пустыми: эти головы у такой
модели отсутствуют. Тест проверяет экспорт двух batched событий. Это проверка
контракта exporter; полная научная оценка HGT/GAT/SAGE ещё не выполнена.

**RA-06, P2: журнал LR показывал следующий rate как текущий.** Печать шла
после `scheduler.step`. Теперь отдельно сохраняются `lr_used` и `lr_next`.
Каждая эпоха сохраняет компонентные train/validation losses и denominators,
полную историю в atomic resume и читаемый `history.json`/live status.
Для старых logs сохраняется исходная интерпретация post-step LR.

**RA-07, P2: preload memory report учитывал только global store.**
`g.stores[0]` исключал tensors node/edge stores и сильно занижал displayed MB.
Суммирование исправлено на все stores. Это число по-прежнему не включает
Python-object overhead и не является peak RSS.

## Что остаётся ограничением

Generic CLI по-прежнему имеет исторический graph-random двухчастный split;
его evaluation half участвует в checkpoint selection. Для нового campaign
используется только отдельный source-run driver, а не этот generic протокол.
Связь DCM collision settings с published event benchmark и primary-truth /
corrected-detector lineage не установлена. Старый test 988976 открыт; новая
серия не оценивает его повторно и не объявляется independent test.

На MPS cluster branch default Net по-прежнему отделяет градиент от hit branch.
Это не CUDA path и не основание переносить CUDA results на MPS training.
Числа старой серии остаются корректными метриками её сохранённых весов;
автоматического пересчёта или заявления нового gain по исправлениям нет.

## Следующий эксперимент

Новая ограниченная серия описана в `reports/dcm_development_protocol_20261017.md`:
reference, LayerNorm, pooling без size feature и меньший initial LR, по три
paired training seeds. Все arms используют population validation selector.
Результаты будут development metrics на reused validation; для независимого
подтверждения потребуются новые source runs.

Технические первоисточники: [BatchNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html),
[LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html),
[CUDA RNG restore](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.set_rng_state_all.html).
Используется уже установленное окружение кластера; версии API не обновлялись.

## Проверки и отправка

После исправлений: 119 Python tests passed (11 новых), shell syntax и diff
whitespace checks passed. Strict table lint: 0 проблем в двух новых notes.
LR schedules и прежние loss trajectories сохранены и просмотрены в
`results/dcm_development_20261017/plan/`; это обоснование масштаба контроля,
не доказательство причины раннего validation minimum.

GPU smoke `4413427` принят в `test`; ожидает Resources, оценка старта Slurm
11:08 MSK. Длинные trainings ещё не отправлены. Начальный memory request
был отвергнут до создания job; исправлены resource flags и создан новый
immutable snapshot. Обучение на login nodes не запускалось.
Точный snapshot, hashes и состояние —
`results/dcm_development_20261017/source_manifest.json` и `launch_status.json`.

## GPU-проверка и полная серия, 2026-10-17 11:05 MSK

Smoke `4413427` завершён за 3:52, exit 0. Все четыре recipes выполнили два
train batches, validation, save/reload выбранного checkpoint и finite
inference. Реальный CUDA RNG replay и сравнение отдельной/batched inference
двух событий с настоящим kNN прошли. Проверены все 180 файлов source snapshot,
prepared/data manifests и hashes четырёх выбранных smoke checkpoints.
Эта короткая проверка подтверждает исполнение, а не classifier quality.

Фактическое число параметров reference, LayerNorm и low-lr равно 12479173;
pool — 16673477, size feature выключен. Число affine parameters LayerNorm
совпало с BatchNorm. Полные config/history receipts лежат в
`results/dcm_development_20261017/failed_c055b8517a/validated_smoke.json`
и подкаталогах (после остановки первого массива каталог архивирован).

После подтверждённого smoke отправлен массив `4413448`, ровно `0-11%2`:
четыре recipes для каждого из seeds 42, 123, 456. Один V100 и 8 CPU на task,
до 5 часов. Все 12 задач приняты; test array не отправлялся. Квитанция
той отправки сохранена в `failed_c055b8517a/submissions.json`. Массив позднее
остановлен; причина и исправление записаны ниже как RA-08.

## RA-08: сбой полного selective loader, 2026-10-17 12:46 MSK

**P1 для новой development series.** Массив `4413448` остановлен до первой
эпохи. Tasks 0–5 завершились с `Incomplete selected graph population`, task 6
отменён во время загрузки, tasks 7–11 отменены до запуска. Завершённых
trainings и научных development metrics нет. Логи и cancellation receipt
сохранены в `results/dcm_development_20261017/failed_c055b8517a/`;
immutable source и удалённые artifacts сохранены.

Новый selective loader принимал номинальный `shard_size=1024` за фактическую
длину каждого shard и вычислял index как `sid*1024+offset`. Builder допускает
короткие промежуточные shards на границах фаз, а frozen indices получены из
последовательного объединения фактических списков в `preload()`. Эти способы
индексации расходятся после первого короткого shard. Первоначальный smoke
проверял лишь первые 512 train / 256 validation graphs; полного покрытия
frozen indices он не проверял. Поэтому его pass не был достаточной проверкой
loader для длинного обучения. Точные длины реального cache будут сохранены
исправленным GPU preflight; синтетический regression уже воспроизводит
расхождение на коротком промежуточном shard.

Исправление использует cumulative actual shard lengths. В новом smoke один
полный проход фиксирует hash и длину всех shards, сохраняет только разрешённые
train/validation graphs и до gradient steps проверяет все frozen indices,
source-run membership, physical-event counts и 101107 validation clusters.
Все trainings повторяют эту проверку и сверяют ordered index/Row/side hash
с passed smoke. Затем выполняются прежние четыре коротких GPU проверки.
Recipes, train-fitted scalers, split, графы и число эпох не меняются. Это
единственный предусмотренный bounded correction/retry, в новом snapshot.

Старые grouped/diagnostic jobs использовали `preload()`; обнаруженный дефект
нового selective loader сам по себе не меняет их результаты. Отдельное
ограничение общего lazy `HGNDGraphDataset.get()` — та же fixed-size формула;
для cache с короткими промежуточными shards этот путь не использовать.

Исправление прошло 120 локальных Python tests; после добавления финального
smoke gate повторно прошли все 8 development tests. Новый immutable snapshot
`7dcd849c18` сверён по 180 file hashes. Smoke `4413503` отправлен в `test`
в 12:50:55 MSK и ожидает Resources; оценка Slurm — 13:10:55 MSK.
Он делает полный inventory и затем повторное чтение точным training loader,
поэтому лимит smoke увеличен до 30 минут (предел `test`). До его verified pass
replacement trainings не отправляются. Состояние и argv сохранены в
`results/dcm_development_20261017/current/smoke_submission.json` и
`launch_status.json`; heartbeat обновлён на этот campaign.

## Полное покрытие подтверждено; replacement отправлен, 2026-10-17 13:25 MSK

Smoke `4413503` COMPLETED, exit 0, 11:13. Inventory измерил 807 graphs в
промежуточном shard 137 и 991 в последнем shard 276; остальные по 1024.
Всего 283398 graphs. После shard 137 fixed-size mapping смещался на 217
позиций относительно `preload()`. Это подтверждает механизм RA-08 на
реальном cache. Второй проход точным training loader восстановил все frozen
train/validation indices и те же ordered index/Row/side hashes.

Проверены train 92252 graphs / 64138 events / 159085 clusters (26244 signal),
validation 59973 / 41808 / 101107 (16877 signal), source membership 988975
и 988974 соответственно. Scalers/prepared artifacts и 180 source files
сверены по hashes. Все четыре коротких trainings, selected checkpoints,
CUDA RNG replay и real-kNN batch invariance прошли. Полная независимая
проверка квитанций сохранена в `current/validated_smoke.json`;
`data_manifest.json` SHA-256:
`1722523f8708a10d3b6eac43c303fd4780ea31e4df671daf25d37ca69c7a7c29`.

После проверки отсутствия active/completed duplicate tasks отправлен
replacement array `4413514` в 13:25:28 MSK: прежние 12 tasks `0-11%2`,
четыре recipes × seeds 42/123/456, один V100, 8 CPU, до 5 часов на task.
Исполняется read-only `7dcd849c18`; recipes и scientific estimator не менялись.
Exact argv, task mapping и номер записаны сразу remotely/locally в
`current/submissions.json`. Test jobs не отправлялись. Второй автоматический
retry не предусмотрен. Научных development results пока нет.

## Первые два полных runs проверены, 2026-10-17 15:06 MSK

Tasks 0/1 (`reference_42`, `layernorm_42`) COMPLETED, exit 0, 30 epochs
каждый. Проверены все component histories, неизменные per-head denominators,
LR used/next, effective configs и population identity. Checkpoint hashes
совпали с complete receipts; selected epochs 6 и 2 (zero-based) совпали
с первыми минимумами population composite validation loss.

На reused validation 988974 (101107 clusters, из них 16877 signal), при
прежних train-fitted scalers, AUC reference = 0.9132589313 и LayerNorm =
0.8551834758. Это два разных checkpoint при одном training seed 42;
uncertainty по training seeds ещё не оценена. На этих partial results
recipes не меняются и вывод о сравнении не формулируется. Остальные десять
tasks остаются в той же очереди; новые задачи не отправлялись.

Первые grid thresholds с count purity >=0.70 равны 0.32036 и 0.522455;
полные 401-point scans сохранены. Receipt с hashes исходных JSON/CSV,
histories, checkpoint и prediction files — `current/completed_artifacts_verified.json`.
Сравнение по трём seeds и обновление manuscript отложены до всех 12 runs.

## RA-09, OPEN: checkpoint I/O failure, 2026-10-17 16:26 MSK

Task `4413514_2` (`pool_42`) FAILED, exit 1, после 52:31 на cn-008.
16 эпох (0–15) сохранены. На следующей записи `_write_resume_state`
`torch.save` сообщил `Parent directory .../pool_42/checkpoints does not exist`.
Обработчик ошибки затем получил `PermissionError: EACCES` при `out.is_dir()`
и не обновил run status. Поэтому `status.json` остался `training`; реальное
состояние определяется Slurm и traceback. Остальные задачи продолжаются;
конфигурации и snapshots не менялись.

При последующей проверке с login node каталог существует, mode 0755,
владелец — тот же uid 2761, чтение/запись доступны. Наблюдения совместимы
с временной недоступностью Lustre; конкретная инфраструктурная причина
не установлена. Это сбой I/O при сохранении, не доказанный numerical failure
pooling. В отличие от dataset reads, checkpoint writes и failure reporting
не имеют bounded filesystem retry. Это открытое замечание P2; новый source
или автоматический повтор в действующую серию не добавлялись.

Atomic `resume.pt` прочитан на CPU и согласован: `next_epoch=16`, история
из 16 эпох, best epoch 5, model/optimizer/scheduler и Python/numpy/torch/CUDA
RNG присутствуют, model tensors finite. `last.pt` уже содержит epoch 16,
отдельный `optim.pt` — next epoch 17. Продолжать можно только из atomic
resume; объединять эти отдельные более поздние файлы с ним нельзя.

Файлы и hashes сохранены в `current/pool_42_resume_inspection.json` и
`pool_42_failure_filesystem_inspection.json`. Подготовлена отдельная копия
состояния и предложение одного продолжения оставшихся 14 epochs при прежнем
общем бюджете 30, в том же immutable source. Команда из
`current/pool_42_resume_proposal.json` не отправлена: текущие heartbeat
instructions явно запрещают дальнейший automatic retry. Предлагаемая
зависимость от окончания исходного массива сохраняет campaign limit не
более двух одновременных задач. Требуется отдельное разрешение пользователя.
Неполный `pool_42` не включается как завершённый seed в AUC comparison.

## Low-LR seed 42 проверен, 2026-10-17 16:45 MSK

`4413514_3` COMPLETED, exit 0, 30 epochs. Actual initial LR = 1e-4;
остальные model kwargs совпали с reference, parameters = 12479173.
Проверены все per-head counts, LR transitions, finite component losses и
выбор checkpoint: epoch 2 (zero-based), первый минимум population validation
loss. Checkpoint/prediction hashes и полная проверка scan добавлены в
`current/completed_artifacts_verified.json`.

На том же reused validation 988974 (101107 clusters, 16877 signal), с прежними
train-fitted scalers, AUC = 0.8722491185, threshold = 0.552395, count purity =
0.7005837522, signal efficiency = 0.5119985779. Это один training seed 42;
training-seed uncertainty и итоговое сравнение пока не оценены. Теперь три
полных runs проверены; partial pooling run остаётся вне comparison. Новых
failures у остальных tasks не обнаружено, разрешение на pool_42 resume ещё
не получено.

## Второй I/O failure и исключение узла, 2026-10-17 17:06 MSK

Task `4413514_4` (`reference_123`) FAILED, exit 1, 41:18, также на cn-008.
На этот раз `torch.save` создал temporary resume, но `os.replace` получил
EACCES; failure reporting снова не смог прочитать run directory. Это тот же
открытый механизм RA-09 в другом recipe. Связь двух failures с одним узлом
наблюдается, но инфраструктурная причина не доказана.

Atomic `resume.pt` прочитан с `next_epoch=13`: 13 committed epochs, best
epoch 5, finite model, optimizer/scheduler и все четыре RNG states согласованы.
Отдельные last/optim опережают его на одну эпоху. Сохранена проверенная копия
всего состояния, включая оставшийся temporary file, 499582906 bytes;
temporary file не повышен до committed resume. Диагностика и предложение
продолжения оставшихся 17 epochs — `current/reference_123_resume_inspection.json`
и `reference_123_resume_proposal.json`. Предложение не отправлено.

Только для ещё pending tasks 7–11 исключён cn-008 через Slurm placement
update. Сверено, что account/partition/features/CPU/GPU/time и command не
изменились. Source, data, recipes и throttle 2 сохранены; новых jobs нет.
Уже работающие tasks 5/6 не остановлены и не перемещены. Это операционная
мера по двум I/O failures, без отбора по classifier quality; healthy задачи
продолжаются. Before/after квитанции — `current/placement_exclusion_20261017.json`.

Три 30-epoch runs остаются проверенными. `pool_42` и `reference_123` нельзя
считать завершёнными seeds; без их разрешённого восстановления недоступны
полные three-seed primary LayerNorm и exploratory pooling comparisons.
Дальнейшие automatic retries не отправляются; ранее поставленный вопрос
о разрешении остаётся открытым и не повторяется при каждом heartbeat.

## Семь полных runs проверены, 2026-10-17 20:25 MSK

Новые completed tasks 5–8: `layernorm_123`, `pool_123`, `low_lr_123`,
`reference_456`. Для каждого сверены все 30 epochs, per-head counts, LR,
effective model/config, population identity, minimum-loss checkpoint и
его hash, full 401-point threshold scan. Выбранные epochs (zero-based):
2, 8, 2, 1 соответственно. Полные значения и input hashes добавлены в
`current/completed_artifacts_verified.json`; fetched JSON/CSV совпали с ними.

На том же reused validation 988974 (101107 clusters, 16877 signal), с теми же
train-fitted scalers, per-seed AUC соответственно 0.8628539402, 0.8781291648,
0.8632810987, 0.9049998582. Каждое значение — один training seed, SE ещё
не оценён. Все четыре достигли conventional validation count purity 0.70
на исходной grid; thresholds сохранены без перенастройки.

Итого семь 30-epoch runs verified, два прежних I/O failures, две задачи
продолжают обучение и одна ожидает. Новых failures нет; ни одного повторного
job не отправлено. Даже успешный `pool_123` на cn-008 не доказывает причину
двух прежних failures; exclusion остаётся операционной предосторожностью.
Final paired comparison и manuscript update по development campaign пока
отложены; неполные seeds не подменяются их ранними checkpoints.

## Девять runs и уточнение manuscript, 2026-10-17 21:10 MSK

`layernorm_456` и `pool_456` COMPLETED, exit 0, 30 epochs; оба полностью
проверены по прежнему протоколу и добавлены в completed-artifact receipt.
Selected epochs (zero-based) 1 и 10; reused-validation AUC 0.8580934338
и 0.9049071998, thresholds 0.57485 и 0.47505. Population/scalers те же;
каждое число — отдельный training seed, без заявления resolved gain.
Все три LayerNorm seeds завершены, но primary paired comparison требует
также отсутствующий complete reference_123. Итого verified 9 runs,
два прежних failures и последний low_lr_456 в работе. Повторов нет.

В Methods manuscript уточнён фактический исторический selector completed
DCM diagnostics и source-run comparison: unweighted mean of minibatch
composite validation losses. Он подтверждён в обоих исполненных snapshots;
evidence с hashes сохранено в `HGNDPaper/data/historical_validation_selector_20261017.json`.
Эффект смены selector отдельно не измерен, прежние AUC/weights сохранены.
Development numbers из незакрытой серии в manuscript не переносились.
Saved-source PDF пересобран, 16 страниц; все изменённые страницы 4–13
просмотрены, остальные pixel-identical прежней проверенной версии.

## Разрешённый бюджет закрыт с двумя incomplete runs, 2026-10-17 23:53 MSK

Последний task `4413514_11` (`low_lr_456`) COMPLETED, exit 0, 1:19:42.
Все 30 epochs, head counts, used/next LR, actual kwargs, population identity,
selected checkpoint hash и полный threshold scan проверены. Selected epoch 4
(zero-based), reused-validation AUC 0.8549932735, threshold 0.407685,
count purity 0.7006890 и signal efficiency 0.4519168 на 101107 clusters
(16877 signal). Это один training seed; per-seed sampling uncertainty
не оценена. Ни одного нового test evaluation или повторного job нет.

Теперь все 12 original tasks terminal: 10 complete по 30 epochs, два прежних
I/O failures с coherent atomic resumes. Резервные копии и неподанные proposals
сохранены. Разрешение на дополнительные продолжения не получено; они не
выполнялись. RA-09 остаётся OPEN, node/filesystem root cause не доказана.

Canonical closure — `results/dcm_development_20261017/closure/summary.json`,
полная таблица — `closure/per_seed.csv`, отчёт —
[`dcm_development_results_20261017.md`](dcm_development_results_20261017.md).
Все 12 исходов сохранены, failed runs не подменены ранними checkpoints.
Из-за missing reference_123 full primary LayerNorm и secondary low-LR
contrasts недоступны; pooling дополнительно требует pool_42. Mean/SE/t
по preregistered трём seeds оставлены null. Descriptive differences только
доступных отдельных пар записаны без aggregate significance claim.

Сохранены per-head histories всех 12 runs и полные scans для 10 complete
models. Пустые panels двух failed runs обозначены; undefined purity при
нуле selected clusters остаётся gap, а не искусственным нулём. Generated
figures просмотрены; input hashes и финальные проверки записаны в `closure/validation.json`.
Результаты conditional на reused validation 988974; independent test и
between-source variance не измерены. Development numbers в manuscript
не добавлены; проверенный PDF с historical-selector correction сохранён.

Финальная проверка: 107 canonical input hashes, 180 read-only execution
files, 12 history и 4 threshold figures; все 16 figures просмотрены,
векторные PDF содержат embedded TrueType. Strict table lint — 0 проблем
в results/protocol/audit/correction log. 45 input hashes старого grouped
result и source/PDF manuscript совпали с прежними. Heartbeat
`finish-hgnd-note-and-paper` переведён в PAUSED после этих проверок;
два неподанных recovery proposals остаются на решение пользователя.
