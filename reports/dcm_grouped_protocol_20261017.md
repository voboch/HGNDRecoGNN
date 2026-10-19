# DCM: независимый source-run контроль расписания обучения

План зафиксирован до оценки нового test. Пользователь 2026-10-17 поручил
продолжать эксперименты и исправить статью по аудиту. Это отдельная серия;
старые checkpoints и рабочая копия кластера не изменяются.

## Вопрос и первичная величина

Меняет ли cosine относительно MultiStepLR cluster ROC AUC на новом исходном
запуске DCM при одинаковых данных, архитектуре, бюджете и training seeds?
Первичная величина — test cluster AUC, безразмерная, больше лучше. Вторичные
величины — count purity/efficiency и energy residual/resolution на замороженном
validation-порогe; они не заменяют event estimator опубликованного benchmark.

До открытия test ожидается положительный эффект cosine порядка 0.004 AUC,
как в диагностике (0.9580−0.9539). Это ориентир для проверки: перенос на новый
source run и train-only scaling может уменьшить эффект до нуля или сменить
знак. Нулевое/неразрешённое различие публикуется без изменения протокола.

Исторические диагностические AUC 0.9539 и 0.9580 получены на повторно
используемой graph-level validation, один training seed, одинаковый бюджет
30 эпох. Они служат мотивацией проверки, а не независимым подтверждением.

## Данные и изоляция

- Замороженный DCM cache: 283398 arm graphs, 277 shards, окно 1.0 ns.
- Raw export: `/home/vbocharnikov/het_nDetHalf/data/ndet_links_Jan_26`.
  Доступны ровно три исходных каталога: `988974`, `988975`, `988976`.
  Они содержат соответственно 134, 198 и 287 hit files до проверки пригодности.
- Три целых source-run каталога переставляются `random.Random(42)` и назначаются
  train, validation и test по одному. Split seed фиксирован отдельно от
  training seed. Реальные размеры разделов сохраняются после чтения cache.
- Все файлы одного source run и обе половины одного physical event остаются
  вместе. Проверяется полное покрытие индексов и отсутствие пересечений Row.
- Старое affine scaling обращается по сохранённым scaler parameters. Новые
  StandardScaler обучаются только на восстановленных float32 train features;
  validation/test на fit не влияют. Графы, timing noise, labels и edges не
  перестраиваются. При выдаче граф клонируется: повторный affine remap не
  меняет общий cached объект.

Один source run на test обеспечивает разрыв с train, но не позволяет оценить
межзапусковую test variance. Seeds измеряют вариацию обучения на фиксированном
test, а не вариацию новых MC productions. Это ограничение переносится в статью.

## Контролируемые изменения

Обе arms: `net_default`, H=512, num_layers=8, Adam, lr=1e-3,
weight decay=1e-5, batch 256, 30 эпох, одинаковые loss weights, без warmup.
Бюджет 30 эпох унаследован от диагностической серии и не объявляется
оптимальным или достаточным для сходимости. Architecture и graph settings
фиксируют воспроизводимый reference recipe; этот опыт их не оптимизирует.

MultiStepLR использует milestones 10/15/25 и gamma 0.1; cosine — тот же
начальный lr с плавным снижением за весь бюджет. Training seeds: 42, 123, 456.
Каждая пара arms использует один и тот же seed, train/validation/test и scalers.
Checkpoint выбирается только по validation loss. Atomic resume поддерживается;
готовый запуск повторно не обучается.

Порог — первое значение сетки 401 точек от 0.001 до 0.999, достигающее count
purity 0.70 на validation. Purity 0.70 — унаследованная опубликованная convention,
а не найденный optimum; сохранены весь scan и его PDF. Test использует этот
порог без подгонки. Test phase отказывается работать до завершения всех шести
заранее заданных обучений и отказывается открывать уже оценённый test повторно.

## Сравнение и ограничения

Сравнение двух наборов весов — case 1 общей методологии: paired differences
test AUC по трём training seeds, SE среднего и t на 2 df; критическое значение
4.303 для двустороннего уровня 0.05. Если отличие не разрешено, так и сообщить.
Не подменять этот SE кластерным bootstrap или evaluation repeat floor.
Не выбирать новую архитектуру, окно, threshold или arm по открытому test.

Вторичные energy metrics сохраняют выбранную signal-cluster population;
ошибка не включает нейтроны, не образовавшие кластера. Для физического спектра
или multiplicity нейтронных tracks нужна отдельная response calibration.

## Снимок и проверки

Текущий immutable source:
`/home/vbocharnikov/hgnd_experiments/dcm_grouped_20261017_767a0e6f2e`.
Source SHA-256:
`767a0e6f2ebe1a1c443bd69ec23ed2e6847f1537019722e34eda9a803c09b07b`.
Base commit: `046cee2b5203c0e6d0f98521efecb8abee62923d`.
Campaign сохраняет исходное имя:
`/scratch/vbocharnikov/hgnd/experiments/dcm_grouped_20261016_077a236ca2`.

Первый test-partition job `4412667` успешно подготовил все индексы и scalers.
После появления atomic manifest он был остановлен: повторное полное чтение
277 shards для каждой короткой smoke arm расходовало лимит времени на I/O.
Снимок R2 меняет только исполнение smoke: подготовка повторно не выполняется,
читаются нужные shards, оба коротких опыта проходят в одном interpreter.
Протокол, графы, индексы и scaler hashes сохранены. Исходный R1 snapshot
`dcm_grouped_20261016_077a236ca2` сохранён для происхождения preparation.

Замороженное назначение: train `988975` — 92252 arm graphs / 64138 physical
events; validation `988974` — 59973 / 41808; test `988976` — 131173 / 91081.
Manifest SHA-256:
`1ab9285e3315dc0f043b7d5116255f1459978477dc9a9b4d95e046649163278b`.
Точные hashes индексов и scaler files хранятся в manifest.

Локально прошли четыре теста: source-run coverage, grouping обеих половин,
отказ от неоднозначного/неполного manifest и отсутствие мутации cached графа
при повторном remap. Проверены Python compilation и shell syntax.

GPU smoke `4412696`: COMPLETED, exit 0, 1 минута 9 секунд, partition `test`,
один V100 и 4 CPU. Обе arms прошли два training batches, validation, запись
и загрузку выбранного checkpoint и inference. Smoke не оценивает classifier
quality: после двух updates фиксирован threshold 0.0; reserved test не открыт.

## Отправленная серия

2026-10-17 00:30 MSK после успешного smoke приняты:

- `4412700`, train array `0-5%2`: multistep/42, cosine/42, multistep/123,
  cosine/123, multistep/456, cosine/456. Один V100, 8 CPU, до 8 часов на задачу.
- `4412701`, test array `0-5%2`, dependency `afterok:4412700`: те же шесть
  наборов весов, один V100, 8 CPU, до 1 часа 30 минут на оценку.

Dependency и программная проверка шести training_complete files независимо
защищают test от преждевременного открытия. Полные квитанции лежат в campaign
`submissions.json` и локально в `results/dcm_grouped_20261017/current/`.
Текущее состояние — `results/dcm_grouped_20261017/launch_status.json`.
Перед любым retry сверять эти IDs с Slurm; активные и completed jobs не
дублировать. Новые численные результаты в статью пока не перенесены.

## Прогресс 2026-10-17, 05:08 MSK

`multistep/42` и `cosine/42` завершили по 30 эпох и validation inference
(COMPLETED, exit 0). Выбраны эпохи 3 и 5 (отсчёт от нуля); checkpoint hashes,
split manifest и сохранённые validation thresholds проверены. Оценено по
101107 validation clusters. Seed 123 выполняется, seed 456 ожидает лимита
массива. Все test tasks ожидают Dependency; reserved test не открыт.
Новые результаты pooling/residual из отдельной диагностической серии
не изменили recipes или primary comparison этого опыта.

## Проверка 2026-10-17, 07:20 MSK: обучение завершено

Все шесть задач `4412700` завершены, exit 0; журналы содержат все 30 эпох.
Для seed 456 выбраны эпохи 3 (`multistep`) и 4 (`cosine`), от нуля.
Проверены квитанции всех шести обучений: общий split manifest, hashes
выбранных checkpoints, 101107 validation clusters и первый из 401 score
thresholds, достигший count purity 0.70. Пороги seed 456 равны 0.42016 и
0.3503; остальные четыре порога сохранены без изменений. Проверка с hashes
всех входных файлов записана в `current/completed_validation_checks.json`.

Массив `4412701` начал test автоматически после успешного завершения всех
шести обучений. Первые две задачи выполняются, четыре ожидают лимита `%2`.
Test metrics пока отсутствуют. Recipes и правило выбора порогов не менялись;
сравнение AUC будет рассчитано после завершения всех шести test tasks.

## Итог 2026-10-17, 08:00 MSK

Все шесть test tasks `4412701` завершены, exit 0. Проверены квитанции,
неизменность selected checkpoint hashes и validation thresholds, общий test
source run 988976 и его counts. Canonical result —
`results/dcm_grouped_20261017/summary.json`; полный итог с per-seed значениями
и ограничениями — `reports/dcm_grouped_results_20261017.md`.
Paired cosine-minus-MultiStepLR AUC +0.0029351 ± 0.0192332 (SE по трём
training seeds), t=0.1526 на 2 df, critical 4.303. Преимущество не разрешено;
SE не включает between-generation variance. Протокол после открытия test
не менялся. Результаты, пороги и validation scan внесены в рукопись.
