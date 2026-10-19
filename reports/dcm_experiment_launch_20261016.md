# Запуски DCM: проверка кластера и новая диагностическая серия

Дата: 2026-10-16, Europe/Moscow. Проверка выполнена через SSH на cHARISMa:
`squeue`, `sacct`, логи заданий и наличие выходных файлов. Перед новыми
отправками активных заданий пользователя не было.

## Что уже выполнено

- `4366196`, `train_dcm`: COMPLETED, exit 0, 2026-09-30 20:24:51–22:04:59 MSK.
  Все 25 эпох, batch 256, H=512, L=8, seed в аргументах 42, `save_best_on=val`.
  Лучшая эпоха 23, validation loss 0.9827832560246602.
- `4366562`, `eval_dcm`: COMPLETED, exit 0, 2026-09-30 22:26:26–22:37:21 MSK.
  Оценён именно checkpoint предыдущего задания. ROC AUC 0.9515279254854021
  на 241259 кластерах из того же графового validation-разбиения, на котором
  выбиралась эпоха. Это историческая оценка, не независимый test и не новое
  измерение в этом отчёте.
- `4367897`, energy flattening: COMPLETED, exit 0, 2026-10-01.
  Оценка `4368122` также COMPLETED, exit 0.
- `4368152`, capped energy flattening: COMPLETED, exit 0, 2026-10-01.
  Checkpoint лежит в `/scratch/vbocharnikov/hgnd/checkpoints/dcm_eflat_capped`.
  Отдельная оценка этого checkpoint среди найденных DCM-артефактов отсутствовала.
- Заданий новой серии `train_dcm_improve` в проверенной истории нет.

Предположение прежнего аудита, что job 4366196 остался на эпохе 2 и его
результат не оценивался, опровергнуто журналом кластера. Дата раннего результата
совпадает с датой этого завершённого запуска. Сравнение с опубликованными
80%/87% всё ещё требует согласования метрик, отбора и источника эталона.

Копии исторических stdout и baseline performance JSON сохранены в
[`results/dcm_diagnostic_20261016/history`](../results/dcm_diagnostic_20261016/history).

## Код, данные и ограничения новой серии

Рабочая копия на кластере находится на старом commit `1ea2683` и содержит
локальные правки. Она сохранена. Новые задания используют отдельный снимок:

```text
source parent: /home/vbocharnikov/hgnd_experiments/dcm_diag_20261016_c36680aba5
campaign: /scratch/vbocharnikov/hgnd/experiments/dcm_diag_20261016_c36680aba5
source base: 53ff00d45305618e91cdd12b670ebd80a4de1dfc
source digest: c36680aba54e47b917e8a56df9ed62f9d2291491ba7901695a71c3d7131a9667
data: /scratch/vbocharnikov/hgnd/cache/ndet_dataset_dcm_xecs_own_tw1p0
```

Снимок включает исправление DCM-06: Python, NumPy и PyTorch получают seed
до создания модели. Проверка CLI подтверждает одинаковые начальные веса
при одинаковом seed и разные при изменённом. Регрессия проверяет реальный
момент вызова фабрики; она не утверждает побитовую детерминированность CUDA.
Исправления фабрики и предупреждения scheduler уже находятся в `53ff00d`.

Используется один существующий cache: 283398 графов, 277 shards, schema v2,
окно 1.0 ns, 10 признаков. Для `net_default` достаточно его hits/clusters;
все 277 shards указаны явно, чтобы не запускать перестроение в schema v3.
Исходные кэши и случайное размытие времени не пересоздаются.

Это диагностическая серия с одним training seed 42 на прежнем графовом
разбиении 50/50. Выбор эпохи и порога по validation сохраняется. DCM-04–05
остаются открытыми: независимого test нет, top/bottom одного события могут
попасть в разные части. Новая серия не подтверждает опубликованный benchmark
и не заменяет будущий протокол с разбиением по событиям и независимым test.

Для контроля исправленной инициализации предусмотрен новый `ref`; он
отличается от исторического запуска тем, что seed фиксирует и начальные веса.
Далее идут `epochs`, `sched`, `warmup`, `size`, `pool`, `residual` из
`train_dcm_improve.hpc.sbatch`. `budget` не отправляется: его конфигурация
совпадает с `warmup`. Настройки заимствованы из проверяемого плана экспериментов;
их оптимальность этим запуском заранее не предполагается.

Новые задания пишут в отдельные каталоги. Повторное использование частично
записанных optimizer/model запрещено защитой `started/`: DCM-09 ещё открыт.
При отказе или timeout требуется проверить согласованность файлов перед
продолжением. Полный stdout сохраняет историю эпох, `effective_run.json`
содержит параметры запуска, `source_manifest.json` содержит хеши файлов.

## Проверки перед отправкой

- CLI и scheduler: 37 тестов прошли, включая проверку seed до фабрики.
- Синтаксис shell, компиляция двух Python entry points, `git diff --check`:
  успешно.
- Проверка CUDA через `srun`, job `4411318`, `test`, V100: успешно,
  PyTorch 2.5.1+cu121, Tesla V100-SXM2-32GB.
- `test` допускает максимум 10 CPU на узел; первая отправка smoke с 16 CPU
  отвергнута без создания задания. Повторная отправка использует 4 CPU.
- Smoke job `4411325`: COMPLETED, exit 0, 3 минуты 52 секунды на `test`
  при лимите 5 минут. Все семь вариантов с H=512/L=8/batch=256 прошли
  настоящий kNN, проверку finite loss/gradients, шаг Adam, сохранение и загрузку
  checkpoint и проверку совпадения предсказаний. Загрузка окружения занимала
  большую часть времени до появления вывода модели.

## Текущий статус отправок

Полные задания приняты Slurm после успешного smoke. Последняя проверка:
2026-10-16 14:40:55 MSK. Все восемь заданий находились в `PENDING`, причина
`Priority`. Это ожидание очереди; фактическое обучение ещё не началось.

Массив `4411340`, максимум два одновременно выполняющихся обучения:

- `4411340_0`: `ref`, 25 эпох, MultiStepLR, контроль фиксированной инициализации.
- `4411340_1`: `epochs`, 30 эпох, MultiStepLR.
- `4411340_2`: `sched`, 30 эпох, cosine без warmup.
- `4411340_3`: `warmup`, 30 эпох, cosine и три эпохи warmup.
- `4411340_4`: `size`, предыдущие настройки и размер кластера.
- `4411340_5`: `pool`, дополнительно mean/max/sum pooling.
- `4411340_6`: `residual`, дополнительно residual в SAGE-стеке.

Каждому заданию выделяется один V100, 16 CPU и лимит 8 часов. Лимит учитывает
более медленный V100 по сравнению с историческим A100 и расширенную
архитектуру `pool`; это ограничение выполнения, не критерий сходимости.
После обучения каждый вариант сам записывает отдельную оценку validation.
Файлы находятся в `$campaign/<variant>/checkpoints` и
`$campaign/<variant>/validation`, stdout/stderr в `$campaign/logs`.

Отдельное задание `4411339`, `dcm_eval_capped`: оценка ранее обученного
`dcm_eflat_capped/model.pt`. Один V100, 8 CPU, лимит один час; историческая
оценка обычного baseline заняла около 11 минут на A100. Каталог результатов:
`$campaign/eflat_capped_validation`. Это дополнительное задание оценки,
вне ограничения массива на два одновременно выполняющихся обучения.

Полная квитанция отправки, снимок очереди и результаты smoke сохранены в
[`launch_status.json`](../results/dcm_diagnostic_20261016/launch_status.json).
В scratch лежат также `submissions.json` и `launch_status.json`. Перед
повторной отправкой нужно сверять эти job IDs с Slurm, чтобы не создавать
дубликаты. Успешную отправку и успешное завершение следует учитывать отдельно.

## Проверка 2026-10-17 и независимый follow-up

По состоянию на 00:30 MSK `4411339` завершён; capped flattening дал reused-
validation AUC 0.82219 и не выбран для дальнейшего улучшения. Диагностические
arms `ref`, `epochs`, `sched`, `warmup` завершились с AUC соответственно
0.9514432, 0.9539114, 0.9580322, 0.9558959 на тех же 241259 clusters, seed 42.
Это single-seed validation point estimates, без training-seed uncertainty;
разрешённое различие или generalisation этим сравнением не установлены.
`size` выполняется, `pool` и `residual` ожидают лимита массива. Точные файлы и
логи сохранены в `results/dcm_diagnostic_20261016/current/`.

Вместо повторной оптимизации по этому validation отправлен отдельный
source-run holdout: шесть matched-seed обучений `4412700` и зависимая test
оценка `4412701`. GPU smoke `4412696` успешно завершился перед отправкой.
Протокол и ограничения: `reports/dcm_grouped_protocol_20261017.md`.

## Проверка 2026-10-17, 01:02 MSK: признак размера кластера

`4411340_4` (`size`) завершился успешно, exit 0, за 2:43:54. Все 30 эпох
выполнены; для оценки загружен checkpoint эпохи 22 (отсчёт от нуля), выбранный
по validation loss. Журнал загрузки подтверждает H=512, L=8 и реальное
`cluster_size_feature=True`; cosine и три эпохи warmup совпадают с контролем
`warmup`. Snapshot, cache, batch 256 и training seed 42 сохранены.

На повторно используемой graph-level validation из 241259 clusters AUC
составил 0.9549973. У непосредственно предшествующего контроля `warmup`
AUC 0.9558959; разность `size − warmup` равна −0.0008986. Это разные веса и
только один matched training seed: SE и t на training seeds отсутствуют,
преимущество или ухудшение статистически не установлены. Независимый test
этой серией не оценивается, scaler fitted на всём frozen cache.

Рабочий threshold 0.65 выбран на той же validation по близости к 0.87
energy-weighted purity: count purity 0.84826, count efficiency 0.61730,
energy-weighted purity 0.87347. Эти величины не являются event-level
benchmark при count purity 0.70. Правило выбора порога унаследовано для
сопоставимости diagnostic ladder и не объявляется optimum.

Проверены completed marker, effective config, фактические kwargs загруженного
checkpoint, все threshold counts и их общий знаменатель 40225 signal clusters.
Файлы и hashes сохранены в
`results/dcm_diagnostic_20261016/current/validated_controls.json` и
`current/size/`. В рукописи пока остаются четыре сравнения расписаний;
архитектурная серия будет сведена после завершения `pool` и `residual`.

`pool` и `residual` выполняются без обнаруженных ошибок. Все шесть задач
`4412700` ожидают Priority, `4412701` ожидает Dependency. Reserved source-run
test не открыт; рецепты не менялись и повторных отправок не было.

## Проверка 2026-10-17, 03:02 MSK: mean/max/sum pooling

`4411340_5` (`pool`) завершился успешно, exit 0, за 2:28:07. Выполнены все
30 эпох. Оценка загрузила выбранный по validation loss checkpoint эпохи 24
(отсчёт от нуля) с фактическими `cluster_pooling=mean_max_sum` и
`cluster_size_feature=True`, H=512/L=8. Сохранились cosine, три эпохи warmup,
batch 256, seed 42 и тот же frozen cache.

Cluster AUC на повторно используемой validation из 241259 clusters равен
0.9620253; у непосредственного контроля `size` — 0.9549973. Point difference
`pool − size` составляет +0.0070280. Это разные веса с одним matched training
seed; SE среднего и t по training seeds не оценены. Статистически разрешённое
улучшение и перенос на независимый source run не установлены. Параметров
16677573 против 12483269 у `size`: изменение pooling одновременно увеличивает
ёмкость сети, поэтому приписать весь сдвиг одной агрегации нельзя.

Validation-selected threshold 0.55 по прежнему правилу близости к 0.87
energy-weighted purity даёт count purity 0.84588 и count efficiency 0.68674
(energy-weighted purity 0.87239). Эти point estimates используют ту же
validation и не являются event-level benchmark при count purity 0.70.
Проверены 40225 signal clusters в знаменателе, threshold counts, completed
marker, effective config и kwargs загруженного checkpoint. SHA-256 весов:
`a67da0e63c6f43669c2f5a9d6d598b32917644d423d73489bd3f0dc6031976a4`.
Числа, hashes и ограничения внесены в `current/validated_controls.json`.

`residual` продолжает обучение без обнаруженных ошибок. Первые два обучения
независимой серии (`4412700_0`, `4412700_1`) выполняются; остальные четыре
ожидают лимита массива. Все test tasks ожидают Dependency. Протокол этой
серии не менялся по результату pooling; новый test не открыт. Рукопись будет
обновлена полной архитектурной серией после завершения `residual`.

## Проверка 2026-10-17, 05:08 MSK: диагностическая серия завершена

Все семь задач `4411340` завершены, exit 0. `residual` выполнил 30 эпох
и оценил validation-selected checkpoint эпохи 24 (отсчёт от нуля); AUC
0.9606160 на той же reused graph-level validation из 241259 clusters.
Фактические kwargs включают mean/max/sum, cluster size и `hit_residual=True`.
Параметров 16677573, как у `pool`. Point difference `residual − pool`
равен −0.0014093. Это один matched training seed, SE/t отсутствуют;
статистически разрешённое ухудшение или улучшение не установлено.

Все counts и общий знаменатель 40225 signal clusters проверены. На score
threshold 0.55, выбранном по близости к 0.87 energy-weighted purity на той же
validation, count purity равна 0.84309, efficiency 0.68259. Независимый test
не оценивался. SHA-256 всех семи выбранных checkpoints записаны в
`current/checkpoint_hashes.json`; полный проверенный итог —
`current/validated_controls.json`.

Таблица II рукописи теперь содержит все семь recipes, эффективные parameter
counts и явное cumulative определение трёх архитектурных rows. Abstract,
Methods, Results и Summary согласованы с диапазоном single-seed validation
AUC 0.9514–0.9620. Рост capacity при pooling отмечен; результат не объявлен
pure pooling gain, multi-seed systematic или independent generalisation.
Canonical manuscript JSON включает реальные checkpoint kwargs, выбранные
эпохи и hashes weights, performance files, configs и журналов.

В source-run серии завершены `multistep/42` и `cosine/42` (по 30 эпох);
выбраны эпохи 3 и 5. Их manifests и frozen validation thresholds проверены.
Задачи seed 123 выполняются, seed 456 ожидают лимита массива. Test остаётся
закрытым до окончания всех шести обучений; прежний протокол сохранён.

## Проверка 2026-10-17, 07:20 MSK: шесть обучений завершены

Все задачи `4412700` завершены, exit 0. Проверены все 30 эпох каждого
обучения, общий split manifest, hashes выбранных checkpoints и замороженные
validation thresholds по count purity 0.70. Итоговая проверка квитанций —
`results/dcm_grouped_20261017/current/completed_validation_checks.json`.
Зависимость `afterok:4412700` снята; первые две задачи `4412701` выполняют
test, остальные четыре ожидают лимита `%2`. До завершения всех шести оценок
primary comparison не рассчитывается; настройки остаются прежними.

## Проверка 2026-10-17, 08:00 MSK: test завершён

Все задачи `4412701` завершены, exit 0; все зарегистрированные DCM jobs
теперь completed. Каждый test содержит 222251 clusters / 36713 signal
clusters source run 988976. Checkpoint hashes, split и неизменность frozen
validation thresholds проверены. Canonical comparison находится в
`results/dcm_grouped_20261017/summary.json`, разбор —
`reports/dcm_grouped_results_20261017.md`. Paired cosine-minus-MultiStepLR
AUC +0.0029351 ± 0.0192332 (SE среднего по трём training seeds), t=0.1526
на 2 df; преимущество не разрешено. Один test source run не измеряет
between-generation variance. Статья дополнена результатом; новые рецепты
по открытому test не выбирались. Ранее завершённые diagnostic и capped
experiments не перезапускаются.
