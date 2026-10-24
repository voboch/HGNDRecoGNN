# Corrected-SMASH: source binding и accounting

Текущее обновление 24 октября: полный frozen transfer завершён.
zeroSpot4437674 COMPLETED/exit0 за15мин53с, defaultSpot4437675 за11мин29с,
bigSpot4437676 за3мин50с. Проверены все1296 prediction CSV, archives и полный
ledger, метрики независимо пересчитаны. Оба графика прошли bounds и visual QA.
Активных campaign jobs нет; закрыт конечный gate audit/cluster transfer.
Числа, n, training-seed SD и пределы: `smash_full_transfer_results_20261024.md`.

Все три полных CPU audits также завершились COMPLETED/exit0: zeroSpot4437503
за19мин33с, defaultSpot4437504 за19мин01с, bigSpot4437505 за2мин50с.
Source binding, input/primary hashes, полное membership, точные kinematics,
geometry и accounting проверены. Test CUDA smoke4437645 пройден ранее.

Источник и frozen recipe сохранены. SMASH training не выполняется.
История отказов ниже сохраняется; Lustre не объявляется исправленной.

## Полные аудиты

Canonical population содержит432 уникальных непустых detector pairs,
100 header-only pairs и36 точных aliases в568 исходных members. Для всех
524496 eligible direct-primary n/p records совпадают Row/Id, PDG, initial
energy/momentum и vertex. После canonicalization source tracks/events не
повторяются среди matched records внутри каждого potential.
Production-job independence не установлена.

При audit graph r3.6/time1.5ns результаты ниже описывают MC label census,
без model scoring и без fitting. Большее число сигналов само по себе не
означает лучшее качество. Model transfer использует прежний time1.0ns;
его candidate counts будут отличаться.
Scored on: frozen corrected-SMASH common-file development population;
fitted on: nothing. No direction is better for these counts.

| Potential | Audit clusters (count) | Native MC-positive clusters (count) | Relaxed MC-positive clusters (count) | Added soft clusters (count) |
|---|---:|---:|---:|---:|
| zeroSpot | 1194556 | 217215 | 220881 | 3666 |
| defaultSpot | 1135419 | 207007 | 210535 | 3528 |
| bigSpot | 315356 | 57662 | 58644 | 982 |

Relaxed definition убирает только прежний Ekin>0.3GeV floor на том же
audit graph и jitter. Energy targets прежних native-positive clusters не
изменились. Эти counts не измеряют изменение ROC-AUC или чувствительности
к симметрийному потенциалу. Число обработанных ROOT events, включая
zero-response events, остаётся null. Проверяемый источник:
`results/smash_full_transfer_20261023/full_audit_validation.json`.

## GPU smoke и полный перенос

Smoke4437645 проверил полные audit bundles всех potentials на compute node,
после чего выполнил inference первого файла каждого potential на CUDA.
Original observation hashes, candidates, labels, energy targets, checkpoints,
frozen cuts, AUC и selected sets совпали с first-file probe4435668.
Числа candidates/MC positives:6233/1116 zeroSpot,1611/329 defaultSpot,
6302/1137 bigSpot, по тем же трём DCM training seeds42/123/456.

Побитного совпадения float32 outputs нет: max|delta score|7.74e-7,
max|delta E_pred|9.60e-7GeV; максимальный сдвиг energy summary2.50e-8GeV.
`torch.testing.assert_close` с defaults для float32 пройден. Отдельное
локальное требование exact bytes было строже scientific gate и сохранено
как failed diagnostic. Механизм этих малых различий не изолирован; ни новый
GPU retry, ни изменение recipe не потребовались. Предсказания и diagnostics
не перезаписаны. Проверяемый источник — `gpu_smoke_validation.json`.

Полные transfer jobs использовали тот же source c206ad68c9, bound inventory
a3baf9d937 и audit bundle index0ef7720326. При PENDING лимиты снижены
с разрешённой верхней границы2ч до1ч zeroSpot/defaultSpot и30мин bigSpot,
чтобы облегчить backfill. Основание —1.8–4.5с на первый файл после27.4с
initial setup в CUDA smoke,195/186/51 canonical nonempty files и запас
не менее трёх относительно линейной оценки. Снижение лимитов не меняет
расчёт или число jobs; автоматический повтор после TIMEOUT не разрешён.
Квитанции: `full_transfer_submissions.json`, `pending_walltime_reduction.json`.

## Что выполнено

Два source-map задания читали frozen corrected raw CSV из проверенных
node-local bundles. Все195/173 detector members входят в исходный inventory
9ca9df3edc0fafdc9d2e1720f37136443d41d96424966ec4fd43c35be786e110.
Поиск точный, по initial MCTrack PDG/Ekin/Px/Py/Pz/vX/vY/vZ, без fitted
filename shift. Рецепт не использует scores модели. Scored on: no model
scoring in this source audit; fitted on: nothing. Counts describe input
accounting; neither higher nor lower is better. N/A для quality direction.

| Potential | Frozen detector members (count) | Bound nonempty exports (count) | Header-only pairs (count) | Условие следующего этапа |
|---|---:|---:|---:|---|
| defaultSpot | 200 | 186 | 14 | Full audit4437504 пройден |
| bigSpot | 173 | 87 | 86 | 51 уникальная непустая пара; 36 aliases; full audit4437505 пройден |
| zeroSpot | 195 | 195 | 0 | Full audit4437503 пройден |

`defaultSpot` — повторная независимая проверка сохранённой карты4435670,
без нового source scan. Все225844 eligible records однозначно связаны,
повторного использования source tracks/events среди matched records нет.
14 vacs и hits одновременно header-only, hashes совпадают с сохранённым
bundle. Processed ROOT event count остаётся null.

`bigSpot`:4436975 COMPLETED/exit0 за9мин03с, cn-031. Все173 primary files
прочитаны. Все106242 eligible detector records имеют один точный source match
с теми же Row/Id. Это не106242 независимых исходных MC-трека: уникальных62133,
из них40437 представлены в двух или трёх detector files. Среди matched MC
records30651 исходных событий повторяются между файлами. Всего51 primary
source-file group;18 групп представлены одним detector file,30 — двумя,
3 — тремя. Это association по содержимому, не восстановленные production jobs.

Независимая проверка SHA256 выявила33 группы побайтно одинаковых непустых
hits/vacs pairs:87 файлов дают51 уникальную пару. Пример:
detector0008 и0031 связываются с primary0005, включая Row18/Id103;
оба hits и оба vacs имеют одинаковые SHA256. Повторы не объясняются
различной случайной реализацией detector transport в этих CSV.

До любых full SMASH model scores подготовлена детерминированная policy:
для совпадающих hits+vacs hashes и одинакового exact primary binding брать
лексикографически первый member, остальные сохранять как aliases в исходном
inventory/ledger. Это даёт51 непустой member и86 header-only, всего137 для
analysis view при173 учтённых исходных members. После такого
canonicalization повторов eligible source tracks/events между выбранными
members нет. Этот пересчёт identities не является model evaluation;
canonicalization реализована в bound inventory и проверена полным audit;
inference завершён после пройденного CUDA smoke. Все исходные файлы
и список36 aliases сохранены. Между source groups может сохраняться общая
production lineage; новое независимое разбиение из этого автоматически не
получается. Данные не объявляются fresh holdout.

## Новый отказ zeroSpot

4436974 FAILED/exit1 за8мин19с, cn-031: EACCES при чтении
`zeroSpot/10943245/0131_prim.csv` в исходном extracted primary tree.
Raw bundle390files/883486720bytes успешно распакован и проверен.
Последний сохранённый progress содержит126 прочитанных primary files и
152047 exact matches при236519 eligible target records. Это неполная карта;
нельзя выдавать её за полный binding195 members.

Попытка только побайтной упаковки primary sources на login-02 затем получила
EACCES на `0071_prim.csv` при повторном stat после копирования. Каталог
`primary_zeroSpot` содержит только incomplete `inputs.tar.partial`, без
manifest; архив не используется и не переименован в complete. Причина
нестабильной доступности не установлена. Успех соседнего чтения не закрывает
RA-09 и не доказывает исправление файловой системы.

Человек явно подтвердил один повтор: «Разрешить один повтор». После этого
создан новый полный primary bundle195files/30490818560bytes, manifest SHA256
726bcc5c87ed5199f26f3bb6aa87e5edd3a8252d077696e74c704732c635e27f.
Все126 ранее прочитанных primary hashes совпали. Helper проверяет fstat
того же открытого descriptor до/после чтения; файловые bytes и matching keys
не меняются. Старый incomplete tar сохранён отдельно и не используется.

Разрешённый повтор4437145 FAILED/exit1 за7мин49с на cn-030. Оба input bundles
проверены и читались из temporary allocation. Отказ теперь при atomic replace
`progress.json.tmp` → `progress.json` на scratch, а не при чтении primary CSV.
Complete summary отсутствует. Сохранённый progress содержит76 прочитанных
primary files/91322 matches; orphan tmp содержит81/97424. Оба имеют state
running, hashes проверены, ни один не используется как полный binding или
complete result. Временный файл не продвигался. Postmortem modes directory755,
files644, uid2761/gid100; это не доказывает точную причину EACCES.

После4437145 единственный подтверждённый повтор был использован; на тот момент
активных jobs этапа не было.
Подготовлен `zeroSpot_local_output_proposal.json`: ещё один CPU4/GPU0 job
до30мин с теми же immutable inputs и mapper, записью всех промежуточных
результатов на проверяемую local filesystem `/tmp` и однократным экспортом
complete compressed summary с SHA256/receipt. Scratch atomic replace в этом
варианте отсутствует. Proposal отдельно подтверждён и выполнен4437449.
Нет автоматических retries и утверждения, что Lustre
исправлена. Любой новый отказ сохраняется и закрывает продолжение.

## Следующий gate

Полный zeroSpot binding завершён. Accounting aliases/header-only реализован
и покрыт локальными tests. CPU full audits всех трёх potentials пройдены;
CUDA smoke и frozen-DCM transfer пройдены. Для этих фаз использовано сохранение результатов
на local filesystem и публикация одним hash-verified bundle вместо частых
atomic replacements на scratch. Source snapshot c206ad68c9 содержит151 проверенный
файл; семь model/input components совпадают с GPU probe22октября.
Предыдущие snapshots aaca41d0e5/b9ac727b05 не исполнялись.
Исходный inventory и first-file результаты сохранены.
Нельзя усреднять повторные files как независимые источники или превращать
неизвестный processed-event count в ноль. SMASH training требует отдельного
source grouping/split/protocol; fresh holdout отсутствует.

Проверяемые результаты:
`results/smash_full_transfer_20261023/bigSpot_binding_diagnostic.json`,
`bigSpot_deduplication_proposal.json`, `defaultSpot_binding_verified.json`,
`zeroSpot_export_receipt.json`, `zeroSpot_recovery_proposal.json`.
Диагностика DCM AUC вынесена в `roc_auc_gap_20261023.md`.
