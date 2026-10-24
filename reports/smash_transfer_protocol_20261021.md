# DCM→SMASH: общая development population и фиксированные модели

21 октября 2026. Пользователь после отчёта об EACCES поручил «переходи к
следующему этапу». Это разрешает одну новую ограниченную проверку доступа,
завершение аудита и подготовку transfer diagnostic. Предыдущий failed
smoke4430110 и immutable source сохраняются. Автоматических retries нет.
Закрытые DCM задания не возобновляются; нового обучения этим этапом нет.

## Популяция до результатов

Frozen name inventory SHA256:
`9ca9df3edc0fafdc9d2e1720f37136443d41d96424966ec4fd43c35be786e110`.
Используется пересечение corrected `data_fixed` и n/p primary exports:
195/200/173 файлов для U_sym=0/18/90MeV. Выбор сделан по наличию primary,
до новых scores. Все эти файлы считаются retrospective development.
Зависимость generation jobs и свежий final holdout не установлены.
Отсутствие такого holdout не выдаётся за независимую верификацию. Новый
train/validation/test split не создаётся перемешиванием transport files.

Результат относится к этой общей популяции; label fractions полного
corrected дерева 3123/2383/1865 файлов здесь не оцениваются. Source identity
каждого события: U_sym/source directory/file/local Row; detector sides не
разделяются. File names/relative paths заморожены; content hashes каждой
тройки hits/vacs/primary фиксируются до inference. Повторные непустые пары,
неоднозначные Row/Instance, отсутствующие nucleon IDs или несовпадение
PDG/production vertex останавливают downstream processing. Совпадение MC
IDs в этих файлах не доказывает независимость generation jobs.

## Проверка доступа и бюджет

Один новый CPU smoke на partition=test, максимум15мин, четыре CPU, GPU0,
--mem=0. Проверяются чтение первой общей тройки каждого U_sym и прежнего
проблемного defaultSpot10943245/1054_vacs.csv, затем scratch write/fsync/
atomic replace/readback. ACL исходных данных не меняются. cn-034 исключён
после воспроизведённого отказа как эксплуатационная мера; это не диагноз
инфраструктуры. cn-007/cn-008 исключены после прежних checkpoint EACCES.

После COMPLETED/exit0 и проверенных трёх ограниченных summaries: три CPU
задачи аудита, максимум2 одновременно, не более2часов на каждую. Это верхний
allocation bound. Время одной пары измеряется smoke до отправки полной
серии. При неуспехе нет автоматической цепочки повторов. Не относящиеся к
этой серии jobs не меняются. Перед длинной GPU inference обязателен её
собственный partition=test smoke; бюджет GPU будет зафиксирован по размеру
проверенных входов до отправки.

## Аудит labels и primary linkage

Аудит наследует SMASH preprocessing: Edep>0.003GeV, sigma_t=0.15ns,
radius3.6cells, time1.5ns, max32neighbors, singleton retained и native
PDG0 half-event rejection. Значения взяты из действующего recipe и здесь
не оптимизируются. Seed42 с hash относительного file path задаёт новую
реализацию detector jitter, общую для native/relaxed label. Это не
восстановление неизвестного исторического RNG state.

Меняется только Ekin>0.3GeV в prompt neutron rule; PDG/Side/ToF остаются
прежними. Ожидание: добавятся мягкие positive components; label и maximum
energy target прежних positives сохранятся. Доли по U_sym неизвестны.
Maximum-cardinality candidate/object matching, merging/splitting и
export-event denominator считаются раздельно с generated inventory.
Все имеющиеся primary файлы проходят binding, а не только первые три.
Primary exports содержат n/p, поэтому mesons не объявляются пропавшими
primary neutrons. vertex tolerance2e-6cm — две единицы шестизначной CSV
сериализации, не physics acceptance cut.

Unscaled observed hits сохраняются tensor-only с hashes для следующей
фазы. Геометрические min/max LayerId/RowId/ColumnId записываются по файлам
для проверки SurfaceHit recipe. До успешной проверки content identity и
geometry downstream модели не применяются.

## Transfer после data gates

Первый family control — уже обученная native GraphSAGE, seeds42/123/456.
Она назначена в плане по простоте до этого SMASH опыта, а не выбрана по
DCM test. Source/checkpoints campaign
`student_pipeline_20261020_56bcc48206` остаются frozen. Checkpoint — прежний
DCM validation minimum. Scalers — DCM train988975, без refit на SMASH.
Первичная transfer оценка использует прежний DCM validation score cut для
каждого seed: никакого подбора отдельно по потенциалам. Общая SMASH
calibration при необходимости будет отдельным заранее описанным опытом.

Transfer сохраняет DCM radius3.6cells и time1.0ns. Audit time1.5ns — отдельный
SMASH label/candidate diagnostic; его components не подаются молча в DCM
модель. Native10features, Gaussian detector timing, singleton candidates,
SurfaceHit definition и cluster order должны быть проверены. MC columns не
входят в forward. Новые weights, recipe, optimiser или energy calibration
в transfer не обучаются. Ожидается возможное ухудшение при generator
transfer; величина ROC/AP/energy bias и U_sym response неизвестна.

Сохраняются per-file/per-U_sym/per-seed scores, ROC/AP, fixed-cut count
purity/efficiency, energy bias/MAE и candidate/object losses. Числа являются
cluster diagnostics против MC, не neutron-track resolution или unfolded
spectrum. По generation-job-independent test выводов не делается. Нельзя
выбирать новый recipe или обучать девять runs до восстановления lineage и
разбиения. Physics energy bands и sensitivity optimisation здесь не
назначаются: finite-band API уже есть, численные границы требуют response.

### Уточнение размещения CPU-проверки до исполнения

Тестовая очередь имеет partition QoS=test с MaxJobsPU=1. Пока её занимало
несвязанное GPU-задание, CPU-only smoke4430367 ожидал этот лимит. Его
ожидающая allocation переведена в rocky с тем же job ID, GPU0/CPU4 и15мин,
без дублирования или изменения data/model recipe. Это read-only data/access
probe, не training/inference smoke. Обязательная CUDA transfer-проверка
остаётся на partition=test перед длинной GPU inference.

### Ограниченный поиск источника после failed binding

Доступ на cn-038 прошёл, но defaultSpot10943245/0001 не прошёл MC binding.
Локальный exact fingerprint check не нашёл ни одной из334 detector primary
nucleons в одноимённом primary file. Первые zeroSpot/bigSpot files прошли.
До обращения к новым данным проверяется вариант перенумерации файлов:
одна CPU-only задача, максимум45мин, CPU4/GPU0, rocky, без requeue/retry.
Она ищет ровно эти334 signatures (PDG,Ekin,Px,Py,Pz) во всех200 известных
primary files defaultSpot. Targets назначены по первому failed file, а не
по model scores. Совпадения сохраняются все, vertex equality проверяется
отдельно; источник или event mapping не выбирается по близости. Если поиск
неполон или неоднозначен, gate остаётся закрытым. Это отдельный source
lookup, не повтор полного аудита или inference. Общий CPU budget остаётся
ниже ранее описанных трёх allocations по2часа; новые модели не запускаются.

### Расширение source lookup на frozen candidates, 22 октября

В промежуточном результате4435586 все334 records corrected
defaultSpot10943245/0001 найдены в primary10943245/0002: точные
PDG/Ekin/Px/Py/Pz/vertex, неизменные Row/Id. Уникальность устанавливается
только после чтения всех200 primary files. Этот результат допускает
ошибку file association для первой пары; постоянный сдвиг имён остальных
файлов из него не следует.

После успешного завершения4435586 — одна новая bounded CPU задача поиска
для всех200 frozen candidate detector files defaultSpot во всех200 primary
files того же frozen inventory. CPU4/GPU0, rocky, максимум45мин, без requeue
и retry. Подбор по downstream scores отсутствует. Ключ поиска — точные
initial MCTrack PDG/Ekin/Px/Py/Pz/vX/vY/vZ; несовпадающие Row/Id, missing,
ambiguous или разделённые между primary files records сохраняются как
unresolved. Ожидание: может восстановиться часть file associations;
полное покрытие и постоянный filename offset неизвестны. Результат сам по
себе не создаёт train/test split и не подтверждает generation-job independence.

До нового аудита требуется проверенный явный detector→primary manifest с
input hashes и отсутствием неоднозначного/повторного event mapping.
Первоначальный immutable inventory не переписывается, downstream gates не
обходятся. Смена file association исправляет provenance, не graph/label/model
recipe. Новые audits/GPU smoke назначаются отдельно после проверки mapping.

При восстановлении generated-neutron denominator требуется отдельная
квитанция detector ROOT event range, включая события без отклика. Для первой
восстановленной пары max detector-vacs Row1021, primary tail Row4067;
по export maxima нельзя объявлять, что все primary events были обработаны
detector conversion. До подтверждения range полная generated detection
efficiency не рассчитывается.
