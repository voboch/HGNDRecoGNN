# SMASH: проверка источника перед переносом DCM моделей

22 октября 2026. Продолжение по поручению пользователя «продолжай работу с
экспериментами». Этот статус обновляет состояние исполнения из
`smash_transfer_protocol_20261021.md` и
`reconstruction_smash_next_steps_20261021.md`. Предыдущие отчёты и immutable
sources сохранены.

## Текущее состояние

CPU smoke4430367 завершился FAILED/exit1:0 на cn-038 за2мин26с. Проверка
чтения входов и scratch write/fsync/atomic replace/readback прошла. Аудит
остановило несовпадение MC identity для corrected
defaultSpot/10943245/0001 и одноимённого primary export. Полный аудит,
transfer inference и новое SMASH обучение не отправлены. Успешная проверка
доступа на cn-038 не закрывает прежнюю EACCES проблему на других узлах.

Состояние `smoke_pending` в локальном launch_status исправлено по Slurm
accounting и сохранённой terminal receipt. Slurm является источником статуса
исполнения; наличие частичного summary не означает успешного завершения.

Поиск4435586 завершился COMPLETED/exit0 на cn-039 за6мин49с. Проверены все200
известных primary files defaultSpot. Все334 primary n/p records проблемного
detector file однозначно найдены в primary10943245/0002: совпали точные
PDG/Ekin/Px/Py/Pz, vertex и Row/Id. Дополнительных matches в остальных файлах
нет. Это восстанавливает первую file association; постоянный сдвиг остальных
имён не проверен.

Задание4435592 завершилось FAILED/exit1:0 на cn-039 за9с при чтении
defaultSpot10943245/0145_vacs.csv: EACCES. Оно остановилось при сборе detector
targets до primary scan; полного mapping summary нет. Ограничения были:
одна CPU-only задача, rocky, CPU4/GPU0,45мин, без requeue и автоматического
retry. Models, thresholds и physics windows в этой проверке не подбирались.
Новых задач после отказа нет.

На login-02 файлы0144/0145/0146_vacs.csv читаются, mode644, UID2761/GID100.
Эта проверка не объясняет compute EACCES. На том же cn-039 первый поиск
прочитал200 primary files успешно; считать узел всегда неисправным оснований
нет. Нужен доступ к corrected detector inputs из compute allocation.

## Что проверено на первых файлах

Все следующие числа — MC source diagnostics на первом файле каждого
потенциала; это retrospective development, без model fitting. Counts точные
для прочитанных exports, статистическая неопределённость к ним не приписана.
Selection: fMotherId=-1, Side=5/6, abs(PDG)=2112/2212, уникальные Row/Id.

<!-- table-provenance: reference-only MC source audit, retrospective development; no model or scaler fitted. No direction is better for inventory counts. -->

| U_sym (MeV) | eligible n/p records (count) | matching Row/Id (count) | matching PDG (count) | matching PDG and vertex (count) |
|---|---:|---:|---:|---:|
| 0 | 1264 | 1264 | 1264 | 1264 |
| 18 | 334 | 308 | 222 | 0 |
| 90 | 1216 | 1216 | 1216 | 1216 |

Таблица относится к первоначальным одноимённым file associations.
Допуск vertex2e-6cm наследует две единицы шестизначной CSV сериализации;
он не является physics acceptance cut. Последующий source lookup использует
точное равенство всех записанных полей без tolerance.

Соответствие названий zeroSpot/defaultSpot/bigSpot потенциалам0/18/90MeV
наследуется из действующего SMASH analysis. Для18MeV отсутствуют21 neutron
ID и5 proton IDs; совпадение числовых Row/Id у308 записей не подтверждает
общую production. Для0 и90MeV первые файлы прошли; остальные члены ещё не
проверены. Эти строки не измеряют detection efficiency.

Exact поиск по (PDG,Ekin,Px,Py,Pz) во всём одноимённом primary file нашёл
0 из334 detector n/p records defaultSpot. Для первых zeroSpot и bigSpot
получены1264/1264 и1216/1216 соответственно. Замена corrected defaultSpot
на старый pre-fix detector file также дала0 согласованных PDG+vertex;
подмена data generation проблему не устраняет.

В local converter sources `ncx/convert2csv.cc` и
`ncx_refactor/Convert2Csv.C` vacs и prim берут Ekin, momentum и production
vertex из CbmMCTrack. Vacs Id — GetTrackID surface point, prim Id — индекс
MCTrack, Row — ROOT event index. При общем исходном ROOT эти поля должны
согласовываться. Vacs Ekin здесь является initial MCTrack kinetic energy;
он не устанавливает энергию прибытия после transport.

195/200/173 совпадающих относительных имён файлов — только candidate name
intersections. Прежнюю формулировку «общая population» из протокола следует
читать как запланированную популяцию после binding gate. Для всех568 файлов
общая physics identity не подтверждена. Никакого выравнивания по близости
vertex, переноса Row или выбора файла по downstream scores не применялось.

## Условия следующего перехода

Locator сохранил все точные matches с detector и primary Row/Id, file и
vertex, а также SHA256 каждого прочитанного primary file. Локально проверены
summary hash, полное покрытие200 filenames, уникальность334 matches,
неизменные Row/Id и exact vertices. Полная карта требует тех же проверок на
остальных detector files и проверки повторных events между ними. Частичное
совпадение или незавершённый поиск не открывают gate.

Для восстановленной первой пары обнаружен отдельный event-coverage вопрос:
detector vacs имеет781 export events с максимальным Row1021, тогда как
последняя строка primary0002 имеет Row4067. Последнее число — tail metadata,
не квитанция числа обработанных событий. Максимум Row в hits/vacs не
восстанавливает события без detector response. Полный primary inventory
нельзя использовать как generated efficiency denominator без подтверждённого
диапазона ROOT events, обработанного detector conversion. Этот вопрос
сохраняется даже после исправления filename association.

Для первого detector file соответствующий primary dump найден. Запрос
corrected ROOT/DST provenance остаётся полезен для generation-job grouping;
для первой file association новый dump уже не требуется. После восстановления
compute read access нужно завершить полную карту и проверить source hashes,
уникальность event associations и отсутствие повторно экспортированных events.
Частичные записи failed mapping job не заменяют эту проверку. Retry не
отправлен; возможная следующая реализация — hash-verified input archive с
чтением из локального scratch compute-узла, без изменения analysis recipe.

Исторические converter presets не доказывают происхождение нынешнего
data_fixed. Legacy wrapper формирует номера через вложенный substring glob,
refactored wrapper — через отсортированный список ROOT paths. Различие этих
алгоритмов проверено в коде, но какая команда создала текущие exports,
не установлено. Оно не объявляется причиной конкретной перенумерации.
Generation-job independence и свежий final holdout не установлены;
перемешивание известных transport files их не создаёт.

Локально подготовлены native DCM graph/feature view и frozen GraphSAGE
transfer driver с запретом full inference до полного аудита всех потенциалов.
Локальные focused checks:20 tests passed, Ruff passed. CUDA transfer smoke
не выполнялся. Его собственный partition=test smoke обязателен перед длинной
GPU inference; нового SMASH training protocol пока нет.

DCM результат остаётся прежним: mean cluster ROC-AUC0.906–0.910 на reused
test988976, sample SD по трём training seeds0.0043–0.0077. Student≈0.98
относится к другой population и recipe. Текущая проверка источника SMASH не
измеряет улучшение reconstruction или чувствительность к U_sym. Manuscript
и закрытая DCM development campaign не изменены.

## Артефакты

- `results/smash_transfer_20261021/launch_status.json` — текущее исполнение.
- `terminal_receipt_4430367.json`, `access_receipt.json`, `failure.json` в том
  же каталоге — terminal status, storage checks и scientific failure.
- `zero_binding_verified.json`, `default_binding_verified.json`,
  `big_binding_verified.json` — single-file counts и input hashes.
- `locator_source.json`, `locator_submission.json` — immutable source SHA,
  точная команда и бюджет поиска4435586.
- `locator_summary.json`, `locator_terminal_receipt.json`,
  `locator_validation.json` — полный результат первого поиска и его проверка.
- `mapping_source.json`, `mapping_submission.json` — источник, бюджет и
  квитанция поиска карты4435592.
- `mapping_terminal_receipt.json`, `mapping_login_access_after_failure.json`
  — новый compute EACCES и read-only login probe.
- `converter_provenance.json` — hashes локальных converter/wrapper sources,
  семантика полей и пределы provenance вывода.
- `first_default_event_coverage.json` — ограничения generated-event denominator.
- `input_inventory.json` — frozen candidate names,
  SHA2569ca9df3edc0fafdc9d2e1720f37136443d41d96424966ec4fd43c35be786e110.

Audit immutable source SHA256:
9b3eed1f0bab25ed51b691fbae225baf7f9911ae9a5d32f44cab6e8a03657960.
Locator source SHA256:
285dfe9838d3cfd3ef48ffb4cebf4f739af5f3844b324e81db4835c948775038.
