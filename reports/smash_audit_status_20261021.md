# SMASH: аудит данных остановлен до переобучения

21 октября 2026. CPU smoke 4430110 завершился FAILED, exit 1:0, через 5:22 на
cn-034. Причина в журнале — EACCES при чтении corrected defaultSpot и затем
при обращении к output directory для записи failure receipt. Полный аудит,
transfer inference и обучение не отправлены. Автоматического повтора нет;
активных jobs этой серии не осталось.

## Доступ к данным

Первый отказ возник при stat файла
`/scratch/vbocharnikov/hgnd/data_fixed/smash_xecs_2.87gev_hardSkyrme_defaultSpot/10943245/1054_vacs.csv`.
На login-02 тот же файл доступен с mode644, владелец vbocharnikov, ACL
user::rw-/group::r--/other::r--. Output directory также доступна с login-02.
Это различие между узлами; оно не доказывает точную инфраструктурную причину.
Права исходных данных не менялись. Журнал сохранён, Slurm FAILED имеет
приоритет над отсутствующим JSON. Старый DCM RA-09 этим не закрыт.

После сбоя локальный failure reporter исправлен: вторичная ошибка записи
больше не скрывает исходное исключение. Добавлен test на отказ output FS.
Эта версия не отправлялась на кластер; immutable source первого smoke и
все его артефакты сохранены.

## Инвентаризация corrected production

<!-- table-provenance: no direction is better; exact directory-name inventory of corrected SMASH detector exports data_fixed and extracted n/p primary inventory on login-02; not model scores or physics estimates; no train/test partition established; Xe+Cs Ekin/A=2.87GeV hardSkyrme directory labels, full generator-job lineage unverified; no b selection (b unavailable). -->

Это числа файлов с совпавшими run/stem в именах. Uncertainty не применяется
к inventory counts; содержимое всех файлов и отсутствие дубликатов ещё не
проверены. Совпадение имени не заменяет связь MC IDs.

| U_sym (MeV) | Corrected hits/vacs pairs (files) | Available matching primary names (files) |
|---:|---:|---:|
| 0 | 3123 | 195 |
| 18 | 2383 | 200 |
| 90 | 1865 | 173 |

Во всех трёх каталогах нет непарных hits/vacs names. Для полной лестницы
generated→detector→clusters имеющийся primary inventory покрывает только
часть corrected export files. Потребуются недостающий inventory или заранее
описанная общая ограниченная популяция. Независимость generation jobs и
наличие свежего blind holdout пока не установлены. Исторические v3 caches
без точного frozen source manifest остаются diagnostic; их глобальные Row
нельзя связывать с текущим raw деревом лишь по современному порядку файлов.

## Завершённый ограниченный probe

Успела обработаться одна пара corrected zeroSpot10943245/0001. Это первый
файл по имени, не случайная выборка и не оценка полного SMASH response.
Seed 42 задаёт новую воспроизводимую реализацию sigma_t=0.15 ns, общую для
двух labels. Graph radius 3.6 cells/time 1.5 ns и Edep>0.003 GeV наследуют
штатный SMASH recipe. После hit cut остаются 20277 hits и 6107 components на
3084 detector-export events. Generated inventory содержит 4179 events;
события без detector exports в denominator detector-only ledger не входят.

Удаление Ekin>0.3 GeV при неизменных PDG/Side/ToF условиях добавило 22 signal
hits и 18 positive clusters: 1111→1129; 12 из 18 — singleton candidates.
Это 18/6107=0.295% всех кандидатов и 18/1111=1.62% от прежнего числа positive
clusters в этом файле. Ни одна прежняя positive label или её maximum-hit
energy target не изменилась. Эти доли не переносятся на U_sym=18/90 MeV.

В native prompt ledger 1100 eligible neutron IDs сопоставляются максимум
1087 кандидатам one-to-one; 15 components содержат несколько eligible IDs,
26 IDs встречаются в нескольких components. После снятия floor получены
1118 eligible IDs, 1105 matches, те же 15 merges/26 splits. Это post-hit-cut
prompt-object matching, не full detection efficiency или neutron-track
energy resolution. Detector export до hit cut содержит 1684 unique neutron
IDs на Side 5/6, включая вторичные нейтроны; native label не требует
fMotherId=-1. Связь с generated primary neutron target требует отдельного
учёта ancestry и не определяется названием столбца cl_label.

Primary CSV содержит только neutron/proton records:604291/469152.
Все 923 detector primary neutron records на Side 5/6 найдены по локальным
Row/Id; PDG и production vertex согласованы. Допуск 2e-6 cm — две единицы
последнего знака шестизначного CSV export, а не физический selection cut.
Начальный широкий probe видел 171 ненайденную частицу:170 pions и 1 kaon;
отсутствие этих частиц в n/p inventory ожидаемо. После проверки species
ни одного отсутствующего primary neutron ID в этом файле не найдено.
Это проверка одного file namespace, не доказательство полной lineage.

## Что требуется перед следующим этапом

Нужен успешный read-only CPU smoke с вычислительного узла с доступом к raw
и scratch output. Затем — полный label ledger, content hashes/duplicates,
согласованный common-truth neutron inventory и manifest разбиения по
generation jobs. До этих gates плановая серия до 9 обучений не запускается.
Численного вывода о росте разделения потенциалов сейчас нет.

Добавлен finite-band API для [E_min,E_L) и [E_H,E_max), с zero-event
inventory и проверкой histogram edges. Legacy hardness counter defaults
сохранены. Условие plateau_min по-прежнему ограничивает верхний край low
band и не исключает мягкую часть знаменателя. Новые численные physics
границы и scan optimum не выбраны.

Canonical receipts: `results/smash_audit_20261021/launch_status.json`,
`inventory_20261021.json`, `zeroSpot_smoke_summary.json`,
`primary_probe_detail.json`, `primary_binding_verified.json`, `smoke_4430110.out`.
Primary binding воспроизводится `scripts/check_smash_primary_binding.py` на
двух файлах с сохранёнными SHA256. Raw CSV и архивы остаются вне Git.
Immutable source SHA256:
`f7c5049eed3ddc0cf40a3abaf8e689591a97a7c3a377fbd2ee54d955c0ec935b`.
Scope и бюджет: `smash_audit_protocol_20261021.md`.
