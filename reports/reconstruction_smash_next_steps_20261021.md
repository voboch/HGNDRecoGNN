# План качества реконструкции и оценки SMASH

Обновление24октября2026: полный corrected-SMASH audit и frozen-DCM transfer
завершены. Source binding, header-only accounting, aliases и1296 prediction
hashes проверены; cluster metrics пересчитаны, figures осмотрены. Итог:
`smash_full_transfer_results_20261024.md`; receipts и история отказов:
`smash_full_transfer_status_20261023.md`. SMASH training не выполнялся;
следующий scientific gate требует production lineage, processed-event ledger
и source-grouped split. Разбор DCM AUC: `roc_auc_gap_20261023.md`.

На промежуточном этапе23октября bigSpot duplicates были обнаружены, а zeroSpot
получил EACCES. Эти отказы сохранены в истории; конечный transfer прошёл после
отдельно разрешённого mapping recovery и учёта дубликатов.

Исполнение22октября, следующий этап: fixed DCM GraphSAGE успешно применена
на CUDA к первым SMASH файлам всех трёх potentials и трём seeds. Полная
defaultSpot source map проверена для186 непустых files,14 header-only
остаются без binding evidence. Метрики, QA и пределы:
`smash_staged_transfer_results_20261022.md`.

Обновление22октября: source lookup4435586 восстановил первую defaultSpot
file association (detector0001→primary0002,334 exact matches). Полная карта
4435592 остановилась на compute EACCES0145_vacs.csv; transfer и обучение не
отправлены. Актуальные receipts и условия продолжения:
`smash_transfer_status_20261022.md`.

Исполнение 21 октября: DCM diagnostics завершены (28 figures), см.
`student_pipeline_diagnostics_20261021.md`. SMASH CPU smoke4430110
остановлен из-за EACCES на cn-034; полный аудит и обучение не отправлены.
Ограниченный zeroSpot probe, source inventory и условия продолжения:
`smash_audit_status_20261021.md`. Ниже сохранён согласованный порядок этапов.

21 октября 2026. Предложение по запросу пользователя; новые training,
preprocessing или evaluation jobs этим документом не запущены. Завершённая
native DCM серия сохраняется как отдельный результат. Старые закрытые
кампании, notebook recipes и manuscript не меняются.

## От чего начинаем

Все12 native обучений4424084 и test4424085 завершены, exit0. На одном reused
test source988976 получен mean cluster ROC-AUC0.906–0.910; sample SD по трём
training seeds составляет0.0043–0.0077. Population:222251 reconstructed
clusters,36713 MC-positive,91081 physical events. Веса и scalers обучены на
988975, checkpoint/cut выбраны на988974. Energy MAE без score cut на
MC-positive clusters составляет244–261MeV по средним трёх seeds. Это ошибка
native max-hit target, не разрешение энергии отдельных нейтронов.

Canonical numbers и пределы: `student_pipeline_results_20261021.md`,
`results/student_pipeline_20261020/evaluation_summary_20261021.json`.
Исходный student ROC-AUC≈0.98 относится к другой популяции, меткам, времени,
кластеризации и training objective. Причина разрыва одним этим сравнением
не выделяется. По уже открытому DCM test не выбираем новый recipe или
победителя архитектур. Прямого измерения эффекта student label на SMASH нет.

## Цель и порядок решений

Цель — оценить, насколько реконструкция сохраняет SMASH response по
U_sym=0/18/90MeV, с учётом фоновых кластеров, энергетической миграции и
потерь нейтронов. ROC-AUC остаётся диагностикой. Решение о пригодности
опирается на common-truth neutron matching, energy response и closure
наблюдаемой величины. Заранее фиксируются ожидаемое направление/масштаб
каждого изменения, популяция, бюджет и правило остановки. Ниже unknown
оставлены unknown; ожидаемый выигрыш в sigma не обещается.

## 0. Закрыть технический отчёт DCM

Выгрузить и визуально проверить12 learning histories, все validation scans,
per-seed purity/efficiency и энергетические остатки. Проверенные компактные
receipts уже получены; inference повторять не требуется. Пересчёт из CSV
проверяет арифметику, не создаёт новый test. Сравнивать классификацию отдельно
по размеру кластера, side, энергии и multiplicity; отдельно проверить влияние
одиночных кандидатов. Не складывать seeds в один ROC.

Результат: воспроизводимый пакет «checkpoint → population → predictions →
metrics» и список конкретных failure modes. Первым вопросом становится
энергозависимая efficiency и bias, а не поиск следующей архитектуры.
Нового обучения:0. Финальные figures требуют осмотра перед использованием
в note/paper; их QA не объявлен завершённым проверкой JSON.

## 1. Зафиксировать пригодные данные и разбиение SMASH

Проверить фактическую detector generation и соответствие raw hits, primary
MC inventory, геометрии и job manifests. Основной результат относится к
исправленной production; superseded _v2 нельзя сделать текущей переименованием
каталога. У старых кешей с неустановленным source list статус diagnostic.

Единый ключ: source generation/job/file + local event + detector side; обе
стороны события и все файлы одного generation job остаются в одной части.
Manifest перечисляет train, validation/calibration и окончательный test для
каждого U_sym. События без кандидатов остаются в event inventory. Проверяются
дубликаты, полное покрытие, отсутствие общих событий/jobs и связь с MC IDs.

Отдельно выяснить, какие jobs уже участвовали в training, calibration,
window scans и просмотре результатов. Если свежих jobs нет, новый split
старых просмотренных данных называется retrospective holdout. Для
независимого подтверждения нужна новая неиспользованная production, а не
повторное перемешивание прежних событий.

Scaler fitted только на train и один для всех сравниваемых U_sym. В transfer
проверке DCM→SMASH сохраняются DCM scaler и весь DCM input recipe; смена
окна/шума/scaler при переносе была бы дополнительным изменением. Для нового
SMASH training общий scaler обучается на его train. Колонка U_sym в модель
не передаётся. Acceptance и центральность согласуются на уровне событий;
truth-b контроль отдельно помечается как MC diagnostic.

Условие перехода: provenance и grouping восстановлены. До этого не нужна
большая серия SMASH и не нужен новый чувствительный спектральный показатель.

## 2. Разделить метку, образование кластеров и энергию без обучения

На одних и тех же SMASH хитах/графах сравнить текущую метку с вариантом без
Ekin>0.3GeV. PDG/Side/ToF условия, шум, признаки и edges остаются прежними.
Посчитать по каждому U_sym, сколько hits и clusters меняют label, в каких
энергиях и multiplicities; сохранить новые/старые targets и hashes.
Ожидание: меняются только кластеры без прежнего qualifying neutron выше
0.3GeV. Максимальный energy target прежних positive clusters сохраняется.
Доля изменений на SMASH пока неизвестна; DCM fraction не подставляется.

Построить одинаковую MC reference inventory прибывших нейтронов с energy/ID,
отдельно сохранив generated inventory для полной detection efficiency.
Обе модели оцениваются против одного внешнего определения, включая
раздельные soft/hard populations. Смена собственного training label не
должна сама улучшать название «purity». Использовать one-to-one matching;
раздельно считать merging, splitting, lost neutrons и fake candidates.
Cluster AUC не заменяет эти показатели.

Лестница на одинаковых событиях: generated/accepted neutrons → detector
neutrons → clusters с MC energy → выбранные clusters с MC energy → те же
выбранные clusters с predicted energy. При фиксированных weights, score cut,
calibration и полосах одно переименование MC labels не меняет reconstructed
R_n. Отличие последнего шага от предыдущего локализует energy migration;
эффект кандидатной сегментации оценивается на предыдущих шагах.

Результат: ledger потерь по U_sym и multiplicity. Ожидаем обнаружить
потенциальную проблему раньше дорогого обучения; направление нового SMASH
response без этого ledger неизвестно. Нового обучения:0.

## 3. Провести transfer diagnostic на development SMASH

Применить зафиксированные DCM модели к согласованным native SMASH inputs,
сохранив scaler, smearing, graphs и frozen checkpoint. Сначала использовать
validation/calibration jobs; окончательный SMASH test не открывать. В качестве
простого заранее заданного candidate reference предлагается GraphSAGE;
выбор объясняется простотой модели, а не максимальным AUC на DCM test.
DynamicEdgeConv — отдельный следующий family control, если его необходимость
подтвердят validation diagnostics. Spectral gate пока не приоритет.

Измерить ROC/AP на общей population, efficiency при common validation
purity convention и bias/resolution по энергии, side и multiplicity. Порог
один на модель, замороженный на общей calibration population; подгонять
порог отдельно под каждый U_sym нельзя. Метрики каждого U_sym нужны для
проверки неодинакового response, а не для трёх разных calibrations.

Результат: различение generator transfer и ошибки reconstruction recipe.
Ожидание — возможное ухудшение при переносе; величина unknown. Никакого
предположения о сохранении DCM0.91 или student0.98 на SMASH не требуется.

## 4. Ограничить первое переобучение девятью runs

После data/observable audits и GPU smoke зафиксировать минимальный протокол:

1. GraphSAGE с native label: seeds42/123/456.
2. Штатный net_default с native label: те же три seeds как pipeline reference.
3. Та же GraphSAGE со student label (только снятый Ekin floor): те же три seeds.

Это9 runs, а не новый sweep всех архитектур. Последние3 нужны только если
этап2 подтвердил реальное изменение training population. Исторический SMASH
checkpoint без подтверждённой generation/split lineage не заменяет новый
reference run. Для architecture comparison используются одни данные,
популяция и общий budget; различие head/loss явно описывается, поэтому это
сравнение pipeline families, не pure-backbone ablation.

Для label contrast остаются идентичными x/edges/noise, architecture,
optimizer, budget и loss coefficients; меняется только target construction.
Данные всех U_sym входят в train по заранее заданному policy. Все настройки
выбираются до длинной серии на train/validation; test не определяет budget,
epoch или class weights. Checkpoint и operating point — только validation.
Если требуется другой sampler/ranking/energy weight, это следующий опыт,
а не незаметная поправка внутри label contrast.

Локальные и CUDA smoke проверки обязательны. Максимум2 tasks одновременно;
automatic retries/sweeps нет. В завершённой DCM серии такой student run
занимал19–22мин на V100 при30epochs и152225 train/validation arm graphs.
Это измеренный ориентир исполнения, не SMASH time estimate. Время и память
нового объёма оцениваются отдельным smoke/pilot до allocation. Ожидание —
неизвестное изменение high-energy efficiency/bias, а не обещание AUC gain.

## 5. Улучшать энергетическую голову по выявленной ошибке

Сохранить observable eToF как простой energy baseline и применить ту же
candidate selection. Для каждого U_sym показать E_pred−E_true, quantiles,
energy-bin migration, включая зависимость от multiplicity и числа truth IDs
в компоненте. Native max-hit energy target и энергия одного neutron track
различаются; соответствие доказывается matching/response, не именем столбца.

Если classifier приемлем, а response теряет знак или масштаб, следующий опыт
меняет energy target/loss/calibration по одному фактору. Occupancy weights
строятся только на train; общая calibration fitted на calibration jobs и
применяется одинаково ко всем U_sym. Не вводить отдельные scale corrections
по потенциалам. Для допуска заранее задать tolerable differential bias через
целевую uncertainty U_sym и sensitivity response; произвольное число MeV или
процент под измеренный эффект не выбирать.

Если основная ошибка — merging/splitting, сначала исправляется образование
кандидатов и multiplicity response. Радиус, time window, min_hits и ToF
definition исследуются отдельными controls с полной scan surface.
fTime_nonoise остаётся ideal-timing diagnostic, не основной детекторной оценкой.
Новый эксперимент выбирается только после указания измеренного failure mode.

## 6. Определить физическую наблюдаемую до final test

Сейчас _counts_reco/_counts_truth в `scripts/hardness_threshold_opt.py`
считают низкую полосу как E<E_low. Условие plateau_min ограничивает E_low
в сканировании, но не задаёт нижнюю границу интеграла. Старую фразу
«Ekin floor ниже scan range и потому не влияет» нельзя использовать как
доказательство. Это замечание к интерпретации legacy analysis, не новое
измерение student-label effect.

Предлагаемый estimator:
R_n=N(E_H≤E_rec<E_max)/N(E_min≤E_rec<E_L), где обе полосы целиком лежат в
проверенном response диапазоне. Ни одна новая численная граница здесь не
выбрана: границы фиксируются на development/calibration по efficiency,
resolution, migration и полной scan surface. Альтернатива — forward-folded
spectral fit с общими response/background nuisance parameters. Открытый
high-energy tail тоже требует контроля, а не только низкий знаменатель.

Response связывает truth neutron energy/multiplicity с reconstructed
candidates и содержит losses, merging, splitting и fakes. Простое деление
cluster counts на efficiency не называется unfolded neutron spectrum.
Одинаковые labels/units/ranges сохраняются для U_sym=0/18/90MeV. Primary
HGND quantity — neutron response. R_n/R_p требует отдельно измеренной
протонной части; truth-proton hybrid остаётся верхней feasibility оценкой.

## 7. Проверить closure, мощность и только затем открыть final test

На development/calibration пройти same-sample null, known injected response,
перенос calibration между source jobs, стабильность к sample composition и
заранее выбранным perturbations energy scale/efficiency. Анализировать slope
по всем трём U_sym и 90/0 response; положительные point estimates не заменяют
интервалы. Немонотонный middle point при недостаточной мощности сам по себе
не опровергает response. До открытия test оценить по job variance число
независимых jobs для целевой precision; новые статистические ресурсы не
подменять повторными bootstrap resamples.

Разброс обучения и MC статистика — разные компоненты uncertainty. Для двух
наборов weights — paired differences по shared training seeds, SE и t с
числом df; при3seeds df=2 и inherited two-sided critical4.303. Для event/job
population — resampling generation jobs, парный между reconstruction variants
на тех же jobs, с повторным fit reweighting внутри replicate. Независимые
productions разных U_sym не получают вымышленного event pairing.

После фиксации weights, thresholds, calibration, bands и uncertainty model
однократно оценить неиспользованный test. Reused test остаётся retrospective.
Отчёт содержит по каждой модели и seed object/energy metrics, response,
полные histories/scans и systematic variations. Улучшение объявляется по
физической точности и closure, а не по одному AUC или максимуму sigma.
Чувствительность к U_sym не отождествляется с точностью EOS parameter L.

## Первый исполнимый пакет

Сначала завершить QA существующих DCM артефактов, проверить SMASH
provenance/split, выполнить общий matching и label-change ledger без нового
обучения. Затем зафиксировать первый SMASH protocol с указанными9 runs и
обоснованным budget. Если data lineage или свежий holdout не восстановлены,
результат пакета — явно ограниченный diagnostic, не заявка независимого
физического измерения. Расширять архитектуры до этих проверок не предлагается.
