# Полный corrected-SMASH audit и frozen-DCM transfer

23 октября 2026. Пользователь поручил перейти к corrected-SMASH audit/transfer
после binding остальных zeroSpot/bigSpot файлов и согласования empty-file
accounting. Это новый конечный этап; закрытые DCM jobs не возобновляются.

До измерения фиксируем популяцию: исходный inventory SHA256
9ca9df3edc0fafdc9d2e1720f37136443d41d96424966ec4fd43c35be786e110,
195 zeroSpot, 200 defaultSpot, 173 bigSpot detector members с доступными
primary-кандидатами. Более широкое raw-хранилище не входит в этот этап.
Все файлы остаются retrospective development; fresh holdout отсутствует.

## Source binding и пустые экспорты

Для zeroSpot/bigSpot каждый eligible direct-primary n/p record сопоставляется
с полным frozen primary-каталогом по точным PDG, Ekin, Px/Py/Pz и vX/vY/vZ.
Требуем единственный source file, неизменные Row/Id, отсутствие неоднозначных
или повторно использованных source tracks/events среди matched records.
Никакой постоянный сдвиг имён, ближайший сосед или подгонка не применяются.
defaultSpot использует уже проверенную карту4435670: 186 связанных файлов и
14 header-only vacs, сохраняя исходные hashes и прежний failed evidence.

Пустой экспорт означает проверенные headers и ноль строк в обоих hits/vacs.
Он остаётся в frozen inventory с hashes, нулём экспортированных строк и
нулём кандидатов, но без утверждения primary binding. Число обработанных
ROOT events, включая zero-response events, остаётся неизвестным (null).
Если hits непустой, а соответствующей truth association нет, gate закрыт.
Непустой файл без однозначного binding также закрывает gate. Одинаковые
header-only bytes допустимы; одинаковые непустые detector exports — отказ.
Пустой после physics cuts файл учитывается отдельно от header-only экспорта.
Это достаточный учёт для conditional cluster metrics; он не даёт знаменатель
generated-neutron efficiency. Решение записано до full model scores.

## Recipe и ожидание

Сохраняются исправленные CSV, путь-зависимый seed42 jitter sigma0.15ns,
аудит label contrast r3.6/time1.5ns и model transfer r3.6/time1.0ns.
Это прежние conventions, а не новая оптимизация. Их изменение изменило бы
кандидатов и нарушило сопоставимость с first-file probe4435668.
В transfer идут три GraphSAGE checkpoint seeds42/123/456, cuts и train988975
scalers из student_pipeline_20261020_56bcc48206; ничего не fitting на SMASH.
MC truth служит для targets и проверки соответствия, не для forward.

Ожидание: полная популяция может изменить first-file AUC; направление и
величина не установлены. Разница student≈0.98 против native DCM≈0.91
не является изолированной абляцией timing, singleton selection, labels,
source split или objective. Дополнительная потеря на SMASH пока не доказана.
Отдельно покажем размер кластеров и фиксированный native/relaxed label contrast.
Не выбираем recipe или threshold по новым SMASH scores.

## Конечный бюджет и gates

1. Login-02: только упаковка/хеширование/копирование, submission и metadata.
   Raw копии per potential распаковываются и проверяются на compute node.
2. Два CPU exact source-map jobs, zeroSpot/bigSpot: rocky/type_d, CPU4/GPU0,
   не более60мин каждый, mem0, no-requeue. Основание бюджета — defaultSpot
   полный exact scan200 primary files завершился за17мин58с. Две задачи
   максимум одновременно; исходные данные read-only.
3. После binding gates — три CPU full audits, CPU4/GPU0, до2часов каждый,
   максимум две задачи одновременно. Каждый вход SHA-verified, geometry и
   duplicate gates обязательны; unscaled observations сохраняются.
4. После всех full audits — один test-partition CUDA smoke, до20мин, GPU1,
   CPU4. Он проверяет новые accounting gates и сохранение first-file output.
5. Full frozen transfer запускается только после exit0 smoke. Бюджет до2часов
   на potential, GPU1/CPU4, rocky/type_a, mem0, no-requeue, максимум две задачи
   этапа одновременно. Это верхняя граница, оценённая по first-file CUDA
   inference и195/200/173 файлам; actual elapsed и counts сохраняются.

Не автоматизируем повтор после нового инфраструктурного или scientific gate
отказа. Ошибка нашего нового accounting code может быть исправлена только
с отдельным записанным bounded engineering continuation, сохраняя failed
source/output. Не запускаем SMASH training, sweeps или новую test evaluation.
Обучение SMASH требует отдельного source grouping/split/protocol.

## Представление результата

ROC/AP, purity/efficiency при frozen DCM cuts и native max-hit energy error
показываются отдельно по potential и seed, с числом clusters/MC positives.
Mean±sample SD по трём DCM training seeds не является MC/source uncertainty.
Seeds не объединяются в одну ROC population. Разные potentials не считаются
парными физическими событиями. Source-job independence не установлена.
Без processed-event range не рассчитываем generated efficiencies, physical
unfolded spectrum, neutron-track resolution или чувствительность к U_sym.
Новые результаты не входят в manuscript без отдельной проверки.

## Дополнение до full model scores

Source scan4436975 установил exact byte duplicates в bigSpot:87 непустых
exports связываются с51 source file,36 members — точные aliases. До model
оценки фиксируем явную canonicalization: лексикографически первый member
с одинаковыми hits+vacs SHA256 и тем же exact primary binding входит в
analysis view, все остальные остаются в исходном frozen inventory/ledger
с duplicate_of. Ни один иной detector realization не исключается этим
правилом. Source overlap после canonicalization по-прежнему закрывает gate.
Пустые exports остаются отдельно, без придуманного processed-event count.
Эта policy меняет только повторный учёт идентичных входов, без score tuning.

Человек явно разрешил один30-минутный CPU повтор zeroSpot4436974. До submission
создан новый полный primary bundle195files/30490818560bytes; все126 ранее
прочитанных primary hashes сохранены. Новый helper проверяет fstat того же
открытого descriptor до/после копирования вместо повторного pathname stat;
по каждому файлу и архиву сохраняются SHA256. Это не утверждение об исправлении
Lustre. Старый partial archive не используется. Recovery4437145 читает raw
и primary с двух отдельно hash-verified node-local copies, точные matching
keys и frozen inventory не меняются. Лимит30мин, CPU4/GPU0, no-requeue;
дальнейших автоматических повторов нет.

4437145 завершился новым EACCES на scratch atomic replace progress, без
complete summary. Этот отказ не даёт права на ещё один job. Подготовлен,
но не разрешён, отдельный local-output proposal: тот же mapper и inputs,
local filesystem проверяется явно, complete результат экспортируется одним
immutable архивом без scratch replace. Будущие CPU audit/GPU transfer jobs
также пишут рабочие результаты локально и публикуют один проверяемый bundle;
число вычислительных tasks, лимиты и scientific recipe не меняются. Все
complete gates дополнительно требуют Slurm exit0 и hash-verified receipt.

Человек отдельно подтвердил «Разрешить один запуск» для local-output proposal.
Задание4437449 ограничено30мин/CPU4/GPU0 и сохраняет неизменные mapper9f9c2bdb90,
inventory9ca9df3edc и два input manifests. `/tmp` и тип local filesystem
проверяются в runtime. Intermediate/access receipts записываются в home;
scratch получает только один complete compressed summary после полного scan.
Архив проверяется по compressed и uncompressed hashes через открытый descriptor;
временные scratch progress files не создаются. Дальнейших автоматических
повторов эта отдельная авторизация не разрешает.

Для full audits из verified bundles исходное raw-хранилище повторно не
перечисляется. Membership берётся из frozen inventory, каждый прочитанный
hits/vacs и primary hash проверяется по bound inventory и archive manifest.
Это исключает ненужное обращение к unrelated Lustre paths и сохраняет тот же
analysis cohort и filename-derived jitter seed. Legacy audit без staged copy
сохраняет проверку исходного directory inventory. Отдельный test скрывает
исходный каталог и требует идентичные saved observation tensors из копии.

CUDA smoke4437645 завершился exit0. При локальной проверке дополнительное
требование побитно одинаковых float32 predictions не прошло: max|delta score|
7.74e-7, max|delta E_pred|9.60e-7GeV. Original observation hashes, candidate
identities, labels, targets, checkpoint hashes, cuts, AUC и selected sets
совпали точно. Для численного gate используется существующий библиотечный
`torch.testing.assert_close` с defaults для float32; tolerance не выбирается
по SMASH quality. Metrics каждого запуска пересчитаны из его собственного
CSV прежним relative1e-6/absolute1e-8 check; cross-run energy summaries
сравниваются тем же float32 library default, что predictions. Полный
diagnostic сохранён, побитная воспроизводимость не
утверждается; механизм малых различий отдельно не изолирован. Нового compute
retry, изменения scientific recipe или fitting нет. Это уточнение локального
engineering comparison, scientific gates остаются прежними.

Уже отправленные full transfer4437674/4437675/4437676 ожидают GPU. Пока они
PENDING, запрошенные лимиты снижены до1ч zeroSpot/defaultSpot и30мин bigSpot,
в пределах исходных верхних границ2ч. Линейная оценка по measured first-file
CUDA durations1.8–4.5с, setup27.4с и195/186/51 canonical files оставляет
не менее трёхкратного запаса. Это operational budget, не новая model recipe
или настройка по quality. Число задач, dependencies и предел concurrent2
сохранены. TIMEOUT не даёт права на автоматический повтор.
