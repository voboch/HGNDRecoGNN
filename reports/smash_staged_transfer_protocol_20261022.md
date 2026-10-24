# SMASH transfer с чтением проверенной локальной копии

22 октября 2026. Пользователь после отчёта о failed4435592 поручил
«переходи к smash». Продолжаем конечный SMASH этап: одна новая CPU source-map
проверка после переноса corrected detector inputs в локальный scratch узла,
один CPU audit первых проверенных файлов и один CUDA transfer smoke.
Предыдущие источники, receipts и failed jobs сохраняются. Автоматического
повторения после нового отказа нет. Закрытые DCM campaigns не возобновляются.

## Данные и предсказание до измерения

Исходный inventory SHA256:
9ca9df3edc0fafdc9d2e1720f37136443d41d96424966ec4fd43c35be786e110.
Bundle содержит200 frozen corrected detector hits/vacs pairs defaultSpot,
первые zeroSpot/bigSpot hits/vacs и три проверенных primary exports.
Файлы копируются побайтно, SHA256 проверяется при создании и распаковке.
Копия меняет только место чтения. Она не устанавливает причину EACCES и не
доказывает, что исходное хранилище исправлено.

Для первых файлов mapping задан по уже проверенной MC identity:
zeroSpot0001→primary0001, defaultSpot0001→primary0002,
bigSpot0001→primary0001, run10943245. Проверки охватили1264/334/1216
primary n/p records. Новый engineering inventory перечисляет только эти
три detector files и явный primary map; оригинальный inventory не меняется.
Эти файлы выбраны как первые engineering probes до model scores. Это
retrospective development; population не представляет весь SMASH dataset.

Ожидание: побайтная копия должна сохранить observed hits, jitter, labels и
predictions при том же source/member. Тест сравнивает их напрямую.
Перенос DCM→SMASH может ухудшить cluster metrics; величина неизвестна.
Никакого ожидания восстановления student ROC-AUC≈0.98 не задаём.

## Бюджет и порядок

1. На login-02 только упаковка/копирование входов, без анализа и обучения.
2. CPU source mapping: rocky/type_d, CPU4/GPU0,45мин, mem0, no-requeue.
   Проверенный bundle распаковывается в отдельный каталог `/tmp` allocation;
   все200 detector vacs читаются оттуда. Primary scan остаётся теми же200
   исходными exports, которые4435586 уже успешно прочитал на cn-039.
   Missing/ambiguous/changed Row/Id не исправляются приблизительным matching.
3. CPU audit первых трёх файлов: rocky/type_d, CPU4/GPU0,15мин, mem0,
   no-requeue. Input bundle проверяется до чтения, primary map явный. Audit
   повторяет прежние r3.6/time1.5ns, seed42 timing noise и label contrast.
   Перестановка primary filenames не меняет detector jitter seed.
4. Только после COMPLETED/exit0 и трёх verified binding summaries — один
   GPU transfer smoke на partition=test/type_a, GPU1/CPU4,20мин, mem0,
   no-requeue. GraphSAGE seeds42/123/456, frozen DCM validation checkpoints,
   cuts, train988975 scalers и input recipe r3.6/time1.0ns остаются прежними.
   MC не входит в forward. Никакого training/calibration на SMASH.

Максимум две задачи этого этапа одновременно; чужие jobs не меняются.
Временные node-local копии удаляются только внутри созданного данной задачей
каталога; архив и результаты остаются на scratch. Code/logs/receipts — home.
До завершения всех data gates длинная full transfer серия не отправляется.

## Значение результата

Первый transfer даёт cluster ROC/AP, fixed-cut purity/efficiency и ошибку
native max-hit MCTrack energy target отдельно по трём potentials и seeds.
Число clusters и MC-positive records сохраняется с каждой метрикой.
Разброс seeds отражает обучение на DCM; он не измеряет source/job variance.
Нельзя складывать три seeds в одну ROC population.

Generated-neutron efficiency и чувствительность к U_sym не рассчитываются:
detector processed ROOT event range пока не подтверждён, включая zero-hit
events. Native maximum qualifying-hit target не называется neutron-track
energy resolution или unfolded spectrum. Новое SMASH обучение требует
восстановленного полного source grouping и заранее заданного split/protocol.

## Ограниченная поправка к исполнению source mapping

4435627 завершился за25с до primary scan на header-only detector vacs0187.
Archive inspection подтвердил14 header-only vacs0187–0200. У `pd.read_csv`
пустые колонки имеют dtype object; `np.isfinite` без numeric conversion в
нашем helper выбрасывал TypeError. Прежний test покрывал conflicting attrs,
но пропустил пустой DataFrame. Новый test воспроизводит этот CSV и требует
нулевой binding evidence, статус unresolved. Исправление — явное float
conversion только для finite check; primary search keys/recipe не меняются.

В рамках текущего поручения пользователя выполняется ровно одно исправленное
продолжение source mapping, с новым immutable script и новым output path.
Оставшийся allocation bound44мин30с: вместе с25с failed task меньше прежних
45мин. Это записанное исключение к запрету automatic retries выше;
дальнейших повторов после нового отказа не будет. Failed source/output
сохраняются. Пустые exports не удаляются из frozen population и не получают
фиктивного primary binding или числа processed events.
