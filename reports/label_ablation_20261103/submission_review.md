# Один CPU submission: GO

03.11.2026. Read-only reviewer `/root/label_ablation_diff_preflight`
разрешил **один CPU job после удалённой проверки hashes**. GPU submission
этим verdict не одобрен. Source commit721beff, sums921e04a4e6d1..., точные
SHA и paths - `source_package.json`.

Проверены все 262 source entries: SHA точны, файлы read-only. Split,
prepare, human waiver и девять baseline receipts совпадают с pins.
Root CODE_EXTRAS присутствуют. CPU argv соответствует своим readers;
модели, F-метрики и scaler fit не вызываются. Resources: rocky/normal,
type_d,4CPU/0GPU,mem0,45мин,2workers,no-requeue. Запас соразмерен
принятому census S286 5:04; нового recovery controller нет.

Runtime guards обязательны: observation/native target exact replay,
superset и неизменность остальных tensors. Publisher читает complete
summary и создаёт один compact archive/manifest, проверяя SHA. Paths
новые, A не затрагивается. Новые промежуточные thresholds не выбираются.

Перед sbatch основной агент проверяет удалённый SHA/read-only inventory,
отсутствие campaign jobs/старой submission receipt, наличие home receipt
directory и logs в submission cwd. Последние научные результаты приняты
сегодня; последний model job29.10; инфраструктурного дрейфа нет.
Reviewer не запускал SSH, jobs, tests, inference или fit.

Удалённая приёмка пакета выполнена: 262 hashes и read-only flags точны,
prepare manifest pinned, source sums A неизменны; receipt directory/logs
и новый campaign существуют. Первый sandbox SSH не разрешил DNS;
тот же read-only запрос с сетевым разрешением выполнился. Это не сбой
научной задачи или повтор её подачи. Job до настоящего gate не подавался.
