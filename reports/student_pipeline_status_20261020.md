# Native HGND check: статус 20 октября 2026

Обновление 21 октября 2026: все12 train runs4424084 и test4424085 завершены, exit0.
Hashes и frozen validation thresholds проверены. На reused test988976 получен
mean cluster ROC-AUC0.906–0.910 по трём seeds на модель. Canonical результат —
`student_pipeline_results_20261021.md`. Ниже сохранён статус на момент
отправки20октября; ожидание Priority/Dependency уже завершилось.

Подключены четыре точных student backbone/head families без GATv2 к штатным
HGND graph membership, десяти признакам, native cluster loss, shared fit и
prediction export. Новое обучение необходимо: прежние six-feature веса
относятся к другой задаче. Качество этой новой серии пока не измерено.

Локальная проверка: 161 tests passed, 13 предупреждений существующего suite;
семь новых tests проверяют cluster/event isolation, singleton, отсутствие
truth inputs, packing/remap, native export и cohort barrier. Ruff, compilation,
shell syntax и strict table lint нового протокола прошли.

GPU preparation/smoke 4424014: COMPLETED, exit0, 12мин30с, partition=test,
V100 cn004. Все четыре модели прошли native fit, backward, checkpoint
roundtrip, coherent atomic model/optimizer generation, CUDA RNG replay,
singleton и batch isolation; предсказания не меняются при удалении truth.
Полная train array **4424084**, 0–11%2, принята и ожидает Priority.
Test job **4424085** ожидает afterok всей array; при failed dependency он
отменяется, а программный barrier дополнительно проверяет все12 receipts.
Test 988976 в этой серии ещё не оценён. Все старые DCM и student jobs завершены;
их snapshots, weights и результаты не изменены.

Immutable source:
`/home/vbocharnikov/hgnd_experiments/student_pipeline_20261020_56bcc48206`,
manifest SHA256 `56bcc48206186e32e182a801410d42865b7f25df656d85166e7a64f67a67fe96`,
113 files проверены локально и при развёртывании. Новый campaign:
`/scratch/vbocharnikov/hgnd/experiments/student_pipeline_20261020_56bcc48206`.

Утверждённый split manifest и train-only scalers сохранены: train988975,
validation988974, reused test988976. Test уже открывался в прежней серии;
новая проверка не будет независимым подтверждением. Неполный cohort не
усредняется, threshold на test не подбирается.

Протокол: `student_pipeline_protocol_20261020.md`. Машинный статус и квитанция:
`results/student_pipeline_20261020/launch_status.json`, `smoke_submission.json`.
Отправлены конечные12 train runs (seeds42/123/456, 30epochs, максимум два
одновременно) и один test job afterok всей array. Следующий шаг — читать
сохранённые receipts и Slurm4424084/4424085; не дублировать активные или
завершённые задания. После оценки canonical metrics появятся в campaign
`test_evaluation/summary.json` и `per_seed.json`. Пока новых quality metrics нет.
Квитанция серии: `results/student_pipeline_20261020/train_test_submission.json`.
