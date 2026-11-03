# No-floor arm: реализация и граница готовности

03.11.2026. После принятого census A реализованы target adapter, CPU
preparation и тонкий training/validation runner. Предзапускный технический
критик дал **Code GO**, затем GO на один CPU submission после удалённой
проверки hashes. CPU preparation4482805 завершена и принята:578 patches,
exact native replay,7:02 на cn-035. Конкретный GPU preflight дал GO на
две плановые arrays; поданы 4482848 и 4482849. На 14:13 МСК PENDING/Priority
и PENDING/Dependency, после whole-array afterok первой. Новых моделей пока нет.

## Реализовано

- `data/label_ablation.py`: native replay 0,3 ГэВ на accepted membership,
  no-floor targets через прежний `threshold_ledger`, immutable candidate keys,
  target-only patch с проверкой неизменности каждого прочего tensor.
- `scripts/label_ablation_prepare.py`: все 578 зарегистрированных content
  groups в union train/validation, включая unbound и нечётные роли; exact
  deterministic re-observation; только compact patches и manifest. Нет
  primary-binding требования, повторного scaler fit или F-метрик.
- `scripts/label_ablation_train.py`: reuse `smash_train.train_run`, прежнего
  recipe и `NativePackDataset`; одна planned boundary 15+15 при total 30.
  В stable identity нет job/segment IDs; coherent mirror проверяется.
  Итоговый selector и полная история проверяются прежним trainer.
- Frozen baseline B0 cut того же fold/seed; raw energy, без refit cut или
  soft calibration. Validation scores e_cls и energy e_en экспортируются
  с native/disabled references и ключами кандидатов. Common positives и
  добавленные soft объекты имеют раздельные bias/MAE, включая target=0.

Exact replay/structural assertions исполнены для всех 578 при preparation;
локально проверены 579 archive members, все patch SHA/keys/targets, metadata
исходных observations/packs и полный S286 replay. Принят новый receipt,
прежние Stage A source/artifacts сохранены. Proof - `prepare_4482805_acceptance.json`.

## Что проверено сейчас

AST parsing трёх новых файлов прошёл; ToF/threshold reader SHA совпадают с
принятым A `audit_summary.json`. Дополнительные unit tests, GPU smoke,
training или inference не выполнялись. Human waiver не обозначен как PASS.
Actual Slurm: 4481662 и 4481705 terminal COMPLETED/0:0; новых campaign GPU
jobs в очереди нет. Чужие задачи не менялись.

## Предзапускная критика

Read-only critic `/root/label_ablation_diff_preflight`: **Code GO**.
Исправлены выявленные gaps: CPU source manifest проверяется целиком;
patches и GPU reader связаны одним immutable source SHA; exact accepted
split/prepare SHA закреплены; исходные ToF/threshold helpers закреплены
SHA из A. Native target math/order, zero-energy population, роли split,
planned boundary и stable resume identity приняты без оставшихся blockers.
Выполнялись только чтение кода/receipts и проверки синтаксиса/SHA.

Текущий runner экспортирует validation нового arm. Совместное baseline/new
сравнение по reference labels и будущая F-оценка остаются следующим этапом;
их результатов сейчас нет. Baseline validation NPZ не содержит candidate
keys, поэтому ключи для его будущего соединения нужно восстановить из тех
же pinned native packs, а не соединять по label/energy.

## Следующий шаг

Новый immutable source sums921e04a4e6d1... (commit 721beff) проверен локально
и удалённо:262 entries, все read-only, A не изменён. Paths/caps и CPU review
сохранены в `source_package.json`, `launch_plan.json`, `submission_review.md`.
Факт единственной CPU подачи - `prepare_4482805_launch.json`, завершение и
приёмка - `prepare_4482805_completion.json`/`prepare_4482805_acceptance.json`.
Повтор CPU не нужен. Patch SHA и конкретные GPU submissions закреплены в
`gpu_submission_plan.json`, verdict - `gpu_submission_review.md`:9 новых
моделей,максимум 2 задачи,15+15 total 30,2:45 на segment. Два массива поданы
один раз с afterok ВСЕЙ первой array; actual controls и receipts находятся
в `GPU_submissions.json`. Наблюдать их, не дублировать и не делать retries.

Прежний бюджет остаётся: 27-28 GPU·ч pure fit, 30-33 с validation/I/O.
30 эпох не расширяются; промежуточный floor не выбран; baseline не повторяется.
