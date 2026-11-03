# No-floor arm: реализация и граница готовности

03.11.2026. После принятого census A реализованы target adapter, CPU
preparation и тонкий training/validation runner. Предзапускный технический
критик дал **Code GO**. **Submission GO ещё нет**, новых задач этого arm нет.

## Реализовано

- `data/label_ablation.py`: native replay 0,3 ГэВ на accepted membership,
  no-floor targets через прежний `threshold_ledger`, immutable candidate keys,
  target-only patch с проверкой неизменности каждого прочего tensor.
- `scripts/label_ablation_prepare.py`: все 578 зарегистрированных content
  groups в union train/validation, включая unbound и нечётные роли; exact
  deterministic re-observation; только compact patches и manifest. Нет
  primary-binding требования, повторного scaler fit или F-метрик.
- `scripts/label_ablation_train.py`: reuse `smash_train.train_run`, прежнего
  recipe и `NativePackDataset`; одна planned boundary 15+15 при total30.
  В stable identity нет job/segment IDs; coherent mirror проверяется.
  Итоговый selector и полная история проверяются прежним trainer.
- Frozen baseline B0 cut того же fold/seed; raw energy, без refit cut или
  soft calibration. Validation scores e_cls и energy e_en экспортируются
  с native/disabled references и ключами кандидатов. Common positives и
  добавленные soft объекты имеют раздельные bias/MAE, включая target=0.

Exact replay/structural assertions реализованы в исполняемом пути, но их
выполнение на всех 578 группах **ещё не состоялось**. Прошлый accepted replay
S286 не подменяет новый preparation receipt. Stage A source/artifacts сохранены.

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

Подготовить конкретный CPU launch plan и Slurm wrapper, новый immutable
source package, output paths и caps; предъявить их submission preflight.
После принятого CPU preparation зафиксировать SHA patches и конкретные
GPU submissions: девять новых моделей, максимум две задачи одновременно,
два сегмента короче 3 ч на модель. Ресурсы/receipts ещё не подготовлены,
поэтому настоящий документ не является разрешением на подачу.

Прежний бюджет остаётся: 27-28 GPU·ч pure fit, 30-33 с validation/I/O.
30 эпох не расширяются; промежуточный floor не выбран; baseline не повторяется.
