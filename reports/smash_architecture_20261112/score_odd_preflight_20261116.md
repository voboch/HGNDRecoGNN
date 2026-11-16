# ODD preflight: delta от принятого EVEN, 16 ноября 2026

Последний принятый научный результат — EVEN architecture reduction от 16 ноября, с воспроизведёнными потолком и выбранными порогами. Последние модельные задания — 18 EVEN inference задач массива 4522048, завершение 16 ноября 18:50 MSK. Работа после них — приёмка и замораживание политики в тот же рабочий день; дрейфа нет.

**Verdict: GO.** Допуск ровно на `submit_score_odd_20261116.sh`: 18 задач ODD inference, одновременно две, лимит 20 минут. Обучения, дополнительных тестов и оптимизации на ODD нет. Прямая оговорка пользователя об отсутствии новых тестов сохранена. Действуют runtime source/input/fit/release guards.

Локально сверены SHA-256 `odd_release_20261116.json` (`680506ddbdc0a66b7cbcb4e1f1b504943e3fdbe811029377c8bc7097ad5698c8`), EVEN summary, ledger и frozen policy с launch plan и release. Политика имеет SHA `8587f2bbdc750adc619daa46d4c707694c6416e652ef0e0dd372e81f56570f69`. Все 18 fit SHA точно совпадают с принятым индексом; source совпадает с EVEN. Выбранные targets совпадают в самой политике, release и launch plan: student_sage 90, student_dynedge 75, smash_hetero_sage 90. Первичная оценка остаётся raw70.

Прочитаны `score_even_acceptance_20261116.json`, `score_even_execution_20261116.json`, `even_acceptance_20261116.json` и `even_independent_replay_20261116.log`. Запись EVEN execution фиксирует старое промежуточное `comparison_state: not yet reduced`; последующие result и even_acceptance уже содержат принятую редукцию. Это историческое поле не закрывает ODD gate повторно. Лог независимого replay сообщает воспроизведение всех выбранных targets/ceiling, matching и всех preselection energy bins. Проверку повторно не запускал.

| Проверка delta | Статус | Основание |
|---|---|---|
| EVEN принят, политика закреплена | ok | Сверенные SHA result/policy/release и acceptance |
| Fit/source остаются прежними | ok | Полное совпадение источника и 18 fit SHA |
| ODD читает exact release | ok | `smash_architecture.py:145–149`; wrapper `:59–62`; исправленный reducer T1 сохраняет policy binding |
| Команда и путь | ok | `submit_score_odd_20261116.sh`: root `slurm/`, scope odd, array 0–17%2, 20 минут |
| Очередь и дубликаты | ok по свежему сообщению ведущего; guards сохранены | Queue empty после EVEN; launcher проверяет jobs, marker и score_odd receipts |
| Публикация release/policy | предусмотрена перед submit | Ведущий публикует immutable 444; launcher проверяет оба точных SHA до sbatch |
| Лимит и стоимость | ok | EVEN max 418 с; 2×418+300=1136 с, округление вверх до 1200 с; ODD max 397060 графов против EVEN 397130 |
| Stop policy | ok | no-requeue, сохранение ошибки, без автоматического повтора |

Ожидание по масштабированию измеренного EVEN — около 1,65 GPU·ч, или 0,83 ч при двух слотах без очереди. Scheduler cap — 18×20 минут = 6 GPU·ч, до 3 часов при двух слотах без очереди. Запуск использует унаследованные type_a/V100, 4 CPU, mem=0 и exclusions неизменного wrapper. При ошибке остановиться и установить механизм; этот GO не разрешает автоматическую повторную отправку.

Результат закрывает зарегистрированное сравнение raw70 на ODD и дополнительную оценку точно замороженной EVEN-политики для NST/раздела реконструкции диссертации. Нечётные файлы не используются для изменения порогов. Дополнительные четыре поля energy replay сохраняют уже вычисленные выборки и не меняют estimator или scorer source. Обзор ограничен delta; SSH, задания, новые тесты и изменения других файлов не выполнялись. Блокеров не найдено.


## Delta guard, 16 ноября 19:00:29 MSK

**Verdict: GO после исправления одной строки duplicate guard.** `score_odd_guard_check_20261116.json` подтверждает: широкий шаблон нашёл только `score_odd_launch_plan_20261116.json`; числовых score receipts нет, submission marker отсутствует, очередь пуста. Остановка exit 4 произошла до sbatch, модельное задание не создавалось.

Прочитана исправленная строка 7 `submit_score_odd_20261116.sh`: `score_odd_[0-9]*_[0-9]*.json`. Она покрывает зарегистрированное именование task/job receipt и не совпадает с launch plan. Marker guard и команда sbatch сохранены; source/policy pins не менялись. Предыдущая широкая маска была ложным срабатыванием проверки дубликатов. Исправление устраняет установленный механизм; новый полный preflight или дополнительные тесты не требуются.
