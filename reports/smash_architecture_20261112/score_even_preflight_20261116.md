# Preflight EVEN, 16 ноября 2026

Последний научный результат, использованный в решении: capped-energy, 12 ноября (дату сообщает ведущий аудита; см. `.claude/paper-review.md`, раздел Number sources). Последние модельные задания: финальные fit завершились 14–15 ноября, последнее — 15 ноября 07:27:55 MSK (`fit_acceptance_20261116.json`). После этого текущая работа — приёмка и аудит в пределах одного рабочего дня; порог инфраструктурного дрейфа не превышен.

**Verdict: GO.** Только EVEN inference по `score_even_launch_plan_20261116.json`, задачи 0–17, одновременно не более двух. Новое обучение, новые тесты и ODD не входят в допуск. Блокеров запуска не найдено. Следующий шаг — один массив скоринга, затем приёмка 18 экспортов и EVEN-редукция зарегистрированного сравнения трёх архитектур; основной raw70 и дополнительная EVEN-политика остаются раздельными.

## Проверенный путь

Локальная сверка SHA-256 подтвердила хеши fit acceptance и fit index из launch plan. Все 18 записей индекса совпадают с принятыми финальными публикациями. Source одинаков в launch plan, fit acceptance и fit launch plan: `2d31e3e25c45dff5b15818357e8d3d185cc752ef229a726971ab234f8f5d3969`. Хеш `plan.json` совпал. Старый `source_package.json` не использован как активный источник. Побайтовое сравнение `scripts/smash_architecture.py` и `slurm/smash_architecture.hpc.sbatch` с commit `1a87c5b6f4dea71580680fc9f70d3cbc5ce790f2` подтвердило неизменность обоих файлов.

`slurm/smash_architecture.hpc.sbatch:51` проверяет хеш индекса и передаёт receipt конкретной задачи; `scripts/smash_architecture.py:132` проверяет receipt, завершённые 30 эпох и полную identity. Строки 150–168 проверяют выбранные веса, берут пересечение TEST-fold с EVEN и запускают обе выбранные головы без fit. Подготовка и роли закреплены в строках 43–66. Выходные ключи включают scope, task и Slurm job ID (`sbatch:64`), локальный каталог создаётся через mktemp. Скрипт не содержит автоматического повторного запуска.

Принятый probe 4508255 имеет тот же source hash. Новые unit tests и test-queue задания не выполнялись: прямое указание пользователя имеет приоритет над общим HPC smoke-требованием; это также отражено в `.claude/paper-review.md`. Существующие runtime SHA/role/checkpoint проверки сохранены. Полные промежуточные resume blobs повторно не проверялись; использована принятая fit-приёмка 72 сегментов/18 моделей.

Локальный diff `scripts/smash_architecture_reduce.py:690` закрывает T1 по прочитанному коду: хеш закреплённого odd release проверяется до чтения политики, SHA политики берётся из release; каждая odd prediction привязана к тому же release, plan и fit (`:252`), численные cuts и fit bindings проверяет `frozen_targets` (`:671`). EVEN отвергает odd-only аргументы. Новых вычислений для проверки diff не выполнялось. Это не разрешение ODD: сначала принятый EVEN result и новая квитанция с точным SHA frozen policy и всеми 18 fit SHA.

## Оценка времени — замечание исправлено

Первоначальная оценка 1–2 ч уточнена ведущим в launch plan до завершения проверки. Сумма `conservative_seconds` по шести комбинациям и трём seed составляет 8,58 GPU·ч, около 4,29 ч при двух постоянно занятых GPU. Только inference: 1,29 GPU·ч, около 0,65 ч wall. Верхний scheduler budget: 13,5 GPU·ч, около 6,75 ч при двух слотах без ожидания очереди. Максимальная консервативная задача — 1843 с, меньше лимита 2700 с. Открытого замечания по оценке времени нет.

| Проверка | Статус | Основание |
|---|---|---|
| Frozen source и план | ok | Локальная сверка SHA и commit bytes |
| 18 финальных fit | ok | Индекс совпадает с acceptance; 30 эпох |
| TEST/EVEN, scaler/preparation | ok | Неизменный scorer, закреплённая preparation |
| Queue, отсутствие score duplicate | ok по сообщению ведущего | Пустая очередь account 16 ноября 17:48 MSK, score receipts отсутствуют; reviewer SSH не выполнял |
| Resources | ok | type_a, 1 GPU/task, 4 CPU, mem=0, array 0–17%2, 45 мин, no-requeue |
| Уникальные выходы | ok | scope/task/job ID, mktemp |
| Stop policy | ok | Ошибка сохраняет выход; без автоматических retries |
| ODD | закрыт | Нужны принятый EVEN и точный frozen release |

Прочитан конкретный `submit_score_even_20261116.sh`: одна отправка с явными `--array=0-17%2 --time=00:45:00`, `ARCH_PHASE=score`, `ARCH_TEST_SCOPE=even`, правильными source/campaign/receipts и закреплённым fit index. Перед отправкой проверяются source bytes, индекс, отсутствие архитектурных заданий и старых score receipts/submission marker. Эти аргументы переопределяют probe defaults неизменного шаблона.

Проверка ограничена локальными источниками и сообщённым свежим состоянием очереди; никаких SSH, submit, тестов или изменений проверяемого кода. Прочитаны AGENTS.md, HPC.md, HPC_DATA.md, shared conventions README и §30, `.claude/paper-review.md`, роль preflight-reviewer, планы, acceptance/index и diff reducer. Отдельный no-ai-slop skill не найден в доступных локальных каталогах; применён чеклист AGENTS.md. Результат предназначен для архитектурного сравнения и раздела реконструкции в NST/диссертации; научный победитель до принятой редукции не объявлен.


## Delta после ошибки пути, 16 ноября 17:51 MSK

**Verdict: GO для исправленного пути запуска.** Первая команда sbatch завершилась до создания задания: `Unable to open HGNDRecoGNN/slurm/smash_architecture.hpc.sbatch`. Ведущий проверил фактический frozen package: wrapper расположен в `root/slurm`, Python package — в `root/HGNDRecoGNN`; source SHA-проверка прошла, очередь пуста. Предыдущий обзор не установил правильное расположение wrapper на удалённой стороне.

Повторно прочитан только изменённый launcher и запись механизма в launch plan. После `cd` в закреплённый source launcher теперь передаёт `slurm/smash_architecture.hpc.sbatch`; значения scope, fit index/hash, array, времени и ресурсов сохранены. Новый marker `submission_score_even_20261116_corrected_path.txt` проверяется перед отправкой и принимает вывод sbatch; старый пустой marker сохраняется. Исправление соответствует сообщённой раскладке файлов. Модельная попытка не начиналась, научный путь не изменён. Новые тесты, SSH или повторный полный обзор не выполнялись.
