# Отдельная ветка student UFC — 19 октября 2026

Подготовлен точный перенос шести архитектур и отдельный путь candidate
inference. На одинаковых случайных весах выходы всех шести перенесённых
моделей совпали с определениями тетрадок: максимальная абсолютная разница
0.0 на проверенной выборке графов из 32 training events. Это инженерная
проверка переноса, не воспроизведение обученного ROC-AUC.

Протокол: `student_ufc_integration_protocol_20261019.md`. Исходный аудит
`student_notebook_audit_20261019.md` сохраняет выводы о первоначальном коде.

## Реализация

- `models/student_ufc.py` содержит исходные residual blocks, attention/mean/
  max/sum pooling и обе головы. Имена модулей state dict сохранены.
  `models/student_ufc_sources.json` связывает классы с hashes тетрадок и
  отмечает исправленные оборванные assertions. Тетрадки не изменены.
- `data/student_ufc.py` строит графы из наблюдаемых колонок. MC targets
  добавляются отдельной функцией после формирования графов. Удаление всех
  MC-полей позволяет выполнять inference; невалидный truth не используется
  для незаметного удаления измеренных хитов.
- `training/student_ufc.py` экспортирует candidate score/energy с source,
  event и candidate IDs. Предсказания сначала собираются через все батчи,
  затем обрабатывается полное событие. Учитываются пустые события и
  максимальное взаимно однозначное сопоставление кандидатов и truth IDs.
- `scripts/student_ufc.py` загружает оригинальный формат checkpoints,
  запускает inference либо пересчитывает AUC отдельно для каждого seed
  из CSV. Дубли ключей и разные популяции между сравниваемыми seeds/models
  отклоняются. При отсутствии ожидаемого seed средний AUC и seed SD
  остаются `null`; pooled AUC не подставляется.

Текущие `net_default`, training registry и упрощённые `thesis_*` не изменены.
Новые модели работают через отдельный homogeneous UFC путь. Их интеграция
в основной training loop и повторное обучение на GPU ещё не выполнены.

## Проверки

Новые 16 tests проверяют label-free признаки, сохранение пустых событий,
различие профилей времени, batch-independent event metrics, максимальное
matching вместо ошибочного greedy matching, merged/duplicate objects,
комплектность seeds и strict checkpoint roundtrip для всех шести моделей.
Полный набор: **140 passed**, 13 существующих предупреждений. Ruff и
компиляция новых Python-файлов прошли.

`scripts/validate_student_ufc.py` дополнительно сравнивает standalone-код
с определениями оригинальных тетрадок и запускает реальную inference-команду
на временных случайных весах и локальных хитах без MC-колонок. Такой
ограниченный запуск помечается как engineering smoke; ROC-AUC не выводится.
Артефакт: `results/student_notebook_audit_20261019/port_validation.json`.

На этих 32 training events получены 33 кандидата. Признаки, рёбра, MC targets,
candidate IDs и четыре Laplacian coordinates совпали с notebook builder;
максимальная разница признаков и PE равна 0.0. Полный standalone builder
повторно обработал все 197027 событий и получил прежние counts: train
168108 кандидатов / 45740 positives, validation 36269 / 9924, test
36343 / 9857. Это проверка preprocessing, без оценки обученного test AUC.

Профиль времени задаётся явно. `fTime_nonoise` воспроизводит исходный рецепт;
`fTime` создаёт другой recipe hash. Unit test подтверждает изменение ToF-
признака при переходе к детекторному времени. Качество с этим временем не
измерено. Все геометрические, временные и multiplicity настройки наследуют
рецепт студента; этот перенос их не оптимизировал.

Object efficiency относится к переданному списку eligible truth objects
после hit selection. Для эффективности относительно всех сгенерированных
нейтронов нужен дополнительный generator-level inventory. Наличие отдельного
CSV событий сохраняет события без кандидатов, которые старый inference
терял. Отсутствующие в самом входном pickle события код не изобретает.

## Использование после получения оригинальных весов

Пример повторной оценки одного оригинального checkpoint. `--seed` —
зафиксированный training seed, который нужно подтвердить receipts; старые
checkpoints не всегда содержат его в metadata. Разбиение `test` ниже
относится к исходному файлу студента, не к source run 988976.

```bash
python scripts/student_ufc.py infer \
  --model baseline_graphsage \
  --checkpoint /absolute/path/to/student_seed42.pt \
  --hits-pickle ../NeutronRecoGNN/data/Jan26/hits_Jan_26.pkl \
  --time-column fTime_nonoise --split test --seed 42 \
  --with-truth --output /absolute/path/to/new_student_seed42_evaluation
```

Результаты — `predictions.csv`, полный `events.csv`, `metrics.json` и
`manifest.json` с hashes входа/весов/кода, recipe и выбранной части событий.
Каталог результата должен быть новым. Operating point не подбирается по
test: событийные метрики вычисляются только при явно переданном
`--threshold`, замороженном на validation.

Если доступны готовые CSV студента:

```bash
python scripts/student_ufc.py csv-metrics \
  --predictions /absolute/path/to/baseline_graphsage_ALL_SEEDS_test_cluster_predictions.csv \
  --source-id student-hits-Jan26 \
  --output /absolute/path/to/new_per_seed_auc.json
```

Эта команда подтверждает арифметику предоставленных строк. Она сама не
доказывает происхождение weights/splits или полноту CSV относительно
исходных шардов. Count и positives каждого seed записываются в результат.
Для полного воспроизведения их надо сверить с исходными manifests.

## Текущий предел результата

Оригинальные checkpoints/CSV по-прежнему не предоставлены. Нового обучения
и заданий кластера не было. ROC-AUC 0.98 остаётся сохранённым результатом
студента, ожидающим независимой оценки. Рукопись и закрытые DCM кампании
не изменялись; source 988976 повторно не оценивался.
