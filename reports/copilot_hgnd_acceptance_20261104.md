# Приёмка HGND Copilot и расход квоты

04.11.2026. Новых платных вызовов при этой проверке не было. Исходные
responses, snapshots, usage и acceptance сохранены в
`outputs/copilot_hgnd_20261104/`; их статусы не переписывались.

## Расход и область учёта

| Задача | Native AI credits | Внутренние API requests | Worker result |
| --- | ---: | ---: | --- |
| 1: Rₙ readiness | 17.931835 | 22 | PARTIAL |
| 2: physical comparison plan | 20.872075 | 7 | PARTIAL |
| 4: energy-window inputs | 21.597780 | 7 | PARTIAL |
| Всего HGND | 60.401690 | 36 | Три ограниченных аудита |

Credits = root `totalNanoAiu / 1e9` каждого usage.json. Во всех трёх
`aiCreditsStatus=complete`, `currentModel=gpt-6.1-terra`, codeChanges=0.
Effort max запрошен; native usage не доказывает фактически применённый effort.
Все inputs/instructions snapshots повторно сверены по их SHA, расхождений нет.

Последнее показание Plan, переданное пользователем: 1068/1500 AIC,
индикатор 71% used, reset через 26 дней. Арифметический остаток — 432 AIC;
это заменяет прежнее приблизительное сообщение о 60% расхода.
Источник — показание пользователя, не самостоятельный запрос account API.
Локальные 60,40 credits HGND составляют около 4,03% лимита и не объясняют
весь месячный расход. Ранее проверенные три HIC-пробы стоили ещё 28,120165 credits;
эти два локальных набора вместе не являются полной месячной ведомостью.
Экономия Codex и время родительской приёмки не измерены.

## Что принимается

- Задача 1 корректно отделяет наличие exports/metadata от независимой
  проверки prediction order и primary coverage. Отсутствие доказательства
  в snapshot не означает отсутствия артефакта в проекте или найденной порчи.
- Задача 2 выявила конкретные уточнения плана: processed span и его
  предположение о последовательной обработке, count-before-ratio,
  независимые draws между потенциалами с общей парностью arms, численная
  определённость ratios и same-split provenance. В локальном плане эти
  уточнения уже внесены без второго обращения к модели. Текущий plan SHA:
  `c872b71a0a8432094635a80503d7a3a338e2fd38998b46dbe9873e07d8441d8a`.
- Задача 4 правильно разделяет fixed-window truth product, uncensored
  primary reduction, MC-label floor, observable bands и actual time selection.
  Ни интерполяция готовых гистограмм, ни census не определяют новый optimum.

Это приёмка ограниченных аудитов. Новых Rₙ, physics closure или окон они
не вычислили. Самостоятельный source/data defect этими ответами не установлен.

## Уточнения по существующим источникам

**Detector side.** В `data/smash_transfer.py:_ordered` side задан как fY>0;
`transfer_pack` сохраняет его в pack.side. `NativePackDataset.__getitem__`
в `data/student_pipeline.py` переносит pack.side в graph.istop;
`training/eval.py:predict` переносит его в cl_istop.
`scripts/smash_transfer_infer.py:check_prediction` прямо сравнивает
pack.side и frame.cl_istop. Это detector-side namespace 0/1; оно не означает
MCTrack Side=5/6 из qualifying-hit condition. На текущем пути семантика
определена кодом, отдельная новая система side-сертификатов не нужна.

**Baseline order.** Copilot верно заметил, что один collector присваивает
baseline scores по позиции до key merge. Сам этот merge не является
независимым доказательством порядка. Однако producer уже есть:
`scripts/smash_train.py:score_pack` использует shuffle=False и
check_prediction; `combine` проверяет ключи выбранных heads; `pooled` и
`validate_arms` сохраняют полученный порядок в predictions_B.npz.
Следующий шаг — связать эту цепочку с уже принятым pinned baseline source
и pack/receipt доказательством, а не повторять inference или строить новый
контроллер. Проверка текущего исходника сама по себе не заменяет такой связи.

**Fixed-window T1.** Уже принятый
`results/smash_train_20261028/heldout_rn/per_file_counts.json` содержит
все 286 truth-bound validation content; primary_member совпал во всех 286
после нормализации суффикса `_prim.csv`. Его SHA
`2f2f7ac1342651631d105701012776c539145166c4ae6826ad3522a239785218`
совпал с `reports/status_20261103_support/input_pins.json`.
Записи содержат processed_span, primary_events и truth_hi_lo для T0/T1/T2.
Этот файл не входил в packet задачи 1.

Прежние report-fold и нынешние validation-fold различаются для этих content:
96 переходов 1→0, 96 переходов 0→2, 94 перехода 2→1. Поэтому нельзя перенести
старые reconstructed predictions/cuts как validation baseline по номеру fold.
Для reuse только primary truth ещё требуется сверить candidate processed
span и coverage на content; после совпадения возможен reuse fixed-window T1
без новой raw-primary reduction. Наличие записи не доказывает совпадение span.
На этой проверке новые T1/Rₙ values не рассчитывались.

Это не устраняет потребность в uncensored primary reduction для произвольных
новых окон: fixed-window counts не позволяют восстановить arbitrary bands.

## Что увеличило контекст

Входные snapshots задач занимали 459141, 529436 и 617910 bytes.
Task-one evidence JSON занимал 401453 bytes и повторно передавался задаче 2;
задача 4 получила полный primary_manifest.json размером 472447 bytes.
Prompt и task_packet стали компактнее, но сами входы остались крупными.
Native inputTokens — суммы по последовательным model requests, включая
повторно передаваемую историю и cache reads; их нельзя трактовать как размер
одного prompt или переводить в цену без native credit telemetry.

Пользователь прямо указал: «copilot пока не используем»; это зафиксировано
в списке задач. Дальнейшие действия
Codex: использовать существующие доказательства; проверить processed span
и producer order на pinned источниках; затем подготовить минимальный Rₙ diff.
Новый window scan, retraining, F scoring и платная повторная критика этой
приёмкой не запускаются. Для будущего адресного ревью достаточно выбранных
строк/записей и конкретного diff вместо повторной передачи больших manifests.


## Продолжение в Codex

После этой приёмки прямое поручение пользователя разрешило продолжить анализ и обновить статью. Все 286 member/span/Row/T0–T2 проверки пройдены, baseline producer связан с pinned source. [Fixed-window Rₙ](rn_validation_20261104/README.md) завершён, Appendix C прошёл четырёх критиков и number-auditor. Copilot не использовался; новых моделей, F scoring или GPU-задач не было. Предыдущие формулировки о недостающем доказательстве относятся к состоянию исходного Copilot packet.
