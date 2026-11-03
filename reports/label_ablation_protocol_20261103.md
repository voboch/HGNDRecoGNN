# Один no-floor arm после принятого census

03.11.2026. Решение после S286 4481705; основание — автономное поручение
пользователя и заранее зарегистрированное сравнение disabled против 0,3 ГэВ.
Статус: конкретизация исполнения, **новых GPU-задач пока нет**.

## Научная переменная и ожидаемое направление

Меняется только энергетическое условие label/target: native Ekin > 0,3 ГэВ
заменяется отсутствием energy clause, остальные PDG/Side/ToF условия прежние.
В S это добавило 5480 positive-кластеров, включая 3500 direct-primary-only.
Сетка промежуточных floor не обучается; второй arm сейчас не выбирается.

Включение мягких объектов может улучшить их распознавание и регрессию,
но расширяет задачу также на nonprimary MCTrack. Численный выигрыш AUC,
hard-energy response и Rₙ заранее неизвестен; допускается отсутствие
улучшения. На S в нынешних Rₙ полосах без переобучения C1/C2/C4 не меняются.

## Данные и артефакты

Тот же split 848ebf49, accepted prepare 4465983, графы 1 нс/r3,6,
наблюдения/jitter, признаки, train-fit scalers, folds и seeds 42/123/456.
Обучение — с нуля, по 30 суммарных эпох; девять принятых baseline не
повторяются. Ни architecture, loss weights, optimiser, scheduler, batch size,
ни per-head selector в этой абляции не меняются.

CPU preparation должна выпустить только per-content patch label/energy
и SHA manifest. Accepted packs читаются с исходным SHA; patch накладывается
после проверки, исходные packs не переписываются. Native reference0,3
должен совпасть точно, disabled — superset, targets общих positives прежние.
`x`, `edges`, PE, cluster membership, indices, Row/Side и pointers сохраняются
побитово. Один patch на порядок кластеров content используется с его
исходными fold-scaled packs; изменение scalers не требуется.

S286 недостаточно для train preparation. Прежние train/validation роли
включают также нечётные и unbound группы. Их targets преобразуются в рамках
тех же зарегистрированных ролей без просмотра F-метрик, выбора floor или
window по F. Unbound detector-labelled группы не исключаются из train/val
из-за отсутствия первичного binding. F outcomes до фиксирования анализа
не используются для настройки; свежего holdout всё равно нет.

## Обучение и общее оценивание

Тонкий runner использует `smash_train.train_run`, `recipe` и
`NativePackDataset`. `run_task`/`validate_arms` целиком не переиспользуются:
они требуют одинаковых A/B targets и заново настраивают cut/calibration
всех arms, что не соответствует чистому label-опыту.

После выбора e_cls/e_en сохраняются immutable candidate keys, raw scores /
energy и обе reference-label системы. Основная cross-label таблица сравнивает
старый и новый arm против одинаковых reference0,3 и disabled. Native-task
AUC остаётся диагностикой. В энергии — raw bias/MAE на common-positive
intersection и отдельно на добавленных мягких объектах. Нулевые positive
targets, если встретятся вне S, учитываются явно: старый helper `positives`
их отбрасывает и не должен незаметно задавать новую популяцию.

Первый operating point использует frozen cut принятого SMASH baseline
того же fold/seed; primary raw-energy сравнение не переподгоняет soft
calibration. Новая validation-only cut-политика может быть только отдельным
помеченным результатом на общей reference задаче. Разные arms соединяются
по candidate keys, не по равенству cl_label/e_true. Один fixed primary truth
используется во всех физических сравнениях.

## Исполнение и поправка бюджета

Read-only проверка исходных complete receipts обнаружила, что прежняя оценка
«около 20 GPU·ч на arm» учитывала финальные сегменты resumed runs неполно.
Шесть непрерванных 30-epoch задач заняли 10650–11249 с pure fit:
355–375 с/эпоху, 2,96–3,12 GPU·ч на модель. Полный arm из девяти моделей
ожидается примерно 27–28 GPU·ч fit или 30–33 GPU·ч с validation/I/O.
Эта оценка заменяет прежние 20 GPU·ч; принятые метрики baseline не меняются.

Для риска отмены задач дольше 3 ч — одна заранее запланированная граница
после 15 эпох, затем continuation ещё15 до total 30. Это не повтор обучения:
coherent mirrored model/optimizer/scheduler/RNG/history переносятся в рамках
той же stable identity. Job/segment IDs — только execution metadata.
Сегмент 15 эпох ожидается около 1,5–1,6 GPU·ч pure fit; конкретный scheduler cap,
накладное время, source SHA, output paths и resume receipts фиксируются
до подачи после готового diff. Не более двух задач кампании одновременно.
Новые baseline runs, скрытые retries и расширение эпох не допускаются.

Пользователь явно разрешил обход test-очереди и отсутствие дополнительных
unit tests. Старый `train()` GPU-smoke guard не обходится поддельным
summary: новый runner вызывает train_run с явной записью human waiver,
проверками source/input/structural identity. Перед подачей необходим критик
конкретного diff и ресурсов. Подготовка не является разрешением удалить
data assertions или считать незавершённый сегмент завершённым seed.

## Состояние review

Физический критик: один no-floor arm оправдан, direct loss подтверждён,
исправление Rₙ не установлено. Статистический: общие reference labels и
paired errors обязательны, prevalence не ограничивает training effect.
Технический: рекомендован patch adapter и reuse train_run; обнаружены
опасные native truth-equality guards и повторная настройка baseline в
старом validate_arms. Это рекомендации по реализации, не GO на ещё
не написанный GPU runner. Следующее действие — реализовать adapters и
предъявить конкретный diff предзапускному критику.
