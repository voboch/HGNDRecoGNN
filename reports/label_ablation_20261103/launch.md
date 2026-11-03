# CPU submission: target patches всех зарегистрированных ролей

03.11.2026. Следующий запуск выпускает только no-floor patches для 578
content groups accepted split 848ebf49/prepare 4465983. Native labels и
float32 targets воспроизводятся точно; остальные tensors сохраняются.
Нет обучения, inference, F-метрик, выбора промежуточного floor или нового
scaler. Stage A 4481705/4481662 не повторяется.

Источники и SHA: `launch_plan.json`; исходные baseline receipts сохранены
для последующего frozen-cut сравнения. Human waiver прочитан из прежней
`rocky_authorization.json`; дополнительных tests/test-queue jobs нет.

## CPU ресурсы и конечность

Один job: rocky/normal, type_d, 4 CPU, 0 GPU, mem=0, 45 минут,
два I/O workers, no-requeue, прежний exclude cn-007/cn-008/cn-034.
Это caps прежнего полного census: 286 групп заняли 5:04 вместе с I/O.
Для 578 ожидается примерно вдвое больший объём; 45 минут сохраняет запас
на чтение accepted archive/raw. При сбое job не повторяется скрытно.

Все observations строятся прежним path-seeded noise reader и сравниваются
по accepted SHA. PackStore проверяет manifest/archive/member SHA. На каждый
content извлекается один native B pack из его прежней train/validation
роли; остальные fold/arm pack SHA записываются в patch manifest. При каждом
применении к своему fold pack заново проверяются keys, native targets и
байты всех прочих tensors. Accepted packs/source не перезаписываются.

Node-local `mktemp` work сохраняется; единственный compact archive export
публикуется в новом campaign `prepare_JOBID`, receipt - в новом home
receipt directory. Hash/state проверяется до использования результата.
Доступ к scratch обрабатывает прежний bounded access helper; нового retry
controller нет. Истинный сбой сохраняется, новая подача требует разбора.

## Source и review

Перед подачей создаётся отдельный immutable source из commit этого diff,
включая новые wrappers, plan, baseline receipt pins и старые CODE_EXTRAS
`slurm/smash_train{,_prepare}.hpc.sbatch` на уровне package root.
Последние нужны фактическому `smash_train.source_identity`; в package A
их не было, поскольку тот CPU runner эту функцию не вызывал.
Source sums/archive SHA, remote paths и verdict конкретного submission
будут записаны в `source_package.json` и `submission_review.md`.

GPU wrapper заморожен здесь для общего immutable source CPU/GPU. Это
не GPU submission GO: сначала нужны принятые patches и их exact SHA.
Его proposal - type_a V100, 1 GPU/4 CPU, 2:45 на segment, 15+15 total30,
max2 tasks. Вторая array зависит afterok от **всей** первой, чтобы две
independent %2 arrays не дали четыре одновременные задачи. Baseline не
повторяется, validation-only export не выдаётся за F performance.

Сегмент с 15 epochs имеет `segment.json` и execution `summary.json` с
`scientific_seed_complete=false`; полного `complete.json` там нет.
Состояние complete у execution summary означает завершение планового
сегмента/export, а не завершённый 30-epoch seed.
