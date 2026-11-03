# GPU submission preflight: GO

03.11.2026. Read-only critic `/root/label_ablation_diff_preflight` дал GO
на две зарегистрированные GPU arrays после финальной проверки очереди и
возможностей Slurm. Конкретный plan - `gpu_submission_plan.json`, принятые
patches - `prepare_4482805_acceptance.json`. Это один no-floor arm, девять
моделей, по 30 суммарных epochs; baseline не повторяется.

Source262, wrapper SHA, manifest/archive/summary hashes patches проверены
независимо и совпадают с plan. CPU4482805 принят на всех578, S286 replay
совпадает; superset и common targets сохранены. Wrapper передаёт source,
input, patches, baseline/fold/seed и стабильный mirror без job IDs в identity.

Type_a V100,1GPU/4CPU,mem0,rocky/normal,2:45 на segment,0-8%2,no-requeue.
Вторая array должна иметь **afterok:<ID всей первой array>**, без task
suffix или aftercorr. Две arrays тогда не пересекаются; максимум2 задачи.
30-33GPU·ч - прежний прогноз с validation/I/O. Перед подачами проверить
отсутствие дубликатов; до первой - отсутствие старого mirror. Никакого
скрытого retry, второго arm или продления epochs.

Epoch15 marker не является complete seed; continuation требует coherent
next_epoch15 и совпадающие config/RNG/history, затем total30. Экспорт -
только validation нового arm с обеими reference labels и candidate keys,
frozen baseline cut и raw energy. Cut/calibration не фитятся; F не оценивается.
Human waiver явный, GPU-smoke PASS не заявляется.

Progress gate: patches приняты сегодня, model outcomes29.10; инфраструктурного
дрейфа нет. Reviewer не запускал SSH, tests, jobs, inference или training.
Остаточных блокирующих ошибок source/argv/resources не обнаружено. Реальные
Slurm receipts обеих подач и dependency будут сохранены отдельно.
