### T1 — Текущий архив и его PDF ещё не приняты

- Severity: [B] только для статуса «готово к подаче».
- Где: `HGNDPaper/scripts/stage_nst_replay.py:95,132,141`; `even_acceptance_20261116.json` (`odd_state`).
- Проверено: even принят на 286 файлах / 1 180 595 событиях; 27 fit identities включают 18 новых и 9 прежних GraphSAGE. Полного odd replay, созданного текущего архива и его PDF в этом ревью нет. Staging проверяет наличие PDF и сохраняет его хеш, но не доказывает соответствие PDF исходнику.
- Минимальное закрытие: дождаться принятого odd-экспорта, выполнить существующую команду staging с четырьмя явно перечисленными `--reviewed-input`, проверить итоговый архив и PDF выбранной версии. Новые unit tests и тестовые jobs не нужны.
- Трудоёмкость: существующий финальный проход после odd-приёмки.

### T2 — Поле selection неоднозначно описывает сохранённую энергетическую выборку

- Severity: [S].
- Где: `HGNDRecoGNN/scripts/smash_architecture_reduce.py:541,582,823–833`; `even_result/summary.json::energy_replay_sample.selection`.
- Проверено: сохранены 165 748 общих изолированных связей до отбора; массив `isolated_range_keep` имеет форму (3, 3, 165748), в нём 372 637 значений false. Поле `selection` сообщает `raw70 classifier cut AND 0.3 < raw reconstructed energy < 4 GeV`, хотя это определение маски, а не всей сохранённой выборки. Код preselection корректно не применяет маску.
- Минимальное исправление: пояснить в README, что `selection` определяет только `isolated_range_keep`; вся энергетическая выборка сохранена до score/energy отбора. Не менять принятые численные массивы и цепочку хешей ради формулировки.
- Трудоёмкость: несколько минут.

### T3 — Заявление о replay matching следует ограничить реально проверяемым срезом

- Severity: [S].
- Где: `HGNDPaper/scripts/verify_nst_architecture.py:90–108`; `stage_nst_replay.py:213`.
- Проверено: independent verifier проверяет pooled score-only matching в истинной полосе 0.3–4 ГэВ; остальные stages, truth ranges и отдельные настройки остаются в accepted summary. Именно проверяемый срез использует `architecture_results.py:159–189`. Все preselection energy bins проверяются отдельно.
- Минимальное исправление: назвать этот срез в README/описании replay. Не расширять вычисления только ради более общего описания.
- Трудоёмкость: несколько минут.

### Проверенная область и версии

Дата: 16 ноября 2026. Аудит для автора и сборщика NST: корректность derived-data replay, сохранность чисел и границы заявления о воспроизводимости. Прочитаны AGENTS.md, `.claude/paper-review.md`, critic-technical и shared conventions §10, §20, §25, §30 (редакция 12 ноября). Отдельный no-ai-slop SKILL.md по указанному внешнему проекту локально не найден; применён встроенный checklist AGENTS.md. История последних 20 коммитов и рабочие изменения обоих репозиториев просмотрены. HEAD HGNDRecoGNN: `bd17f185e25e6620e1aa7bf13b9ac345f20562b3`, ветка `smash-train-20261028`; HGNDPaper: `ce39645163a5ed6db167f5e1d2f622bad0472eaa`, ветка `main`. Проверялись рабочие файлы, а не только HEAD.

SHA-256 проверенных исходников:

- `HGNDPaper/scripts/stage_nst_replay.py`: `41670fcc6c91c9b32c255fcf5c90c0d4fc6823320e92d8e14b7c61ba2616628c`.
- `HGNDPaper/scripts/verify_nst_architecture.py`: `a4fefcd345e30f64563b566e0322660f9056beac2e591512b72ab168f5c5661f`.
- `HGNDPaper/scripts/architecture_results.py`: `2d94d992e76ef949c95d3c00a43cd794783aa12648abece9e0974ce8bde7d347`.
- `HGNDRecoGNN/scripts/smash_architecture_reduce.py`: `58d93c67a0d9378747ef523eaa425404ddc4efa9244819fb83ac491d9f8ac547`.

SHA-256 проверенных свидетельств:

- `even_result/summary.json`: `cfa8b34a7d4a9da7f5cad0bb20b4070dabb52c8d2823e0da67a69ac68a4418bf`.
- `even_result/file_counts.npz`: `0781663d4b830b2da2cae3dd2edcaf28d140635e375914ee289bed4b796ee53a`.
- `even_result/frozen_policy.json`: `8587f2bbdc750adc619daa46d4c707694c6416e652ef0e0dd372e81f56570f69`.
- `reference_ceiling.json`: `2eb8bdfa344b5fc447ec255af3043971c0dee47e6f143b81366046fa43fe42c7`.
- `even_acceptance_20261116.json`: `0ee10e662778ad9e010c4e8e7f29802c338aaa639da6772e188920dcd281db6d`.
- `score_even_acceptance_20261116.json`: `7d27c619c9f5fd8b145a9d301d41563257af7936616b6a4a01365d29ea533476`.
- `fit_acceptance_20261116.json`: `cb47b3a3ea1f5a507359f37145c69ec7acadec665de56f2526a7544c72adbd31`.
- `parameter_inventory_20261116.json`: `04e1ffe9a3b28e9fecf9e75cdb296a6329ac6a283a182a041732bd347e066ac1`.
- `methodology_review_20261116.md`: `8be289437e8e05561c590f16ae11ab7ab3f5f60b5a103601672315961802f3c2`.

Summary и acceptance совпадают по текущим хешам reducer, ledger, policy и verifier. Score acceptance ссылается на существующий fit acceptance с совпадающим хешем. Сохранённый `even_independent_replay_20261116.log` сообщает replay потолка, замороженных целей, coverage/misses/splits/merges и всех preselection energy bins. В этом аудите полный численный replay повторно не запускался; локально прочитаны JSON, NPZ shapes/dtypes и исходники. Jobs, inference и unit tests не выполнялись.

Staging заменяет только строковый `content` и повторно сохраняет остальные массивы без численного преобразования (`stage_nst_replay.py:185–200`). В inspected even ledger нет event/track IDs; есть необходимые file indices для общего bootstrap. В canonical even summary и parameter inventory приватных абсолютных путей не найдено. Это проверка текущих входов и кода, не аудит ещё не созданного релиза. Hash цепочка различает upstream NPZ и опубликованный NPZ; замороженная policy копируется без изменения байтов. Архитектурные файлы добавляются автоматически лишь после явного допуска всех четырёх canonical inputs.

Сохранённые draws общие для методов и seed, а preselection energy использует тот же индекс файла; средние берутся по seed внутри каждого draw. В verifier правильны sample SD (ddof=1), линейные weighted quantiles и парные разности. Provenance стратификации принят из upstream ledger; verifier не восстанавливает part strata. ROC-AUC/AP, native-positive MAE и seed t/Holm явно обозначены как accepted-summary inputs. First-hit результат и own-selected energy cohorts не становятся общим сравнением архитектур: текущая таблица использует общий preselection cohort, и это ограничение необходимо сохранить.

**Would I accept this code base for submission?** Пока нет: нет принятого текущего odd/archive/PDF комплекта. **GO для продолжения уже разрешённой сборки и публикационного использования принятого even-результата в указанной области; финальная приёмка условна по T1.** Блокирующих ошибок в просмотренной арифметике и цепочке even-хешей не обнаружено.
