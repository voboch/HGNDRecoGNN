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

## Дополнение: принятый odd и фактически собранный пакет

Проверен `HGNDPaper/output/submission/nst_20261116_architectures/hgnd_nst_architecture_replay` после полного staging. Это дополнение заменяет прежний статус отсутствия odd и архива в T1. S4-уточнение о Student t, df=2 и семье двух сравнений внесено в рабочий main.tex после staging; окончательная привязка обновлённого PDF ещё ожидается и здесь не объявляется выполненной.

- Все 86 записей `SHA256SUMS` совпали с файлами; лишних файлов нет. Все 23 canonical pins и все 20 записей `NUMERICAL_OUTPUTS.sha256` совпали.
- Прямое сравнение upstream и released architecture NPZ для обоих scopes: наборы полей одинаковы, все массивы кроме `content` поэлементно совпали; `content` точно заменён последовательностями `even_000…` / `odd_000…`. Порядок settings и все 400 draws совпали также с соответствующим историческим benchmark ledger. Таким образом, общие файлы и bootstrap pairing сохранены фактически, а не только заявлены.
- Просмотрены поля всех шести released NPZ. Исходных detector hits, checkpoint arrays, event/track IDs в этих полях нет. Поиск `/Users/`, `/scratch/`, `/home/`, `vovvy`, `@hpc`, `BEGIN PRIVATE KEY` во всех текстовых JSON/Python/Markdown/TeX/TXT пакета совпадений не дал. Это ограниченная проверка перечисленных признаков, не универсальная гарантия анонимности.
- T2 закрыт явным erratum в PROVENANCE с сохранением принятых even summary bytes. T3 закрыт точным scope: pooled even score-only matching, true 0.3–4 ГэВ. Accepted-summary ограничения для AUC/AP, native-positive MAE и t/Holm перечислены явно.
- Прочитаны odd acceptance и полный replay log. Odd принят на 282 файлах / 1 165 216 событиях, с теми же 27 моделями (18 новых, 9 прежних). Лог staging подтверждает benchmark, scan, odd, capped и architecture replay и построение из 23 inputs. В этом дополнении повторный научный replay не запускался; выполнены проверки сохранённых байтов и метаданных.

Точные SHA-256 дополнения:

- `odd_acceptance_20261116.json`: `c0281f3fa349f6d4306a0f09bb6a0dc96d53c8a8fdac3e6c8905031e7570e864`.
- `score_odd_acceptance_20261116.json`: `1513c7b39e4a969926c48444703b1fa1324032b74e6bef1456f99c9a60dd94b0`.
- `odd_result/summary.json`: `8e13095ea37b0074650d8bf553b410fa72c19a99e14aec909cad52ba58f35a77`.
- Upstream odd ledger: `632e168f12f91829b8750a22aebc099ebd5e80280be5fd002df07e548e647ca3`.
- Released even ledger: `f028441d6f42a1f09e72655ef2d7097b296e46c8e21f37802db23cdeebc5616f`.
- Released odd ledger: `593715d0bcb78e000b3411f6d363ab983a2f5ead5e0ef52514d2b2d800444f53`.
- Frozen policy остаётся `8587f2bbdc750adc619daa46d4c707694c6416e652ef0e0dd372e81f56570f69`.
- `full_independent_replay_20261116.log`: `b93abf01bcbc69435df7934e143d9fcfc0483376702c76a563916e0ff40a1617`.
- Пакет до S4 PDF refresh: `PROVENANCE.json` — `379c35a22b4b02d0cdc8404c4a056642ba90bc4cf3ea6e97b127015fdbbfdf54`; `SHA256SUMS` — `ae34fe8a99ed92fc14e9a886d7ba8331d7d192dd71e974363b7ab35f0db92ff9`.
- `NUMERICAL_OUTPUTS.sha256`: `0b7da8e2d8e9c9d87ae9e7758c2db32918acc24984c4542726a914c5e8db8766`.
- `stage_nst_replay.py`: `7e7cf95d5c97c8e93dead8d925a00a375ea5c1c798170f7308dec586fb10c12b`.
- `architecture_results.py`: `b6630ef4934561b7302c4212643ab438a9312379a6d905055a435a5f15f1e93b`.
- Независимый verifier остаётся `a4fefcd345e30f64563b566e0322660f9056beac2e591512b72ab168f5c5661f`.

**Вердикт дополнения:** GO для численного состава локального release candidate. T1 остаётся открытым только для окончательной привязки source/PDF после S4; публичный выпуск отдельно требует разрешений владельцев (B1). При доказанном неизменном составе численных inputs/scripts/outputs повторять scientific replay для одной текстовой поправки не требуется.

### Финальная привязка после S4: проверка 87 файлов

Все 87 записей итогового SHA256SUMS совпали; лишних файлов нет. Main и PDF совпали одновременно с PROVENANCE и BUILD_VALIDATION:

- `manuscript/main.tex`: `ab87c6fc9d7c6baec403340d0a0052845d40f6955714762d01a071e284323b07`.
- `reference.pdf`: `31a65f30f6adcb7f297c0070f590adf390db011b2a3ed48ab4d9e33403ac2e2a`.
- `BUILD_VALIDATION.json`: `0ca688c22bbdec46a740952cd708903a9d1d16ba8aaf075b465942aef80b82b7`.
- `PROVENANCE.json`: `4150c01a2c08ac42e1b42f55614d38ce4f3e470ebe568cc9ad53f06f6d77619c`.
- `SHA256SUMS`: `8d6f1e6d648999dd9d63f922df2871974ab8cf6bd37c867b3dfb7fb8116c4b67`.

Numerical output manifest остался `0b7da8e2d8e9c9d87ae9e7758c2db32918acc24984c4542726a914c5e8db8766`; ссылка на staging replay log совпала с фактическим `/tmp/hgnd-nst-architecture-stage.log`. Визуальные и font checks здесь приняты по записи сборщика; критик повторно PDF не рендерил.

Обнаружено одно уточнение [S] к BUILD_VALIDATION: поле `compiled_asset_sha256` перечисляет 50 файлов дерева сборки, включая 18 figure PDFs, отсутствующих в пакете и не вызываемых шестью `includegraphics` текущего main.tex. Все 50 хешей совпали с рабочим деревом; это инвентаризация дерева, а не доказательство 50 фактически прочитанных компилятором assets. Все шесть вызываемых рисунков и все явно подключённые таблицы присутствуют. Исправление: назвать список инвентаризацией дерева либо сузить его по фактическим входам; не добавлять неиспользуемые рисунки ради названия поля. Запрос передан сборщику.

**Итог технического ревью: локальный пакет — GO; T1 закрыт.** Источники, PDF и численный пакет связаны проверенными хешами. Уточнение названия inventory выше не меняет байты результата и не блокирует локальную приёмку. **Публичный выпуск остаётся за B1 — разрешениями владельцев.**

### Закрытие последнего замечания к metadata

[S] закрыто: BUILD_VALIDATION теперь разделяет `source_asset_inventory_sha256` (50 файлов дерева, включая неиспользуемые) и `included_figure_sha256` (шесть рисунков из main.tex). Все шесть файлов присутствуют и совпали по хешам; обновлённый manifest проверен: 87/87. SHA-256 BUILD_VALIDATION: `38a41adabe9d3654e78b7c49b1e6eb4498617a140ce641e7818bf33d4cfe23b3`; SHA256SUMS: `fece5c957e88aebedb05ec6f98bd52dd524695fc72611877051851f39e9babf1`. Предыдущие main/PDF/numerical hashes сохраняются. **Локальный GO подтверждён; открытых технических замечаний в проверенной области нет. Публичный B1 остаётся открытым.**
