# Подготовка проверки label floor и энергетических полос

29.10.2026. Протокол: [energy_label_optimization_protocol_20261029.md](../energy_label_optimization_protocol_20261029.md).
Проверка критиков: [preflight_review.md](preflight_review.md).

Перед эмпирическим census добавлена
[оценка из первых принципов](../energy_label_first_principles_20261029.md):
кинематика, роль time window, prompt-полоса и критерий информации отношения.
Три аналитических рисунка и `first_principles_numbers.json` получены без
чтения MC events/predictions. Порог0,3ГэВ как требование разметки из них не следует.

Воспроизведение аналитической части без данных:

```bash
python reports/energy_label_optimization_20261029/first_principles.py
python -m pytest tests/test_neutron_tof_bounds.py -q -p no:cacheprovider
```

Нужны NumPy/Matplotlib и соседний `HGNDPaper/scripts/plot_style.py` для
общего стиля. Числа рассчитаны с физической массой нейтрона PDG; production
`_dfToF` и его параметры не изменены.

**Готово:** протокол, конфигурация, инвентарь S286 и исполняемый кластерный
census разметки. Научные входы census ещё не связаны; никаких заданий,
переобучения, нового inference и эмпирического скана не запускали.

Порог разметки проверяется на сетке `disabled, 0.1, 0.2, 0.3, 0.4, 0.5 GeV`.
`disabled` снимает energy clause; временное условие ±0.3 ns остаётся прежним.
0.3 GeV сохраняется контрольным историческим значением, его целесообразность
не предполагается заранее. Код подготовки не меняет production labels.

`input_requirements.json` получен только из split и prepare **метаданных**:
286 truth-bound selection groups, 213 с записью audit-reference; исходный
prepare не экспортировал observations. Это не runnable input manifest.
Большие tensors/raw не скопированы на локальный диск. В этом инвентаре есть
SHA observation/native pack для каждого файла, его validation-fold и raw pins.

## Вход следующей стадии A

Нужен JSON схемы `label-energy-audit-inputs-v1`:

```json
{
  "schema": "label-energy-audit-inputs-v1",
  "config_sha256": "SHA256 фактического config.json",
  "split": {"path": "путь к исходному split_manifest.json", "sha256": "его SHA256"},
  "prepare": {"path": "путь к prepare/summary.json", "sha256": "его SHA256"},
  "files": {
    "tag/run/member": {
      "observation": {"path": "путь к сохранённому observation.pt", "sha256": "его SHA256"},
      "pack": {"path": "путь к исходному B validation pack.pt", "sha256": "его SHA256"}
    }
  }
}
```

Записи `files` должны покрывать **все** S286 и только их. `prepare` означает
внутренний `prepare/summary.json` схемы `smash-train-prepare-v1`, а не внешний
bundle `manifest.json`. Относительные пути разрешаются от input manifest.
Рецепт графа и роль validation проверяются по prepare receipt.
Получение недостающих observations — отдельный ограниченный CPU шаг на
тех же исходных байтах и с тем же noise key; текущее приложение его не выполняет.

Проверка связанного manifest без загрузки tensor:

```bash
python scripts/label_energy_audit.py --inputs /absolute/path/inputs.json --check-only
```

После same-path smoke и проверки фактической стоимости полный census:

```bash
python scripts/label_energy_audit.py --inputs /absolute/path/inputs.json --output /new/output/directory
```

Runner отказывается менять существующий output. Он сохраняет integer ledger,
гистограммы с отдельным overflow и нецензурированным мягким диапазоном,
native-replay guards, matching по Row/Id и PDF доли positives от threshold
с парной bootstrap SD. Matching здесь не устанавливает genealogy первичного
нейтрона. Выход не является результатом переобучения или оптимизации R_n.

## Проверка подготовки

```bash
ruff check analysis/label_energy_scan.py scripts/label_energy_audit.py tests/test_label_energy_scan.py
python -m compileall -q analysis/label_energy_scan.py scripts/label_energy_audit.py tests/test_label_energy_scan.py
python -m pytest tests/test_label_energy_scan.py tests/test_energy_bands.py -q -p no:cacheprovider
```

25 тестов прошли у основного агента и независимо у технического критика.
Проверены native replay, strict/disabled/zero semantics, неизменные targets,
границы 0/0.3/3 GeV, conservation histogram+overflow, точный processed span,
zero-event inventory, отказы при повреждённых pins/roles/recipes, и полный
synthetic путь из девяти toy-файлов до summary/PDF с shared bootstrap.
Синтетические результаты находятся во временной директории тестов и не
выдаются за измерения SMASH. Полный исторический набор тестов репозитория
не запускался: production paths не менялись, проверены новые механизмы и
используемый интерфейс энергетических полос.

Следующие B/C имеют план и ограничения, но ещё не имеют готового training
diff, ancestry ledger, exact-primary reduction и утверждённого window selector.
Автоматически перейти от этого census к GPU-кампании runner не может.
