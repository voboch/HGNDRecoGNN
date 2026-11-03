# Полный отчёт SMASH, 03.11.2026

Редактируемый текст: `../smash_performance_20261103.md`.
PDF: `../../output/pdf/hgnd_smash_performance_20261103.pdf`.
23 семейства рисунков сохранены в `figures/` как PDF и PNG.

Отчёт использует принятые baseline predictions 29.10, certified primary
spectra 4453896 и S286 census 4481705. Здесь нет новых inference, training,
fit или выбора рабочего порога. `input_pins.json` фиксирует 43 источника.

Для повторной сборки из сохранённых редукций нужны Python, numpy, pandas,
matplotlib, pypdf, reportlab и шрифт DejaVu Sans. `plot.py` использует общую
`scripts/plot_b_full.py:apply_style`. Запускать из корня HGNDRecoGNN:

```sh
python reports/smash_performance_20261103/plot.py
python reports/smash_performance_20261103/build_report.py
```

В текущем окружении научные пакеты доступны в conda env `pyg`, а reportlab
в bundled Python Codex; поэтому сборка PDF использовала второй интерпретатор.
Путь к DejaVu в `build_report.py` соответствует текущей установке matplotlib.

`counts.npz` и `data.json` содержат per-file histograms, score-grid counts,
матрицы миграции, точные cuts и SHA всех 5112 прочитанных CSV. Редукцию можно
повторить `collect.py`, если сохранены локальные accepted CSV и certified
primary файлы в `results/`; модели и GPU для неё не нужны. Native truth,
candidate keys, полные интегралы и per-fold operating counts проверяются
assertions. `primary_spectra.npz` воспроизводит сохранённые центральные
спектры и SD 400 whole-file replicas с точным краем 0,3 ГэВ.

Для переноса спектра в плотность делить counts/bin/event на точную ширину
bin. В `counts.npz` первый energy bin - underflow E<0, последний E>=6,5 ГэВ.
Native negatives с target=0 не означают физически нулевую энергию.

`validation.json` фиксирует итоговые SHA, проверки арифметики/геометрии/fonts
и факт просмотра страниц именно этой версии PDF. `review.md` содержит
критические замечания и их разрешение. Intermediate page PNG не коммитятся.
