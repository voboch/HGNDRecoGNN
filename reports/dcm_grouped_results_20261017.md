# DCM: итог зарегистрированного source-run сравнения

Все шесть обучений `4412700` и шесть test evaluations `4412701` завершены,
exit 0. Test открыт только после завершения всех обучений. Изменений recipes
по test outcomes и повторных оценок выбранных весов не было.

Canonical result: `results/dcm_grouped_20261017/summary.json`; генератор
`scripts/summarize_dcm_grouped.py`. Он проверяет Slurm receipts, 30 эпох,
hashes selected checkpoints, split manifest, counts и frozen thresholds.
Immutable execution snapshot и протокол сохранены без изменений.

Таблица показывает test cluster ROC AUC, безразмерную величину (higher is better),
scored on held-out source run 988976: 222251 clusters, в том числе
36713 MC-labelled signal clusters, из 131173 arm graphs / 91081 physical
events. Scalers fitted on training hits source run 988975; checkpoints and
thresholds selected on validation source run 988974. DCM production —
`ndet_links_Jan_26`, окно 1.0 ns; это не SMASH v2/v2fix. Обе arms имеют
исходную архитектуру H512/L8, 30 эпох без warmup, одинаковые train/validation,
loss weights, optimiser и cached graphs/noise. Каждый row — один training seed,
point estimate без отдельной sampling uncertainty; разность сравнивает разные
weights, paired by shared training seed (case 1).
AUC is dimensionless; higher is better.

| training seed | MultiStepLR test AUC (dimensionless) | cosine test AUC (dimensionless) | cosine minus MultiStepLR AUC (dimensionless) |
|---|---|---|---|
| 42 | 0.889530 | 0.854847 | −0.034683 |
| 123 | 0.885981 | 0.900769 | +0.014788 |
| 456 | 0.876356 | 0.905056 | +0.028700 |

Средняя paired difference +0.0029351, SE среднего 0.0192332. SE равен sample
SD трёх разностей / sqrt(3), а не cluster bootstrap или evaluation repeat
floor. t=0.1526 на 2 df; двустороннее критическое значение 4.303 при 0.05.
95% interval [−0.0798251, +0.0856954]. Преимущество cosine не разрешено;
эквивалентность тоже не установлена. Предварительное ожидание +0.004 AUC не
получило разрешённого подтверждения, протокол не изменён.

Пороги — первая из 401 точек [0.001,0.999], достигающая count purity 0.70
на validation. Это inherited comparison convention, не optimum для EoS:
более строгая purity обычно снижает signal efficiency. Все шесть validation
scans сохранены; generated figure показывает полный tradeoff. Test purity
0.6861–0.7089 и efficiency 0.4202–0.6052 оценены на frozen thresholds, без
подгонки к test labels. Точные thresholds, counts и energy metrics каждой
модели хранятся в canonical JSON и generated operating-point table статьи.

Training-seed SE условна на одном test source run; между независимыми MC
generations variance не измерена. Более низкий AUC, чем в historical
graph-random diagnostic controls, не изолирует эффект scaling или split.
Не установлены идентичность опубликованному event estimator, физический
unfolded neutron spectrum или neutron-track multiplicity response.

Рукопись обновлена этим результатом. Все зарегистрированные эксперименты
завершены; дальнейший sweep по открытому test не запускается.
