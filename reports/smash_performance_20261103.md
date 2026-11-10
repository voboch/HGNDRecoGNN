# HGND: качество реконструкции на SMASH

## Полный отчёт • 3 ноября 2026

Исправленный SMASH Xe+Cs, E_kin/A = 2,87 ГэВ; U_sym = 0/18/90 МэВ. На общей выборке SMASH схема, обученная на SMASH, показала выше полноту и меньшую ошибку энергии. Обучающие выборки различаются энергией пучка, мишенью и генератором. Схемы используют разные нормировки признаков. Здесь приведены исходные результаты при целевой чистоте 70%. [Пересчёт при пороге по контрасту Rₙ](rn_score_cut_20261110/fixed_results/README.md) отдельно показывает проверку при 90%, ширину контраста и изменение спектров.

Статус baseline: девять обучений по 30 эпох и три оценки приняты 29.10; census разметки на S286 (286 truth-bound групп чётных частей 0/2/4) принят 03.11. Ниже приведены baseline результаты, спектры, диагностика и фактический выбор порогов. Документ дополняет smash_train_results_20261029.md; её исходные результаты сохраняются.

Дополнение 04.11: [девять no-floor моделей завершены](label_ablation_results_20261104/README.md);
[fixed-window Rₙ проверен на selected validation](rn_validation_20261104/README.md).
Выигрыша Rₙ не установлено; сравнение использует raw B0 и прежний численный
cut, без новой calibration. Эти validation результаты следует читать отдельно
от baseline/F таблиц и PDF среза 03.11 ниже. Новых model calls для анализа нет.

| Arm | ROC-AUC ↑ | AP ↑ | Purity (%) | Eff. (%) ↑ | Bias (МэВ) →0 | MAE (МэВ) ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 0.8995 ± 0.0073 | 0.7001 ± 0.0089 | 69.87 ± 0.32 | 54.9 ± 1.0 | -49 ± 53 | 169 ± 15 |
| A1 | 0.8995 ± 0.0073 | 0.7001 ± 0.0089 | 70.109 ± 0.046 | 54.6 ± 1.5 | 20.1 ± 3.1 | 167 ± 13 |
| B0 | 0.954684 ± 0.000018 | 0.84701 ± 0.00011 | 70.227 ± 0.055 | 80.447 ± 0.060 | -8.4 ± 3.1 | 81.83 ± 0.35 |
| B1 | 0.954684 ± 0.000018 | 0.84701 ± 0.00011 | 70.227 ± 0.055 | 80.447 ± 0.060 | 7.76 ± 0.82 | 84.75 ± 0.22 |
| Bcomp | 0.954682 ± 0.000024 | 0.84702 ± 0.00012 | 70.239 ± 0.025 | 80.439 ± 0.028 | 7.77 ± 0.89 | 84.67 ± 0.21 |

В таблице mean ± SD трёх training seeds; каждый seed сначала усреднён равными весами по трём fold-метрикам. Это разброс инициализации, не полная file-sampling ошибка. Оценка: 568 truth-bound test-файлов, 3 572 710 кандидатов, 640 277 MC-positive. Energy bias/MAE - на всех этих positives, target - максимум начальной энергии подходящего MCTrack в кластере.

Парное сравнение B1−A1 по трём общим training seeds (разные веса и scaler): ΔAUC=0.0552 ± 0.0042, t=13,09; Δefficiency=25.89 ± 0.81 п.п., t=32,01; ΔMAE=-81.8 ± 7.4 МэВ, t=−11,04. Здесь ±SE среднего парных разностей, df=2, двусторонний t_crit=4,303; эти три изменения разрешены по registered seed-критерию. Это не ошибка между независимыми productions.

B0 имеет MAE 81,8 МэВ, B1 - 84,7 МэВ: калибровка улучшает медианный response, но не минимизирует MAE. У B1 reconstructed 90/0 контраст Rₙ = +1,82…+2,01% по seed при T1 MC truth +4,72 ± 0,57%; парный дефицит около 2,7-2,9 п.п. сохраняется.

## 1. Датасеты, обучение и допустимое сравнение

| Arm | Веса и scaler | Cut и energy postprocessing |
| --- | --- | --- |
| A0 | Frozen DCM GraphSAGE, три seed; frozen DCM scaler | DCM-validation cut, raw energy |
| A1 | Те же DCM веса и scaler | Cut и calibration на SMASH fold-validation |
| B0 | SMASH fold-train, один train-fit scaler на сторону | SMASH-validation cut; raw energy |
| B1 | Те же B0 веса; scores e_cls, energy e_en | Тот же B cut; validation calibration |
| Bcomp | SMASH, один composite-selected checkpoint | Свой validation cut и calibration |

B1−A1 сравнивает адаптированные pipelines: отличаются веса и scaler. Это не чистая абляция только обучающей выборки или только весов. Observations, графы, candidates и MC labels общие. Scalers B фитятся только на train-файлах fold, в смеси всех трёх потенциалов; U_sym и имя файла не являются признаками. Cut и calibration не подбираются отдельно для потенциалов.

Каждый fold: student_sage H128/L4, dropout 0,1, batch 256, Adam lr 10⁻³, weight decay 10⁻⁵, 30 эпох, decays в эпохах 10/15/25 с gamma 0,1. Loss: BCE + 0,5 SmoothL1 на positives. Это унаследованный рецепт, не результат оптимизации на данном физическом отклике. Seeds 42/123/456; гиперпараметры в этой кампании не сканировались.

| Популяция | 0 МэВ | 18 МэВ | 90 МэВ | Значение |
| --- | --- | --- | --- | --- |
| Full primary events | 804369 | 824338 | 717286 | 2 345 993, включая пустые |
| Processed-span events T1 | 804300 | 824270 | 717241 | 2 345 811, Row<processed_span |
| Events with candidates T2 | 594190 | 610225 | 531709 | 1 736 124, есть observed candidates |
| Truth-bound test files | 195 | 200 | 173 | 568 групп |

Full-primary спектры, T1 и candidate-level response имеют разные знаменатели. Их нельзя делить друг на друга и называть detector efficiency. Для протонов прямолинейное окно не включает магнитный транспорт и tracking efficiency.

## 2. Разбиение, повторения и нормировка

![Рис. 1. Схема accepted cross-fitting. Группы построены по content/event/direct-primary связям; random cluster split не применяется. Это схема ролей, не измерение; точные counts приведены ниже.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/20_crossfit.png)

Рис. 1. Схема accepted cross-fitting. Группы построены по content/event/direct-primary связям; random cluster split не применяется. Это схема ролей, не измерение; точные counts приведены ниже.

| Fold | Train: 0/18/90 | Validation: 0/18/90 | Test: 0/18/90 | Unbound в test parts |
| --- | --- | --- | --- | --- |
| 0 | 96 / 99 / 90 | 33 / 33 / 31 | 66 / 68 / 58 | 4 |
| 1 | 98 / 101 / 92 | 32 / 33 / 30 | 65 / 66 / 58 | 3 |
| 2 | 98 / 100 / 92 | 33 / 34 / 31 | 64 / 66 / 57 | 3 |

578 content groups, 568 truth-bound. Train/validation включают unbound detector-labelled группы:5/1,6/1,5/2 по folds. Они не превращаются в negatives; в физический test входят только bound группы. Внутри fold train/validation/test не пересекаются; train-наборы разных folds пересекаются.

Каждый test-файл оценён моделями, которые не использовали его для своего train или validation. Но frozen transfer уже открывал эти данные: это retrospective cross-fitting, не свежий blind holdout. DCM student≈0,98 относится к иной random-split задаче с повторяющимися inputs; для этой pipeline не является независимым целевым benchmark.

## 3. Все пороги: определения и мотивация

| Условие | Фактический смысл / основание | Статус |
| --- | --- | --- |
| Edep >3 МэВ | Отбор наблюдаемых hits; не равен энергии падающего нейтрона | Унаследованный observation cut |
| Jitter σ=0,15 нс | Deterministic path-seeded Gaussian на hits | Один зафиксированный шумовой view |
| Label Ekin >0,3 ГэВ | Строгий floor начальной энергии подходящего MCTrack | Историческое определение; теперь проверяется no-floor |
| ToF label ±0,3 нс | Prompt-согласие Ekin с ToF; условно 2σ однохитового jitter при идеальном start | Унаследовано, не оптимизировано |
| Graph radius 3,6 | Расстояние в индексах Layer/Row/Column, не подтверждённые 3,6×4 см | Унаследовано |
| Graph Δt1 нс; ≤32 соседей | Пересечение spatial/temporal radius_graph внутри event/side; компоненты могут образовывать цепочки | Не DAQ window и не предел времени всей компоненты |
| Count purity target 0,70 | N(selected native-positive)/N(selected) | Унаследованный компромисс, не optimum Rₙ |
| Score cut | Первый из 401 cuts, достигший target на fold-validation | Подобран; затем frozen на test |
| Rₙ low [0,3;1), high ≥2 | Определение наблюдаемой, отдельное от MC-label floor | 1/2 ГэВ унаследованы; не оптимизировались |
| 35 нс | Опубликованный операционный пример; в observe/transfer_pack такого cut нет | Не считать применённым |

Графовые параметры не выводят автоматически физический размер shower или neutron-track response. Требование ToF относится к MC-разметке и использует truth, а не к самостоятельному измерению. MC-label требует Ekin>0,3 ГэВ, а нижняя полоса Rₙ включает E≥0,3 ГэВ: это также разные граничные конвенции.

## 4. Как выбраны score cut, checkpoints и calibration

Фактический путь: smash_train.choose_cut → reco_performance.threshold_scan. На pooled-potential validation каждого fold/seed перебираются 401 равноотстоящий threshold от 0,001 до 0,999; selection строго score>threshold. Выбирается первый, то есть минимальный grid cut с count purity≥0,70. Если такого cut нет, run не evaluable. Это не F1/Youden/AUC optimum и не подбор по test или Rₙ.

| Fold | Seed | e_cls | e_en | e_comp | A0 cut | A1 cut | B0=B1 cut | Bcomp cut |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 42 | 28 | 21 | 27 | 0.465070 | 0.475050 | 0.302895 | 0.302895 |
| 0 | 123 | 29 | 22 | 29 | 0.357785 | 0.352795 | 0.300400 | 0.300400 |
| 0 | 456 | 24 | 21 | 24 | 0.397705 | 0.415170 | 0.300400 | 0.300400 |
| 1 | 42 | 26 | 23 | 27 | 0.465070 | 0.477545 | 0.302895 | 0.302895 |
| 1 | 123 | 28 | 22 | 28 | 0.357785 | 0.357785 | 0.302895 | 0.302895 |
| 1 | 456 | 23 | 25 | 25 | 0.397705 | 0.420160 | 0.302895 | 0.302895 |
| 2 | 42 | 29 | 22 | 29 | 0.465070 | 0.470060 | 0.300400 | 0.300400 |
| 2 | 123 | 21 | 28 | 28 | 0.357785 | 0.352795 | 0.295410 | 0.295410 |
| 2 | 456 | 23 | 26 | 26 | 0.397705 | 0.415170 | 0.300400 | 0.300400 |

Эпохи zero-based. e_cls - первый минимум population-validation BCE; e_en - первый минимум positive SmoothL1; e_comp - первый минимум их composite. Scores и energies B0/B1 берутся из разных checkpoints e_cls/e_en. Bcomp - отдельный selector control. Два minima e_cls на эпохе 29 помечены budget-limited, эпохи не продлевались.

Calibration фитится на всех MC-positive validation candidates, а не только выбранных. Используются унаследованные E_true bins 0,3-6,5 ГэВ, минимум 25 объектов/bin. Из median E_pred и median E_true строится монотонная inverse-median map: weighted PAVA, затем linear interpolation и постоянные end ratios за краями. Это не оптимизация MAE и не unfolding. Функция одна для всех потенциалов внутри fold/seed; test не участвует.

Точные float cuts и maps сохранены в complete.json до test; таблица округлена до 6decimal. CSV сканов имеет округлённые записи, поэтому не является источником точного frozen cut. Legacy CLI с 87% energy purity к этой кампании не относится. Глобальная purity 0,70 не означает такую же purity в каждой energy band.

## 5. Физические спектры generated n/p

![Рис. 2. MC truth, full-primary inventory: 568 файлов и 2 345 993 события, без B-selection. Спектры в fixed straight-ray rectangle, не зарегистрированные частицы. Центральные значения и 400 whole-file SD поделены на точные ΔE; нижние панели - отношения всех трёх пар. Показаны все заполненные bins; выше последнего края 7,77 ГэВ сохранённые counts равны нулю. Данные ниже 0,3 ГэВ цензурированы; не показаны как нулевой physical yield.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/01_primary_spectra.png)

Рис. 2. MC truth, full-primary inventory: 568 файлов и 2 345 993 события, без B-selection. Спектры в fixed straight-ray rectangle, не зарегистрированные частицы. Центральные значения и 400 whole-file SD поделены на точные ΔE; нижние панели - отношения всех трёх пар. Показаны все заполненные bins; выше последнего края 7,77 ГэВ сохранённые counts равны нулю. Данные ниже 0,3 ГэВ цензурированы; не показаны как нулевой physical yield.

Источник - certified aggregate4453896 spectra.npz с точным краем 0,3 ГэВ. Его per_event означает counts/bin/event, а не dN/dE: деление на ширину bin обязательно. В spectral edges нет точных 1 и 2 ГэВ; вертикальные линии показывают определения полос, а Rₙ вычислен отдельными exact counters, не интегрированием ближайших bins.

Прямоугольник: проекция луча на z=718,42 см, −146,61≤x≤−106,67 см, |y|≤77,34 см. Он не учитывает module gaps/inactive regions. Proton magnetic response не моделируется. Ниже 0,3 ГэВ отсутствие сохранённых спектральных counts не устанавливает отсутствие мягких primaries.

## 6. Спектры энергии одинаковых MC-positive кластеров

![Рис. 3. Истинный maximum qualifying MCTrack target и A1/B0/B1 reconstructed energies на тех же 640 277 MC-positive candidates. Каждый потенциал нормирован на своё число positive-кандидатов и ΔE. Показано среднее трёх fixed-model histograms, не тройная статистика; полосы - file SD этого среднего.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/02_common_cluster_spectra.png)

Рис. 3. Истинный maximum qualifying MCTrack target и A1/B0/B1 reconstructed energies на тех же 640 277 MC-positive candidates. Каждый потенциал нормирован на своё число positive-кандидатов и ΔE. Показано среднее трёх fixed-model histograms, не тройная статистика; полосы - file SD этого среднего.

Background-кластеры с e_true=0 исключены из energy response: нулевой target означает отсутствие native-positive regression target, а не нулевую физическую энергию. Histogram ratio снизу показывает изменение формы при предсказании, не efficiency относительно generated neutrons.

Не нормируем отдельно видимую часть каждого спектра: underflow/overflow сохраняются. Для A1 при seed456 есть 7 positive predictions≥6,5 ГэВ; для B1 по seed42/123/456 -2/4/5. В raw B0 таких overflow нет. Таблица полных интегралов и хвостов дана в конце.

## 7. Спектры после frozen отбора: A1

![Рис. 4. A1: C2 - true max-target selected positives; C3 - те же exact components с reconstructed energy; C4 - все selected components. Бары/полосы -400 shared file-draw SD. Все стадии используют единый знаменатель all MC-positive; lower panels показывают парные stage ratios.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/03_selected_A1.png)

Рис. 4. A1: C2 - true max-target selected positives; C3 - те же exact components с reconstructed energy; C4 - все selected components. Бары/полосы -400 shared file-draw SD. Все стадии используют единый знаменатель all MC-positive; lower panels показывают парные stage ratios.

На полном диапазоне интегралы C2 и C3 точно совпадают; конечные пределы рисунка не заменяют эту проверку. C4 добавляет native-label negatives. Это не чистый физический фон: native positives уже включают nonprimary tracks, а часть negatives содержит genuine soft primary объекты.

Истинный target кластера - максимум подходящей начальной MCTrack энергии, не deposited energy и не установленная энергия единственного первичного нейтрона. Набор selected positives у A1 и B1 различается; каждый C3 сравнивается со своим C2.

## 8. Спектры после frozen отбора: B1

![Рис. 5. B1: C2 - true max-target selected positives; C3 - те же exact components с reconstructed energy; C4 - все selected components. Бары/полосы -400 shared file-draw SD. Все стадии используют единый знаменатель all MC-positive; lower panels показывают парные stage ratios.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/04_selected_B1.png)

Рис. 5. B1: C2 - true max-target selected positives; C3 - те же exact components с reconstructed energy; C4 - все selected components. Бары/полосы -400 shared file-draw SD. Все стадии используют единый знаменатель all MC-positive; lower panels показывают парные stage ratios.

На полном диапазоне интегралы C2 и C3 точно совпадают; конечные пределы рисунка не заменяют эту проверку. C4 добавляет native-label negatives. Это не чистый физический фон: native positives уже включают nonprimary tracks, а часть negatives содержит genuine soft primary объекты.

Истинный target кластера - максимум подходящей начальной MCTrack энергии, не deposited energy и не установленная энергия единственного первичного нейтрона. Набор selected positives у A1 и B1 различается; каждый C3 сравнивается со своим C2.

## 9. ROC и precision-recall на test

![Рис. 6. Каждый fold и seed показан отдельно: одна pooled ROC смешала бы шкалы scores разных обученных сетей. Точки - исходные frozen operating points. Диагональ ROC - случайный классификатор; горизонталь PR - фактическая prevalence данного fold. AUC/AP в таблице вычислены по исходным полным scores, не по рисунку.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/05_roc_pr.png)

Рис. 6. Каждый fold и seed показан отдельно: одна pooled ROC смешала бы шкалы scores разных обученных сетей. Точки - исходные frozen operating points. Диагональ ROC - случайный классификатор; горизонталь PR - фактическая prevalence данного fold. AUC/AP в таблице вычислены по исходным полным scores, не по рисунку.

Визуализация ROC/PR использует 401-grid принятого validation scan и интерполяцию для отображения; по test не выбирается альтернативный threshold. Полосы -SD 400 file draws, стратифицированных по potential×fold. Стиль линии различает seeds; три seed не являются тремя независимыми тестовыми выборками.

B1 AUC0,9547 выше A1 около 0,8995 на этой продукции. Малый validation/test gap сам по себе не доказывает отсутствие leakage: опора - grouping/split и проверка membership. LOPO и новая production не проверены.

## 10. Score distributions и зависимость от размера

![Рис. 7. Fold-separated score density внутри native MC-positive/negative классов. Среднее трёх fixed-model histograms, file SD; отдельные нормировки классов указаны явно. Эта картинка описывает разделение MC-задачи, а не состав физического neutron sample.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/06_score_distributions.png)

Рис. 7. Fold-separated score density внутри native MC-positive/negative классов. Среднее трёх fixed-model histograms, file SD; отдельные нормировки классов указаны явно. Эта картинка описывает разделение MC-задачи, а не состав физического neutron sample.

Совпадение B0/B1 score cuts связано с общей classification head; калибровка изменяет энергию, не scores. Рабочая purity 0,70 достигается при B cut около 0,30, а при DCM A1 - около 0,35-0,48: purity target и числовой cut различаются.

## 11. Качество по числу observed hits

![Рис. 8. ROC-AUC, count purity, candidate efficiency и energy MAE по размерам 1 /2-3 /4-7 /8+ hits. Центральные значения - equal-fold mean каждого seed, затем mean трёх seed; бары - seed SD. Energy error считается только на MC-positive, efficiency - относительно signal candidates.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/10_cluster_size.png)

Рис. 8. ROC-AUC, count purity, candidate efficiency и energy MAE по размерам 1 /2-3 /4-7 /8+ hits. Центральные значения - equal-fold mean каждого seed, затем mean трёх seed; бары - seed SD. Energy error считается только на MC-positive, efficiency - относительно signal candidates.

У singletons меньше доступной graph information; размерная диагностика не является новым отбором и не удаляет эти кандидаты из основного результата. Удаление сложной категории изменило бы популяцию сравнения.

## 12. Test energy response: все MC-positive

![Рис. 9. Медиана E_reco/E_target, половина 16-84% ширины относительного residual, mean bias и MAE против истинного max-MCTrack target. Raw A0/B0 - dashed, calibrated A1/B1 - solid. Показаны mean ± SD трёх seed binned metrics; это seed spread, не file error и не ошибка квантильной ширины.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/07_energy_mc_positive.png)

Рис. 9. Медиана E_reco/E_target, половина 16-84% ширины относительного residual, mean bias и MAE против истинного max-MCTrack target. Raw A0/B0 - dashed, calibrated A1/B1 - solid. Показаны mean ± SD трёх seed binned metrics; это seed spread, не file error и не ошибка квантильной ширины.

Половина(P84−P16) распределения (E_reco−E_target)/E_target характеризует разброс, а не доверительный интервал среднего. Bins с n<25 отсутствуют/маскируются, не заменяются нулём и не соединяются через пропуск.

Оценка B1 median response ≤0,84% от единицы относится к all-positive профилю 0,3-3 ГэВ на рис.9, отдельно для каждого seed; selected-positive профиль показан на рис.10. Этот вывод не распространяется на весь хвост до 6,5 ГэВ. Калибровка меняет mean bias и ширину и может ухудшать MAE; B0 остаётся лучше B1 по MAE. Selected populations различаются между arms, поэтому их profile не изолирует одну калибровку.

## 13. Test energy response: selected MC-positive

![Рис. 10. Медиана E_reco/E_target, половина 16-84% ширины относительного residual, mean bias и MAE против истинного max-MCTrack target. Raw A0/B0 - dashed, calibrated A1/B1 - solid. Показаны mean ± SD трёх seed binned metrics; это seed spread, не file error и не ошибка квантильной ширины.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/08_energy_selected_positive.png)

Рис. 10. Медиана E_reco/E_target, половина 16-84% ширины относительного residual, mean bias и MAE против истинного max-MCTrack target. Raw A0/B0 - dashed, calibrated A1/B1 - solid. Показаны mean ± SD трёх seed binned metrics; это seed spread, не file error и не ошибка квантильной ширины.

Половина(P84−P16) распределения (E_reco−E_target)/E_target характеризует разброс, а не доверительный интервал среднего. Bins с n<25 отсутствуют/маскируются, не заменяются нулём и не соединяются через пропуск.

Оценка B1 median response ≤0,84% от единицы относится к all-positive профилю 0,3-3 ГэВ на рис.9, отдельно для каждого seed; selected-positive профиль показан на рис.10. Этот вывод не распространяется на весь хвост до 6,5 ГэВ. Калибровка меняет mean bias и ширину и может ухудшать MAE; B0 остаётся лучше B1 по MAE. Selected populations различаются между arms, поэтому их profile не изолирует одну калибровку.

## 14. Энергетическая зависимость отбора

![Рис. 11. Сверху efficiency C2/C1 против true cluster target; снизу purity C3/C4 против reconstructed energy при frozen cuts. Общие file draws сохраняют covariance числителя/знаменателя. Это MC-assisted candidate diagnostics, не эффективность к generated neutrons.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/09_energy_selection.png)

Рис. 11. Сверху efficiency C2/C1 против true cluster target; снизу purity C3/C4 против reconstructed energy при frozen cuts. Общие file draws сохраняют covariance числителя/знаменателя. Это MC-assisted candidate diagnostics, не эффективность к generated neutrons.

Горизонталь 0,70 - только общий validation target. В мягкой reconstructed band и в хвостах purity существенно отличается от него. Поэтому единый count-purity cut сам по себе не оптимизирует и не защищает Rₙ от energy-dependent состава.

## 15. Миграция энергии во всех seed

![Рис. 12. Условные частоты reconstructed-energy bin при фиксированном true-target bin на all MC-positive. Ряды нормированы на все positives данного true bin; reco tails за пределами рисунка сохранены отдельно, сумма видимой строки может быть меньше 1. Цвет - finite-sample frequency, per-cell CI не заявлен.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/11_migration.png)

Рис. 12. Условные частоты reconstructed-energy bin при фиксированном true-target bin на all MC-positive. Ряды нормированы на все positives данного true bin; reco tails за пределами рисунка сохранены отдельно, сумма видимой строки может быть меньше 1. Цвет - finite-sample frequency, per-cell CI не заявлен.

Показаны A1/B1 и каждый seed, а не единственный удобный пример. Матрица относится к maximum-target cluster задаче. Она не включает all-primary detection efficiency и не является готовой матрицей neutron-track response для unfolding.

## 16. Истории и выбор checkpoints

![Рис. 13. Девять population-validation историй BCE и positive SmoothL1 с выбранными zero-based эпохами. Кривые показывают три фиксированных seed, а не confidence band. По каждому head выбран первый argmin; e_cls и e_en используются раздельно.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/12_head_histories.png)

Рис. 13. Девять population-validation историй BCE и positive SmoothL1 с выбранными zero-based эпохами. Кривые показывают три фиксированных seed, а не confidence band. По каждому head выбран первый argmin; e_cls и e_en используются раздельно.

Истории сохранены полностью; последнее значение не подменяет минимум. Два classification minima на последней эпохе отмечены budget-limited; небольшое улучшение после decay не стало основанием продлить 30-epoch бюджет. Это не настройка epochs по test.

## 17. Полные validation сканы score-cut

![Рис. 14. Все девять run и три policies A1/B1/Bcomp. Solid purity, dashed efficiency, вертикальные линии - frozen выбранные cuts. Целевая count purity 0,70 показана горизонталью; все потенциалы объединены в validation. Три seed-кривые не являются CI.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/13_threshold_scans.png)

Рис. 14. Все девять run и три policies A1/B1/Bcomp. Solid purity, dashed efficiency, вертикальные линии - frozen выбранные cuts. Целевая count purity 0,70 показана горизонталью; все потенциалы объединены в validation. Три seed-кривые не являются CI.

Cut выбран первым достижением purity≥0,70 среди 401 значений. Кривые показаны целиком, а не только выбранная точка. F1, ROC-AUC, Rₙ significance или closure gap для выбора не использовались. Недостижение target означало бы not_evaluable, а не скрытый fallback.

## 18. Калибровка: отдельно от test performance

![Рис. 15. VALIDATION diagnostic: raw median response dashed, применение сохранённой monotone calibration solid. Для отчёта карты только переиспользованы, fit не выполнялся. Это та выборка, на которой калибровка построена; test profiles приведены отдельно на рис.9-10.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/14_calibration_validation.png)

Рис. 15. VALIDATION diagnostic: raw median response dashed, применение сохранённой monotone calibration solid. Для отчёта карты только переиспользованы, fit не выполнялся. Это та выборка, на которой калибровка построена; test profiles приведены отдельно на рис.9-10.

Калибровка pooled по потенциалам; отдельный fit для каждого U_sym мог бы поглотить физический спектральный эффект. Медианная коррекция не обязана обнулять mean bias или минимизировать абсолютную ошибку. Статистическая ошибка медиан здесь не оценивалась; показаны фиксированные seed curves.

## 19. Потенциальная разность energy bias

![Рис. 16. Δbias(18−0) и Δbias(90−0) по true energy bins, каждый seed отдельно. Бары - SD 400 whole-file bootstrap. Серое окно[1,75;3)ГэВ унаследовано от критического review, не выбрано по максимуму эффекта. Малые x-offsets служат только читаемости markers.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/15_delta_bias.png)

Рис. 16. Δbias(18−0) и Δbias(90−0) по true energy bins, каждый seed отдельно. Бары - SD 400 whole-file bootstrap. Серое окно[1,75;3)ГэВ унаследовано от критического review, не выбрано по максимуму эффекта. Малые x-offsets служат только читаемости markers.

В окне[1,75;3)ГэВ A1 Δbias(90−0) около−3,6 МэВ, B1 около−0,6 МэВ; file SD порядка 1,8-2,0 и 0,9 МэВ. Изменение между arms не получило собственного paired file-bootstrap в этой старой analysis: нельзя объявлять его разрешённым по двум отдельным error bars. Разность mean E_target≈+3,8 МэВ - контроль спектрального состава внутри окна.

## 20. Разброс между folds

![Рис. 17. Принятая fold-level диагностика и paired reconstruction steps; показатели различаются по fold, а seeds внутри fold используют общие файлы. Ошибки file-bootstrap и разброс seed подписаны отдельно.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/21_bias_folds.png)

Рис. 17. Принятая fold-level диагностика и paired reconstruction steps; показатели различаются по fold, а seeds внутри fold используют общие файлы. Ошибки file-bootstrap и разброс seed подписаны отдельно.

Последующий cross-scoring на одинаковых чётных файлах показал max разность selection steps0,89 п.п., max1,54 SD между двумя B-сетями. Это ограничивает простое объяснение «неустойчивые обучающие файлы». Особенность odd part3 не установлена; её нельзя приписать только модели или только test files. Ранжирование pooled потерь не переносится автоматически на каждый fold.

## 21. Физический отклик: лестница T0-C4

![Рис. 18. Контраст 90/0 при Rₙ=N(E≥2)/N(0,3≤E<1), без upper cap. Слева truth→reconstruction; справа paired B1−A1 по тем же 400 whole-file draws. T0 full primary, T1 processed span, T2 events with candidates; C1 true-target positives, C2 выбранные positives, C3 их reco energy, C4 все selected.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/16_heldout_ladder.png)

Рис. 18. Контраст 90/0 при Rₙ=N(E≥2)/N(0,3≤E<1), без upper cap. Слева truth→reconstruction; справа paired B1−A1 по тем же 400 whole-file draws. T0 full primary, T1 processed span, T2 events with candidates; C1 true-target positives, C2 выбранные positives, C3 их reco energy, C4 все selected.

T1=+4,72±0,57%; B1 C4=+1,94/+2,01/+1,82% при SD≈0,9%. Gate G2 не пройден: отклик не замыкается. B1−A1 C4=+0,80±0,88 /+0,12±1,00 /−0,86±1,03 п.п. по seed, то есть улучшение конечного контраста не разрешено.

C4−C3 - вклад native-label negatives, не чистый secondary/charged background. В pooled B1 его величина около−1,1…−1,3 п.п.; вклад selection около−0,4…−0,6. На отдельных folds их знак и порядок меняются. G6 pass означает лишь малый наблюдаемый 3-seed spread, не устранение file/domain зависимости.

## 22. Все пары потенциала симметрии

| U_high/U_low (МэВ) | Seed | T1 (%) | A1 C4 (%) | B1 C4 (%) | B1 C4−T1 (п.п.) |
| --- | --- | --- | --- | --- | --- |
| 18/0 | 42 | 0.70 ± 0.60 | -0.45 ± 0.94 | -0.52 ± 0.81 | -1.21 ± 0.86 |
| 18/0 | 123 | 0.70 ± 0.60 | -0.99 ± 0.97 | -0.05 ± 0.81 | -0.75 ± 0.84 |
| 18/0 | 456 | 0.70 ± 0.60 | -0.6 ± 1.0 | -0.44 ± 0.77 | -1.14 ± 0.81 |
| 90/0 | 42 | 4.72 ± 0.57 | 1.15 ± 0.95 | 1.94 ± 0.86 | -2.77 ± 0.93 |
| 90/0 | 123 | 4.72 ± 0.57 | 1.9 ± 1.1 | 2.01 ± 0.89 | -2.70 ± 0.94 |
| 90/0 | 456 | 4.72 ± 0.57 | 2.7 ± 1.1 | 1.82 ± 0.85 | -2.90 ± 0.91 |
| 90/18 | 42 | 3.99 ± 0.61 | 1.6 ± 1.0 | 2.47 ± 0.95 | -1.52 ± 0.96 |
| 90/18 | 123 | 3.99 ± 0.61 | 2.9 ± 1.2 | 2.07 ± 0.94 | -1.92 ± 0.92 |
| 90/18 | 456 | 3.99 ± 0.61 | 3.3 ± 1.1 | 2.27 ± 0.91 | -1.72 ± 0.90 |

Это processed-span T1 той же ladder, не full-primary headline статьи. Все ошибки -400 saved joint file-bootstrap SD; seed показан отдельно. Для 90/18 в каждой replica вычислено 100×[(1+Δ90/0/100)/(1+Δ18/0/100)−1], проценты не вычитались. Три пары алгебраически зависимы, не являются тремя независимыми подтверждениями.

![Рис. 19. T1 MC truth, A1/B1 reconstructed contrast и paired C4−T1. Вертикальные масштабы панелей различаются. Пара 18/0 не разрешена; меньший дефицит 90/18 не доказывает замыкания во всех seeds.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/17_all_potential_pairs.png)

Рис. 19. T1 MC truth, A1/B1 reconstructed contrast и paired C4−T1. Вертикальные масштабы панелей различаются. Пара 18/0 не разрешена; меньший дефицит 90/18 не доказывает замыкания во всех seeds.

## 23. Необходимость MC-label floor: новый census

![Рис. 20. Только S286, неизменные observed graphs, без model calls. Grid off/0,1/0,2/0,3/0,4/0,5 ГэВ зафиксирован до расчёта. Native0,3 labels/float32 targets совпали точно на всех 286 файлах. Бары -SD 400 potential×part group draws.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/18_label_floor_scan.png)

Рис. 20. Только S286, неизменные observed graphs, без model calls. Grid off/0,1/0,2/0,3/0,4/0,5 ГэВ зафиксирован до расчёта. Native0,3 labels/float32 targets совпали точно на всех 286 файлах. Бары -SD 400 potential×part group draws.

На 1 798 225 кластерах positives322058→327538: снятие energy clause добавляет 5480, или 1,70% к native positives, около 0,305 п.п. ко всем candidates. Это новая разметка наблюдаемой задачи, не новая оценка ROC-AUC или detector efficiency. No-floor модели пока не обучались.

## 24. Какие мягкие объекты исключались

![Рис. 21. S286:3500 direct-primary-only и 1980 nonprimary-only по qualifying-хитам;4087 добавленных singletons. Все 5480 targets строго<0,3 ГэВ:859/1726/2895 в bins[0;0,1)/[0,1;0,2)/[0,2;0,3). Бары -shared group SD.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/19_soft_origins.png)

Рис. 21. S286:3500 direct-primary-only и 1980 nonprimary-only по qualifying-хитам;4087 добавленных singletons. Все 5480 targets строго<0,3 ГэВ:859/1726/2895 в bins[0;0,1)/[0,1;0,2)/[0,2;0,3). Бары -shared group SD.

Only относится к qualifying-хитам, не чистоте всей компоненты. Nonprimary не означает автоматически фон; полной genealogy нет. Counts кластеров не равны числу независимо реконструированных нейтронов. Наблюдаемые hits≥35 нс присутствуют, так что условие аналитической избыточности floor при универсальном t<35 нс здесь не выполнено.

На S при fixed predictions/cut C1/C2 и C4 в нынешних Rₙ bands не меняются; C3 может меняться из-за диагностического relabelling migrated soft объектов. Прямого исправления C1−T2 gap снятием floor не показано. Знак и размер влияния переобучения остаются неизвестными.

## 25. Интегралы спектров и хвосты

| Arm | Seed | All MC+ | C2=C3 total | C4 total | All-MC+ Epred≥6,5 ГэВ | C3≥6,5 ГэВ | C4≥6,5 ГэВ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 42 | 640277 | 351680 | 501785 | 0 | 0 | 0 |
| A1 | 123 | 640277 | 357240 | 509780 | 0 | 0 | 0 |
| A1 | 456 | 640277 | 339049 | 483239 | 7 | 7 | 7 |
| B0 | 42 | 640277 | 515332 | 734177 | 0 | 0 | 0 |
| B0 | 123 | 640277 | 515256 | 733975 | 0 | 0 | 0 |
| B0 | 456 | 640277 | 514637 | 732165 | 0 | 0 | 0 |
| B1 | 42 | 640277 | 515332 | 734177 | 2 | 2 | 3 |
| B1 | 123 | 640277 | 515256 | 733975 | 4 | 3 | 6 |
| B1 | 456 | 640277 | 514637 | 732165 | 5 | 4 | 9 |

Counts точны для конечной accepted test population; это не population estimates с binomial ошибкой. В этих сохранённых A1/B0/B1 outputs отрицательных reconstructed energies и positive predictions<0,1 ГэВ нет. Они всё равно присутствуют как отдельные underflow categories в export. Истинные native positives ниже 0,3 ГэВ отсутствуют по определению label, а не по утверждению о физическом спектре.

При полном диапазоне C2 и C3 имеют одинаковое число объектов; таблица это проверяет для каждого arm/seed. C4 больше из-за native negatives. Нормировка спектров не перераспределяет пропущенный хвост обратно в видимую область. Counts.npz хранит все negative/finite/overflow bins и per-file matrices; data.json привязывает каждый прочитанный CSV к accepted SHA.

## 26. Подбор энергетических границ: сделано и не сделано

Score cuts подобраны на validation по указанному правилу; e_cls/e_en и calibration также зафиксированы до test. Label floor0,3 и bands[0,3;1)/≥2 ГэВ не являются результатом оптимизации Rₙ. Дополнительная reference с finite high cap3 ГэВ относится к будущему скану; текущий опубликованный здесь числитель открыт вверх.

Аналитическая оценка не выводит 0,3 ГэВ из ToF resolution. При 7 м окно 35 нс дало бы Tmin≈0,322 ГэВ; observed path такого универсального окна не задаёт. Эти кинематические оценки не устанавливают full efficiency или фон детектора. Скан 148 окон не запускался: требуются exact uncensored primary counters и заранее фиксированное правило выбора с возможностью «ни одно окно не подходит».

Следующий зарегистрированный опыт - один no-floor arm с теми же graph/scaler/folds/recipe/seeds, всего 30 эпох, максимум 2 одновременных задач. План 15+15 эпох с coherent resume. Девять baseline не повторяются; ожидаемая стоимость 27-28 GPU·ч fit, около 30-33 GPU·ч с validation/I/O. Подготовка adapters и конкретный preflight ещё не равны запуску обучения.

Все arms будут оцениваться против одинаковых reference0,3 и disabled. Primary сравнение энергии - raw на common positives и отдельно added soft support; сначала frozen baseline cuts. Иначе изменение label, operating point и calibration нельзя было бы отделить. F282 (282 truth-bound группы нечётных частей 1/3/5) уже открывались baseline и не являются свежим holdout; для настройки нового выбора используются только S286.

Раздельные модели по потенциалу сейчас не обучаются; U_sym не служит условием выбора реконструктора. DCM сохраняется transfer-контролем; joint DCM/SMASH или pretrain→finetune требуют отдельной абляции. Нужны новая production/генератор и LOPO, чтобы говорить об обобщении вне этой SMASH смеси.

## 27. Полнота данных: census и дизайн split

![Рис. 22. Принятый input census/shared-record diagnostic. Дополнительные exports не означают независимые события. Counts описывают конечный инвентарь; статистическая ошибка к числу записанных файлов не приписывается.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/22_census.png)

Рис. 22. Принятый input census/shared-record diagnostic. Дополнительные exports не означают независимые события. Counts описывают конечный инвентарь; статистическая ошибка к числу записанных файлов не приписывается.

![Рис. 23. Принятая до обучения проекция counting precision против доли test. Это обоснование split trade-off, не измеренная uncertainty обученной сети и не подбор split по лучшему ROC-AUC.](/Users/vovvy/Project/BM@N/HGND/HGNDRecoGNN/reports/smash_performance_20261103/figures/23_split_precision.png)

Рис. 23. Принятая до обучения проекция counting precision против доли test. Это обоснование split trade-off, не измеренная uncertainty обученной сети и не подбор split по лучшему ROC-AUC.

## 28. Воспроизводимость и границы вывода

Этот отчёт читает только сохранённые accepted predictions, complete/history/threshold/calibration artefacts и certified primary spectra. Проверены 5112 CSV для A1/B0/B1, candidate keys/truth и per-fold counts воспроизвелись. Никаких inference, raw-event reduction, новых fit, threshold choice или training jobs отчёт не запускает. Дополнительные unit tests не выполнялись по прямому указанию пользователя; арифметика и SHA assertions остаются частью export.

Ключевые источники: results/smash_train_20261028/results_table.json; analysis/summary.json; heldout_rn/summary.json и per_file_counts.json; train_4466016/task_*/complete.json; evaluate_*/extracted/evaluation/predictions; certified aggregate4453896 spectra; reports/status_20261103_support/all_pair_contrasts.json; reports/energy_label_results_20261103/. Input pins, file hashes, figure manifest, per-file histogram counts и PDF validation находятся рядом с этим отчётом.

Не установлены: независимый generator/production holdout; physical unfolded spectrum; neutron-track response; эффективность к всем generated neutrons; корректность proton magnetic acceptance; centrality dependence при B недоступном. Улучшение native ROC-AUC не заменяет проверку Rₙ. Seed SD и file SD измеряют разные источники вариации и не включают транспортные систематики.

Сохранены отрицательные результаты: незамыкание Rₙ, отсутствие разрешённого C4 улучшения B1−A1, MAE B1 хуже B0, условный характер fold explanations. Ни слабый 18/0 контраст, ни 90/18 не превращаются в самостоятельное подтверждение symmetry-energy measurement. Manuscript/canonical inputs и его PDF этим отчётом не меняются.

Воспроизведение: collect.py (проверенные CSV→counts), plot.py (figures), build_report.py (единые Markdown/PDF блоки). Исторические census/design figures сохранены по исходным SHA, их vector scale увеличен только для читаемости. Validation содержит проверки text/data bounds, embedded fonts, постраничный render и read-only critic review.
