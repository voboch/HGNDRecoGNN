"""Build the full Russian SMASH performance report with vector figures."""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output/pdf/hgnd_smash_performance_20261103.pdf"
MD = HERE.parent / "smash_performance_20261103.md"
SOURCE = json.loads((HERE / "input_pins.json").read_text())
for source, digest in SOURCE["files"].items():
    assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest, source
TABLE = json.loads((ROOT / "results/smash_train_20261028/results_table.json").read_text())
ANALYSIS = json.loads((ROOT / "results/smash_train_20261028/analysis/summary.json").read_text())
PAIRS = json.loads((ROOT / "reports/status_20261103_support/all_pair_contrasts.json").read_text())
LABEL = json.loads((ROOT / "reports/energy_label_results_20261103/results.json").read_text())
PLOT = json.loads((HERE / "plot_statistics.json").read_text())
DATA = json.loads((HERE / "data.json").read_text())
SEEDS = ("42", "123", "456")


def H(s): return {"type": "h", "text": s}
def P(s): return {"type": "p", "text": s}
def T(h, r, w=None): return {"type": "table", "head": h, "rows": r, "widths": w}
def F(name, s): return {"type": "figure", "name": name, "caption": s}


def fmt(v, sd, factor=1):
    v, sd = v*factor, sd*factor
    if sd == 0:
        return f"{v:.3f} ± 0"
    dec = max(0, 1-math.floor(math.log10(abs(sd))))
    return f"{v:.{dec}f} ± {sd:.{dec}f}"


def metric(arm, key, factor=1):
    j = TABLE["classification_energy"]["arms"][arm][key]
    return fmt(j["mean"], j["seed_sd"], factor)


pages = []
pages.append([
    {"type": "title", "text": "HGND: качество реконструкции на SMASH"},
    H("Полный отчёт • 3 ноября 2026"),
    P("Исправленный SMASH Xe+Cs, E_kin/A = 2,87 ГэВ; U_sym = 0/18/90 МэВ. Итог: адаптация GraphSAGE к SMASH улучшила классификацию и оценку cluster energy, но полный физический отклик Rₙ пока не восстановлен. Ни новых моделей, ни новых порогов для этого отчёта не подбиралось."),
    P("Статус: девять baseline обучений по 30 эпох и три оценки приняты 29.10; новый census разметки на S286 (286 truth-bound групп чётных частей0/2/4) принят 03.11. No-floor arm ещё не обучен. Этот документ объединяет результаты, спектры, все доступные семейства диагностики и описание фактического выбора порогов. Он дополняет старую сводку smash_train_results_20261029.md; её исходные результаты сохраняются."),
    T(["Arm", "ROC-AUC ↑", "AP ↑", "Purity (%)", "Eff. (%) ↑", "Bias (МэВ) →0", "MAE (МэВ) ↓"],
      [[a, metric(a, "auc"), metric(a, "ap"), metric(a, "purity", 100), metric(a, "efficiency", 100), metric(a, "bias_GeV", 1000), metric(a, "mae_GeV", 1000)] for a in ("A0", "A1", "B0", "B1", "Bcomp")], [.08,.15,.15,.14,.14,.17,.17]),
    P("В таблице mean ± SD трёх training seeds; каждый seed сначала усреднён равными весами по трём fold-метрикам. Это разброс инициализации, не полная file-sampling ошибка. Оценка: 568 truth-bound test-файлов, 3 572 710 кандидатов, 640 277 MC-positive. Energy bias/MAE — на всех этих positives, target — максимум начальной энергии подходящего MCTrack в кластере."),
    P("Парное сравнение B1−A1 по трём общим training seeds (разные веса и scaler): ΔAUC="+fmt(TABLE['classification_energy']['B1_minus_A1']['auc']['mean'],TABLE['classification_energy']['B1_minus_A1']['auc']['se_of_mean'])+", t=13,09; Δefficiency="+fmt(TABLE['classification_energy']['B1_minus_A1']['efficiency']['mean'],TABLE['classification_energy']['B1_minus_A1']['efficiency']['se_of_mean'],100)+"п.п., t=32,01; ΔMAE="+fmt(TABLE['classification_energy']['B1_minus_A1']['mae_GeV']['mean'],TABLE['classification_energy']['B1_minus_A1']['mae_GeV']['se_of_mean'],1000)+"МэВ, t=−11,04. Здесь ±SE среднего парных разностей, df=2, двусторонний t_crit=4,303; эти три изменения разрешены по registered seed-критерию. Это не ошибка между независимыми productions."),
    P("B0 имеет MAE 81,8 МэВ, B1 — 84,7 МэВ: калибровка улучшает медианный response, но не минимизирует MAE. У B1 reconstructed 90/0 контраст Rₙ = +1,82…+2,01% по seed при T1 MC truth +4,72 ± 0,57%; парный дефицит около 2,7–2,9 п.п. сохраняется."),
])

pages.append([
    H("1. Датасеты, обучение и допустимое сравнение"),
    T(["Arm", "Веса и scaler", "Cut и energy postprocessing"], [
        ["A0", "Frozen DCM GraphSAGE, три seed; frozen DCM scaler", "DCM-validation cut, raw energy"],
        ["A1", "Те же DCM веса и scaler", "Cut и calibration на SMASH fold-validation"],
        ["B0", "SMASH fold-train, один train-fit scaler на сторону", "SMASH-validation cut; raw energy"],
        ["B1", "Те же B0 веса; scores e_cls, energy e_en", "Тот же B cut; validation calibration"],
        ["Bcomp", "SMASH, один composite-selected checkpoint", "Свой validation cut и calibration"],
    ], [.12,.43,.45]),
    P("B1−A1 сравнивает адаптированные pipelines: отличаются веса и scaler. Это не чистая абляция только обучающей выборки или только весов. Observations, графы, candidates и MC labels общие. Scalers B фитятся только на train-файлах fold, в смеси всех трёх потенциалов; U_sym и имя файла не являются признаками. Cut и calibration не подбираются отдельно для потенциалов."),
    P("Каждый fold: student_sage H128/L4, dropout0,1, batch256, Adam lr10⁻³, weight decay10⁻⁵, 30 эпох, decays в эпохах10/15/25 с gamma0,1. Loss: BCE + 0,5 SmoothL1 на positives. Это унаследованный рецепт, не результат оптимизации на данном физическом отклике. Seeds42/123/456; гиперпараметры в этой кампании не сканировались."),
    T(["Популяция", "0 МэВ", "18 МэВ", "90 МэВ", "Значение"], [
        ["Full primary events", "804369", "824338", "717286", "2 345 993, включая пустые"],
        ["Processed-span events T1", "804300", "824270", "717241", "2 345 811, Row<processed_span"],
        ["Events with candidates T2", "594190", "610225", "531709", "1 736 124, есть observed candidates"],
        ["Truth-bound test files", "195", "200", "173", "568 групп"],
    ], [.30,.14,.14,.14,.28]),
    P("Full-primary спектры, T1 и candidate-level response имеют разные знаменатели. Их нельзя делить друг на друга и называть detector efficiency. Для протонов прямолинейное окно не включает магнитный транспорт и tracking efficiency."),
])

pages.append([
    H("2. Разбиение, повторения и нормировка"),
    F("20_crossfit", "Рис. 1. Схема accepted cross-fitting. Группы построены по content/event/direct-primary связям; random cluster split не применяется. Это схема ролей, не измерение; точные counts приведены ниже."),
    T(["Fold", "Train: 0/18/90", "Validation: 0/18/90", "Test: 0/18/90", "Unbound в test parts"], [
        ["0", "96 / 99 / 90", "33 / 33 / 31", "66 / 68 / 58", "4"],
        ["1", "98 / 101 / 92", "32 / 33 / 30", "65 / 66 / 58", "3"],
        ["2", "98 / 100 / 92", "33 / 34 / 31", "64 / 66 / 57", "3"],
    ], [.09,.24,.25,.24,.18]),
    P("578 content groups, 568 truth-bound. Train/validation включают unbound detector-labelled группы:5/1,6/1,5/2 по folds. Они не превращаются в negatives; в физический test входят только bound группы. Внутри fold train/validation/test не пересекаются; train-наборы разных folds пересекаются."),
    P("Каждый test-файл оценён моделями, которые не использовали его для своего train или validation. Но frozen transfer уже открывал эти данные: это retrospective cross-fitting, не свежий blind holdout. DCM student≈0,98 относится к иной random-split задаче с повторяющимися inputs; для этой pipeline не является независимым целевым benchmark."),
])

pages.append([
    H("3. Все пороги: определения и мотивация"),
    T(["Условие", "Фактический смысл / основание", "Статус"], [
        ["Edep >3 МэВ", "Отбор наблюдаемых hits; не равен энергии падающего нейтрона", "Унаследованный observation cut"],
        ["Jitter σ=0,15 нс", "Deterministic path-seeded Gaussian на hits", "Один зафиксированный шумовой view"],
        ["Label Ekin >0,3 ГэВ", "Строгий floor начальной энергии подходящего MCTrack", "Историческое определение; теперь проверяется no-floor"],
        ["ToF label ±0,3 нс", "Prompt-согласие Ekin с ToF; условно2σ однохитового jitter при идеальном start", "Унаследовано, не оптимизировано"],
        ["Graph radius3,6", "Расстояние в индексах Layer/Row/Column, не подтверждённые3,6×4см", "Унаследовано"],
        ["Graph Δt1 нс; ≤32 соседей", "Пересечение spatial/temporal radius_graph внутри event/side; компоненты могут образовывать цепочки", "Не DAQ window и не предел времени всей компоненты"],
        ["Count purity target0,70", "N(selected native-positive)/N(selected)", "Унаследованный компромисс, не optimum Rₙ"],
        ["Score cut", "Первый из401 cuts, достигший target на fold-validation", "Подобран; затем frozen на test"],
        ["Rₙ low [0,3;1), high ≥2", "Определение наблюдаемой, отдельное от MC-label floor", "1/2 ГэВ унаследованы; не оптимизировались"],
        ["35 нс", "Опубликованный операционный пример; в observe/transfer_pack такого cut нет", "Не считать применённым"],
    ], [.24,.50,.26]),
    P("Графовые параметры не выводят автоматически физический размер shower или neutron-track response. Требование ToF относится к MC-разметке и использует truth, а не к самостоятельному измерению. MC-label требует Ekin>0,3 ГэВ, а нижняя полоса Rₙ включает E≥0,3 ГэВ: это также разные граничные конвенции."),
])

cutrows=[]
for task in range(9):
    j=json.loads((ROOT/f"results/smash_train_20261028/train_4466016/task_{task}/complete.json").read_text())
    c=j['cuts'];q=j['selection']
    cutrows.append([j['fold'],j['seed'],q['e_cls'],q['e_en'],q['e_comp'],f"{c['A0']:.6f}",f"{c['A1']:.6f}",f"{c['B1']:.6f}",f"{c['Bcomp']:.6f}"])
pages.append([
    H("4. Как выбраны score cut, checkpoints и calibration"),
    P("Фактический путь: smash_train.choose_cut → reco_performance.threshold_scan. На pooled-potential validation каждого fold/seed перебираются401 равноотстоящий threshold от0,001 до0,999; selection строго score>threshold. Выбирается первый, то есть минимальный grid cut с count purity≥0,70. Если такого cut нет, run не evaluable. Это не F1/Youden/AUC optimum и не подбор по test или Rₙ."),
    T(["Fold", "Seed", "e_cls", "e_en", "e_comp", "A0 cut", "A1 cut", "B0=B1 cut", "Bcomp cut"],cutrows,[.07,.08,.08,.08,.10,.14,.15,.15,.15]),
    P("Эпохи zero-based. e_cls — первый минимум population-validation BCE; e_en — первый минимум positive SmoothL1; e_comp — первый минимум их composite. Scores и energies B0/B1 берутся из разных checkpoints e_cls/e_en. Bcomp — отдельный selector control. Два minima e_cls на эпохе29 помечены budget-limited, эпохи не продлевались."),
    P("Calibration фитится на всех MC-positive validation candidates, а не только выбранных. Используются унаследованные E_true bins0,3–6,5ГэВ, минимум25 объектов/bin. Из median E_pred и median E_true строится монотонная inverse-median map: weighted PAVA, затем linear interpolation и постоянные end ratios за краями. Это не оптимизация MAE и не unfolding. Функция одна для всех потенциалов внутри fold/seed; test не участвует."),
    P("Точные float cuts и maps сохранены в complete.json до test; таблица округлена до6decimal. CSV сканов имеет округлённые записи, поэтому не является источником точного frozen cut. Legacy CLI с87% energy purity к этой кампании не относится. Глобальная purity0,70 не означает такую же purity в каждой energy band."),
])

def figure_page(title,name,caption,paragraphs=()):
    pages.append([H(title),F(name,caption),*[P(p) for p in paragraphs]])

figure_page("5. Физические спектры generated n/p", "01_primary_spectra",
    "Рис. 2. MC truth, full-primary inventory: 568 файлов и2 345 993 события, без B-selection. Спектры в fixed straight-ray rectangle, не зарегистрированные частицы. Центральные значения и400 whole-file SD поделены на точные ΔE; нижние панели — отношения всех трёх пар. Показаны все заполненные bins; выше последнего края7,77ГэВ сохранённые counts равны нулю. Данные ниже0,3ГэВ цензурированы; не показаны как нулевой physical yield.",
    ["Источник — certified aggregate4453896 spectra.npz с точным краем0,3ГэВ. Его per_event означает counts/bin/event, а не dN/dE: деление на ширину bin обязательно. В spectral edges нет точных1 и2ГэВ; вертикальные линии показывают определения полос, а Rₙ вычислен отдельными exact counters, не интегрированием ближайших bins.",
     "Прямоугольник: проекция луча на z=718,42см, −146,61≤x≤−106,67см, |y|≤77,34см. Он не учитывает module gaps/inactive regions. Proton magnetic response не моделируется. Ниже0,3ГэВ отсутствие сохранённых спектральных counts не устанавливает отсутствие мягких primaries."])

figure_page("6. Спектры энергии одинаковых MC-positive кластеров", "02_common_cluster_spectra",
    "Рис. 3. Истинный maximum qualifying MCTrack target и A1/B0/B1 reconstructed energies на тех же640 277 MC-positive candidates. Каждый потенциал нормирован на своё число positive-кандидатов и ΔE. Показано среднее трёх fixed-model histograms, не тройная статистика; полосы — file SD этого среднего.",
    ["Background-кластеры с e_true=0 исключены из energy response: нулевой target означает отсутствие native-positive regression target, а не нулевую физическую энергию. Histogram ratio снизу показывает изменение формы при предсказании, не efficiency относительно generated neutrons.",
     "Не нормируем отдельно видимую часть каждого спектра: underflow/overflow сохраняются. Для A1 при seed456 есть7 positive predictions≥6,5ГэВ; для B1 по seed42/123/456 —2/4/5. В raw B0 таких overflow нет. Таблица полных интегралов и хвостов дана в конце."])

for section,name,arm in (("7", "03_selected_A1", "A1"),("8", "04_selected_B1", "B1")):
    figure_page(f"{section}. Спектры после frozen отбора: {arm}",name,
        f"Рис. {4 if arm=='A1' else 5}. {arm}: C2 — true max-target selected positives; C3 — те же exact components с reconstructed energy; C4 — все selected components. Бары/полосы —400 shared file-draw SD. Все стадии используют единый знаменатель all MC-positive; lower panels показывают парные stage ratios.",
        ["На полном диапазоне интегралы C2 и C3 точно совпадают; конечные пределы рисунка не заменяют эту проверку. C4 добавляет native-label negatives. Это не чистый физический фон: native positives уже включают nonprimary tracks, а часть negatives содержит genuine soft primary объекты.",
         "Истинный target кластера — максимум подходящей начальной MCTrack энергии, не deposited energy и не установленная энергия единственного первичного нейтрона. Набор selected positives у A1 и B1 различается; каждый C3 сравнивается со своим C2."])

figure_page("9. ROC и precision-recall на test", "05_roc_pr",
    "Рис. 6. Каждый fold и seed показан отдельно: одна pooled ROC смешала бы шкалы scores разных обученных сетей. Точки — исходные frozen operating points. Диагональ ROC — случайный классификатор; горизонталь PR — фактическая prevalence данного fold. AUC/AP в таблице вычислены по исходным полным scores, не по рисунку.",
    ["Визуализация ROC/PR использует401-grid принятого validation scan и интерполяцию для отображения; по test не выбирается альтернативный threshold. Полосы —SD400 file draws, стратифицированных по potential×fold. Стиль линии различает seeds; три seed не являются тремя независимыми тестовыми выборками.",
     "B1 AUC0,9547 выше A1 около0,8995 на этой продукции. Малый validation/test gap сам по себе не доказывает отсутствие leakage: опора — grouping/split и проверка membership. LOPO и новая production не проверены."])

figure_page("10. Score distributions и зависимость от размера", "06_score_distributions",
    "Рис. 7. Fold-separated score density внутри native MC-positive/negative классов. Среднее трёх fixed-model histograms, file SD; отдельные нормировки классов указаны явно. Эта картинка описывает разделение MC-задачи, а не состав физического neutron sample.",
    ["Совпадение B0/B1 score cuts связано с общей classification head; калибровка изменяет энергию, не scores. Рабочая purity0,70 достигается при B cut около0,30, а при DCM A1 — около0,35–0,48: purity target и числовой cut различаются."])

figure_page("11. Качество по числу observed hits", "10_cluster_size",
    "Рис. 8. ROC-AUC, count purity, candidate efficiency и energy MAE по размерам1 /2–3 /4–7 /8+ hits. Центральные значения — equal-fold mean каждого seed, затем mean трёх seed; бары — seed SD. Energy error считается только на MC-positive, efficiency — относительно signal candidates.",
    ["У singletons меньше доступной graph information; размерная диагностика не является новым отбором и не удаляет эти кандидаты из основного результата. Удаление сложной категории изменило бы популяцию сравнения."])

for section,name,label in (("12","07_energy_mc_positive","все MC-positive"),("13","08_energy_selected_positive","selected MC-positive")):
    figure_page(f"{section}. Test energy response: {label}", name,
        "Рис. "+("9" if section=="12" else "10")+". Медиана E_reco/E_target, половина16–84% ширины относительного residual, mean bias и MAE против истинного max-MCTrack target. Raw A0/B0 — dashed, calibrated A1/B1 — solid. Показаны mean ± SD трёх seed binned metrics; это seed spread, не file error и не ошибка квантильной ширины.",
        ["Половина(P84−P16) распределения (E_reco−E_target)/E_target характеризует разброс, а не доверительный интервал среднего. Bins с n<25 отсутствуют/маскируются, не заменяются нулём и не соединяются через пропуск.",
         "Оценка B1 median response ≤0,84% от единицы относится к all-positive профилю0,3–3ГэВ на рис.9, отдельно для каждого seed; selected-positive профиль показан на рис.10. Этот вывод не распространяется на весь хвост до6,5ГэВ. Калибровка меняет mean bias и ширину и может ухудшать MAE; B0 остаётся лучше B1 по MAE. Selected populations различаются между arms, поэтому их profile не изолирует одну калибровку."])

figure_page("14. Энергетическая зависимость отбора", "09_energy_selection",
    "Рис. 11. Сверху efficiency C2/C1 против true cluster target; снизу purity C3/C4 против reconstructed energy при frozen cuts. Общие file draws сохраняют covariance числителя/знаменателя. Это MC-assisted candidate diagnostics, не эффективность к generated neutrons.",
    ["Горизонталь0,70 — только общий validation target. В мягкой reconstructed band и в хвостах purity существенно отличается от него. Поэтому единый count-purity cut сам по себе не оптимизирует и не защищает Rₙ от energy-dependent состава."])

figure_page("15. Миграция энергии во всех seed", "11_migration",
    "Рис. 12. Условные частоты reconstructed-energy bin при фиксированном true-target bin на all MC-positive. Ряды нормированы на все positives данного true bin; reco tails за пределами рисунка сохранены отдельно, сумма видимой строки может быть меньше1. Цвет — finite-sample frequency, per-cell CI не заявлен.",
    ["Показаны A1/B1 и каждый seed, а не единственный удобный пример. Матрица относится к maximum-target cluster задаче. Она не включает all-primary detection efficiency и не является готовой матрицей neutron-track response для unfolding."])

figure_page("16. Истории и выбор checkpoints", "12_head_histories",
    "Рис. 13. Девять population-validation историй BCE и positive SmoothL1 с выбранными zero-based эпохами. Кривые показывают три фиксированных seed, а не confidence band. По каждому head выбран первый argmin; e_cls и e_en используются раздельно.",
    ["Истории сохранены полностью; последнее значение не подменяет минимум. Два classification minima на последней эпохе отмечены budget-limited; небольшое улучшение после decay не стало основанием продлить30-epoch бюджет. Это не настройка epochs по test."])

figure_page("17. Полные validation сканы score-cut", "13_threshold_scans",
    "Рис. 14. Все девять run и три policies A1/B1/Bcomp. Solid purity, dashed efficiency, вертикальные линии — frozen выбранные cuts. Целевая count purity0,70 показана горизонталью; все потенциалы объединены в validation. Три seed-кривые не являются CI.",
    ["Cut выбран первым достижением purity≥0,70 среди401 значений. Кривые показаны целиком, а не только выбранная точка. F1, ROC-AUC, Rₙ significance или closure gap для выбора не использовались. Недостижение target означало бы not_evaluable, а не скрытый fallback."])

figure_page("18. Калибровка: отдельно от test performance", "14_calibration_validation",
    "Рис. 15. VALIDATION diagnostic: raw median response dashed, применение сохранённой monotone calibration solid. Для отчёта карты только переиспользованы, fit не выполнялся. Это та выборка, на которой калибровка построена; test profiles приведены отдельно на рис.9–10.",
    ["Калибровка pooled по потенциалам; отдельный fit для каждого U_sym мог бы поглотить физический спектральный эффект. Медианная коррекция не обязана обнулять mean bias или минимизировать абсолютную ошибку. Статистическая ошибка медиан здесь не оценивалась; показаны фиксированные seed curves."])

figure_page("19. Потенциальная разность energy bias", "15_delta_bias",
    "Рис. 16. Δbias(18−0) и Δbias(90−0) по true energy bins, каждый seed отдельно. Бары — SD400 whole-file bootstrap. Серое окно[1,75;3)ГэВ унаследовано от критического review, не выбрано по максимуму эффекта. Малые x-offsets служат только читаемости markers.",
    ["В окне[1,75;3)ГэВ A1 Δbias(90−0) около−3,6МэВ, B1 около−0,6МэВ; file SD порядка1,8–2,0 и0,9МэВ. Изменение между arms не получило собственного paired file-bootstrap в этой старой analysis: нельзя объявлять его разрешённым по двум отдельным error bars. Разность mean E_target≈+3,8МэВ — контроль спектрального состава внутри окна."])

figure_page("20. Разброс между folds", "21_bias_folds",
    "Рис. 17. Принятая fold-level диагностика и paired reconstruction steps; показатели различаются по fold, а seeds внутри fold используют общие файлы. Ошибки file-bootstrap и разброс seed подписаны отдельно.",
    ["Последующий cross-scoring на одинаковых чётных файлах показал max разность selection steps0,89п.п., max1,54SD между двумя B-сетями. Это ограничивает простое объяснение «неустойчивые обучающие файлы». Особенность odd part3 не установлена; её нельзя приписать только модели или только test files. Ранжирование pooled потерь не переносится автоматически на каждый fold."])

figure_page("21. Физический отклик: лестница T0–C4", "16_heldout_ladder",
    "Рис. 18. Контраст90/0 при Rₙ=N(E≥2)/N(0,3≤E<1), без upper cap. Слева truth→reconstruction; справа paired B1−A1 по тем же400 whole-file draws. T0 full primary, T1 processed span, T2 events with candidates; C1 true-target positives, C2 выбранные positives, C3 их reco energy, C4 все selected.",
    ["T1=+4,72±0,57%; B1 C4=+1,94/+2,01/+1,82% при SD≈0,9%. Gate G2 не пройден: отклик не замыкается. B1−A1 C4=+0,80±0,88 /+0,12±1,00 /−0,86±1,03п.п. по seed, то есть улучшение конечного контраста не разрешено.",
     "C4−C3 — вклад native-label negatives, не чистый secondary/charged background. В pooled B1 его величина около−1,1…−1,3п.п.; вклад selection около−0,4…−0,6. На отдельных folds их знак и порядок меняются. G6 pass означает лишь малый наблюдаемый3-seed spread, не устранение file/domain зависимости."])

pairrows=[]
for key,p in PAIRS['pairs'].items():
    for seed in SEEDS:
        a,b=[p['reconstructed'][arm][seed] for arm in ['A1','B1']]
        f=lambda x:fmt(x['percent'],x['whole_file_bootstrap_sd_percent'])
        pairrows.append([key,seed,f(p['MC_truth_T1']),f(a['C4']),f(b['C4']),f(b['C4_minus_T1_points'])])
pages.append([
    H("22. Все пары потенциала симметрии"),
    T(["U_high/U_low (МэВ)","Seed","T1 (%)","A1 C4 (%)","B1 C4 (%)","B1 C4−T1 (п.п.)"],pairrows,[.17,.09,.18,.18,.18,.20]),
    P("Это processed-span T1 той же ladder, не full-primary headline статьи. Все ошибки —400 saved joint file-bootstrap SD; seed показан отдельно. Для90/18 в каждой replica вычислено100×[(1+Δ90/0/100)/(1+Δ18/0/100)−1], проценты не вычитались. Три пары алгебраически зависимы, не являются тремя независимыми подтверждениями."),
    F("17_all_potential_pairs", "Рис. 19. T1 MC truth, A1/B1 reconstructed contrast и paired C4−T1. Вертикальные масштабы панелей различаются. Пара18/0 не разрешена; меньший дефицит90/18 не доказывает замыкания во всех seeds."),
])

figure_page("23. Необходимость MC-label floor: новый census", "18_label_floor_scan",
    "Рис. 20. Только S286, неизменные observed graphs, без model calls. Grid off/0,1/0,2/0,3/0,4/0,5ГэВ зафиксирован до расчёта. Native0,3 labels/float32 targets совпали точно на всех286файлах. Бары —SD400 potential×part group draws.",
    ["На1 798 225 кластерах positives322058→327538: снятие energy clause добавляет5480, или1,70% к native positives, около0,305п.п. ко всем candidates. Это новая разметка наблюдаемой задачи, не новая оценка ROC-AUC или detector efficiency. No-floor модели пока не обучались."])

figure_page("24. Какие мягкие объекты исключались", "19_soft_origins",
    "Рис. 21. S286:3500 direct-primary-only и1980 nonprimary-only по qualifying-хитам;4087 добавленных singletons. Все5480 targets строго<0,3ГэВ:859/1726/2895 в bins[0;0,1)/[0,1;0,2)/[0,2;0,3). Бары —shared group SD.",
    ["Only относится к qualifying-хитам, не чистоте всей компоненты. Nonprimary не означает автоматически фон; полной genealogy нет. Counts кластеров не равны числу независимо реконструированных нейтронов. Наблюдаемые hits≥35нс присутствуют, так что условие аналитической избыточности floor при универсальном t<35нс здесь не выполнено.",
     "На S при fixed predictions/cut C1/C2 и C4 в нынешних Rₙ bands не меняются; C3 может меняться из-за диагностического relabelling migrated soft объектов. Прямого исправления C1−T2 gap снятием floor не показано. Знак и размер влияния переобучения остаются неизвестными."])

tails=[[r['arm'],r['seed'],r['MC_positive'],r['selected_positive_C2_C3'],r['selected_all_C4'],r['pred_ge_6p5'],r['C3_ge_6p5'],r['C4_ge_6p5']] for r in PLOT['tails']]
pages.append([
    H("25. Интегралы спектров и хвосты"),
    T(["Arm","Seed","All MC+","C2=C3 total","C4 total","All-MC+ Epred≥6,5 ГэВ","C3≥6,5 ГэВ","C4≥6,5 ГэВ"],tails,[.09,.09,.14,.17,.15,.14,.11,.11]),
    P("Counts точны для конечной accepted test population; это не population estimates с binomial ошибкой. В этих сохранённых A1/B0/B1 outputs отрицательных reconstructed energies и positive predictions<0,1ГэВ нет. Они всё равно присутствуют как отдельные underflow categories в export. Истинные native positives ниже0,3ГэВ отсутствуют по определению label, а не по утверждению о физическом спектре."),
    P("При полном диапазоне C2 и C3 имеют одинаковое число объектов; таблица это проверяет для каждого arm/seed. C4 больше из-за native negatives. Нормировка спектров не перераспределяет пропущенный хвост обратно в видимую область. Counts.npz хранит все negative/finite/overflow bins и per-file matrices; data.json привязывает каждый прочитанный CSV к accepted SHA."),
])

pages.append([
    H("26. Подбор энергетических границ: сделано и не сделано"),
    P("Score cuts подобраны на validation по указанному правилу; e_cls/e_en и calibration также зафиксированы до test. Label floor0,3 и bands[0,3;1)/≥2ГэВ не являются результатом оптимизации Rₙ. Дополнительная reference с finite high cap3ГэВ относится к будущему скану; текущий опубликованный здесь числитель открыт вверх."),
    P("Аналитическая оценка не выводит0,3ГэВ из ToF resolution. При7м окно35нс дало бы Tmin≈0,322ГэВ; observed path такого универсального окна не задаёт. Эти кинематические оценки не устанавливают full efficiency или фон детектора. Скан148окон не запускался: требуются exact uncensored primary counters и заранее фиксированное правило выбора с возможностью «ни одно окно не подходит»."),
    P("Следующий зарегистрированный опыт — один no-floor arm с теми же graph/scaler/folds/recipe/seeds, всего30эпох, максимум2одновременных задач. План15+15эпох с coherent resume. Девять baseline не повторяются; ожидаемая стоимость27–28GPU·ч fit, около30–33GPU·ч с validation/I/O. Подготовка adapters и конкретный preflight ещё не равны запуску обучения."),
    P("Все arms будут оцениваться против одинаковых reference0,3 и disabled. Primary сравнение энергии — raw на common positives и отдельно added soft support; сначала frozen baseline cuts. Иначе изменение label, operating point и calibration нельзя было бы отделить. F282 (282 truth-bound группы нечётных частей1/3/5) уже открывались baseline и не являются свежим holdout; для настройки нового выбора используются только S286."),
    P("Раздельные модели по потенциалу сейчас не обучаются; U_sym не служит условием выбора реконструктора. DCM сохраняется transfer-контролем; joint DCM/SMASH или pretrain→finetune требуют отдельной абляции. Нужны новая production/генератор и LOPO, чтобы говорить об обобщении вне этой SMASH смеси."),
])

pages.append([
    H("27. Полнота данных: census и дизайн split"),
    F("22_census", "Рис. 22. Принятый input census/shared-record diagnostic. Дополнительные exports не означают независимые события. Counts описывают конечный инвентарь; статистическая ошибка к числу записанных файлов не приписывается."),
    F("23_split_precision", "Рис. 23. Принятая до обучения проекция counting precision против доли test. Это обоснование split trade-off, не измеренная uncertainty обученной сети и не подбор split по лучшему ROC-AUC."),
])

pages.append([
    H("28. Воспроизводимость и границы вывода"),
    P("Этот отчёт читает только сохранённые accepted predictions, complete/history/threshold/calibration artefacts и certified primary spectra. Проверены5112 CSV для A1/B0/B1, candidate keys/truth и per-fold counts воспроизвелись. Никаких inference, raw-event reduction, новых fit, threshold choice или training jobs отчёт не запускает. Дополнительные unit tests не выполнялись по прямому указанию пользователя; арифметика и SHA assertions остаются частью export."),
    P("Ключевые источники: results/smash_train_20261028/results_table.json; analysis/summary.json; heldout_rn/summary.json и per_file_counts.json; train_4466016/task_*/complete.json; evaluate_*/extracted/evaluation/predictions; certified aggregate4453896 spectra; reports/status_20261103_support/all_pair_contrasts.json; reports/energy_label_results_20261103/. Input pins, file hashes, figure manifest, per-file histogram counts и PDF validation находятся рядом с этим отчётом."),
    P("Не установлены: независимый generator/production holdout; physical unfolded spectrum; neutron-track response; эффективность к всем generated neutrons; корректность proton magnetic acceptance; centrality dependence при B недоступном. Улучшение native ROC-AUC не заменяет проверку Rₙ. Seed SD и file SD измеряют разные источники вариации и не включают транспортные систематики."),
    P("Сохранены отрицательные результаты: незамыкание Rₙ, отсутствие разрешённого C4 улучшения B1−A1, MAE B1 хуже B0, условный характер fold explanations. Ни слабый18/0 контраст, ни90/18 не превращаются в самостоятельное подтверждение symmetry-energy measurement. Manuscript/canonical inputs и его PDF этим отчётом не меняются."),
    P("Воспроизведение: collect.py (проверенные CSV→counts), plot.py (figures), build_report.py (единые Markdown/PDF блоки). Исторические census/design figures сохранены по исходным SHA, их vector scale увеличен только для читаемости. Validation содержит проверки text/data bounds, embedded fonts, постраничный render и read-only critic review."),
])

# Keep numerical units readable in both output formats.
def spacing(s):
    s = re.sub(r"(?<=[а-яА-ЯёЁ])(?=\d)|(?<=\d)(?=[а-яА-ЯёЁ])", " ", str(s))
    s = re.sub(r"(?<=\d)(?=(?:GeV|MeV|GPU|CPU|SD|ns|ГэВ|МэВ))", " ", s)
    s = re.sub(r"\b(dropout|batch|lr|decay|gamma|Seeds|bins|SD|cut|target|purity|radius)(?=\d)", r"\1 ", s)
    return s.replace("–", "-").replace("—", "-").replace("‑", "-")

for page in pages:
    for block in page:
        for key in ("text", "caption"):
            if key in block: block[key] = spacing(block[key])
        if block["type"] == "table":
            block["head"] = [spacing(s) for s in block["head"]]
            block["rows"] = [[spacing(s) for s in row] for row in block["rows"]]

# One block source produces both the editable Markdown and the PDF.
md=[]
for page in pages:
    for b in page:
        if b['type'] in ('title','h'):
            md.append(('# ' if b['type']=='title' else '## ')+b['text'])
        elif b['type']=='p': md.append(b['text'])
        elif b['type']=='table':
            md.append('| '+' | '.join(b['head'])+' |\n| '+' | '.join(['---']*len(b['head']))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in b['rows']))
        elif b['type']=='figure': md.append(f"![{b['caption']}]({HERE/'figures'/(b['name']+'.png')})\n\n{b['caption']}")
        md.append('')
MD.write_text('\n'.join(md).rstrip()+'\n')

fontdir=Path('/opt/homebrew/Caskroom/miniconda/base/envs/pyg/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf')
for name,file in [('Report','DejaVuSans.ttf'),('Report-Bold','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='Report-Bold',italic='Report',boldItalic='Report-Bold')
ink=colors.HexColor('#173e65')
styles={
 'p':ParagraphStyle('body',fontName='Report',fontSize=9.1,leading=12.6,spaceAfter=9),
 'h':ParagraphStyle('head',fontName='Report-Bold',fontSize=15,leading=19,spaceAfter=12,textColor=ink),
 'title':ParagraphStyle('title',fontName='Report-Bold',fontSize=25,leading=30,spaceAfter=16,textColor=ink),
 'caption':ParagraphStyle('caption',fontName='Report',fontSize=8,leading=10.7,spaceAfter=10,textColor=colors.HexColor('#405161')),
 'cell':ParagraphStyle('cell',fontName='Report',fontSize=7.6,leading=10),
 'th':ParagraphStyle('th',fontName='Report-Bold',fontSize=7.6,leading=10,textColor=ink),
}


def para(t,kind='p'): return Paragraph(html.escape(str(t)),styles[kind])
placements=[]


class Vector(Flowable):
    def __init__(self,path):
        super().__init__();self.path=path
        box=PdfReader(path).pages[0].mediabox
        w,h=float(box.width),float(box.height)
        # Two archived design figures were drawn for a small manuscript column;
        # enlarge their vectors, rather than shrink labels in the report.
        self.scale=1.55 if path.stem in ('22_census','23_split_precision') else 1.
        self.width,self.height=w*self.scale,h*self.scale
    def draw(self):
        x,y=self.canv.absolutePosition(0,0)
        placements.append({'page':self.canv.getPageNumber()-1,'x':x,'y':y,'width':self.width,'height':self.height,'scale':self.scale,'file':str(self.path)})


def frame(canvas,doc):
    canvas.saveState();canvas.setFont('Report',8);canvas.setFillColor(colors.HexColor('#526272'))
    canvas.drawString(43,A4[1]-26,'HGND | Качество реконструкции на исправленном SMASH | 03.11.2026')
    canvas.setStrokeColor(colors.HexColor('#ccd6df'));canvas.line(43,A4[1]-34,A4[0]-43,A4[1]-34)
    canvas.drawString(43,24,'Внутренний отчёт | MC truth и reconstructed величины обозначены отдельно')
    canvas.drawRightString(A4[0]-43,24,str(doc.page));canvas.restoreState()


base=HERE/'report_base.pdf'
doc=BaseDocTemplate(str(base),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=49,bottomMargin=42,invariant=1)
f=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='normal')
doc.addPageTemplates(PageTemplate(id='normal',frames=f,onPage=frame))
story=[]
for i,page in enumerate(pages):
    if i: story.append(PageBreak())
    for b in page:
        kind=b['type']
        if kind in ('title','h','p'): story.append(para(b['text'],kind))
        elif kind=='table':
            data=[[para(c,'th') for c in b['head']]]+[[para(c,'cell') for c in row] for row in b['rows']]
            weights=b['widths'] or [1/len(b['head'])]*len(b['head'])
            assert abs(sum(weights)-1)<1e-6
            t=Table(data,colWidths=[v*doc.width for v in weights],repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Report'),('FONTSIZE',(0,0),(-1,-1),7.6)]))
            t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f6')),('LINEBELOW',(0,0),(-1,0),.6,ink),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d7dfe6')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([t,Spacer(1,10)])
        elif kind=='figure': story.extend([Vector(HERE/'figures'/(b['name']+'.pdf')),Spacer(1,5),para(b['caption'],'caption')])


def cv(*a,**kw):
    kw['initialFontName']='Report';return Canvas(*a,**kw)


doc.build(story,canvasmaker=cv)
reader=PdfReader(base);writer=PdfWriter()
for p in reader.pages:writer.add_page(p)
for p in placements:
    assert p['x']>=42 and p['y']>=41 and p['x']+p['width']<=A4[0]-42 and p['y']+p['height']<=A4[1]-48,p
    src=PdfReader(p['file']).pages[0]
    writer.pages[p['page']].merge_transformed_page(src,Transformation().scale(p['scale']).translate(p['x'],p['y']),expand=False)
writer.add_metadata({'/Title':'HGND: полный отчёт по реконструкции SMASH','/Author':'HGND internal analysis','/Subject':'Spectra, classification, energy response, threshold selection and Rn closure'})
with OUT.open('wb') as stream:writer.write(stream)
(HERE/'placements.json').write_text(json.dumps(placements,indent=2)+'\n')
base.unlink()
print('Built',len(writer.pages),'pages;',len(placements),'vector figures;',OUT)
