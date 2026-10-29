"""Build the Russian progress document and embed its figures as vector PDFs."""

from pathlib import Path
import hashlib
import html
import json
import math
import re

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D = json.loads((HERE / "data.json").read_text())
for source, expected in D['sources_sha256'].items():
    assert hashlib.sha256((REPO/source).read_bytes()).hexdigest()==expected, ('Source changed after snapshot',source)
STATUS = json.loads((HERE / "slurm_status.json").read_text())
OUT = HERE / "HGND_progress_20261021_20261029.pdf"
MD = HERE.parent / "HGND_progress_20261021_20261029.md"
FIG = HERE / "figures"
TRAIN = D["training"]
ARMS = TRAIN["classification_energy"]["arms"]
POOLED = TRAIN["rn_ladder_primary"]["pooled"]["arms"]


def pair(v, sd, factor=1, signed=False):
    v, sd = v * factor, sd * factor
    dec = 1 - math.floor(math.log10(sd))
    dec = 1 - math.floor(math.log10(round(sd, dec)))
    val = round(v, dec)
    prec = max(0, dec)
    sign = "+" if signed and val != 0 else ""
    return f"{val:{sign}.{prec}f} ± {sd:.{prec}f}"


def metric(arm, key, factor=1):
    v = ARMS[arm][key]
    return pair(v["mean"], v["seed_sd"], factor)


def c4(arm, i):
    a = POOLED[arm]
    return pair(a["C4_percent"][i], a["C4_boot_sd"][i], signed=True)


def n(v):
    return f"{v:,}".replace(",", " ")


def P(text): return {"type": "p", "text": text}
def H(text): return {"type": "h", "text": text}
def T(head, rows, widths=None): return {"type": "table", "head": head, "rows": rows, "widths": widths}
def F(name, caption): return {"type": "figure", "name": name, "caption": caption}
def C(text): return {"type": "callout", "text": text}


pages = []
pages.append([
    {"type": "title", "text": "HGND: прогресс и ход работ"},
    H("21–29 октября 2026"),
    P("Срез материалов: 29.10.2026, 14:09 МСК; Slurm проверен в 13:47:57. Внутренний технический отчёт. Источники: сохранённые результаты, протоколы, receipts и свежая read-only проверка Slurm. Новые обучение, inference и физические эксперименты для отчёта не запускались."),
    C("Главный итог: воспроизводимый сигнал жёсткости установлен на MC truth. Новое обучение на SMASH заметно улучшило классификацию и оценку cluster energy, но пока не восстановило полный отклик реконструированной жёсткости."),
    T(["Направление", "Статус на момент среза"], [
        ["MC truth, точный диапазон", "568 primary-файлов; 2 345 993 представленных события. Rₙ: +4.71 ± 0.58%; Rₙ/Rₚ: +11.59 ± 0.89% между 90 и 0 MeV. [S7]"],
        ["SMASH-обучение", "9 завершённых и принятых исходов: 3 фолда × 3 seed, по 30 эпох. Три fold-evaluation завершены и приняты. Активных задач этой серии нет. [S8, S10]"],
        ["Классификация", f"A1 (DCM): AUC {metric('A1','auc')}; B1 (SMASH): 0.9547, seed SD < 0.0001. Данные: 3 572 710 test-кластеров. [S8]"],
        ["Реконструированная жёсткость", "B1 C4: +1.82…+2.01% по seed при T1 MC truth +4.72 ± 0.57%. Парный дефицит около 2.7–2.9 процентного пункта; G2 fail. [S8]"],
        ["Статья", "Версия e6afb98: 7 проверенных страниц, 14 тестов; number-auditor проследил 132 расчётных выражения без оставшихся расхождений. Новые числа обучения SMASH в статью не внесены. [S7]"],
        ["Следующий этап", "Purity-scan отозван после NO-GO 5a49497. Вместо него сохранён draft 020e766: поправка фона по полосам при фиксированном cut; принятых результатов ещё нет. [S9, S11]"],
    ], [.27,.73]),
    H("Как читать этот документ"),
    P("MC truth использует истинные частицы/энергии симуляции. Reconstructed C4 строится из выбранных кластеров и предсказанных энергий. MC-assisted ступени нужны для локализации потерь и не являются измерением детектора. Погрешности seed SD, SE и file-bootstrap SD имеют разные смыслы; они подписаны у каждой таблицы и фигуры."),
    P("Замыкание (closure) — согласие восстановленного отклика с MC-truth эталоном в заданной точности. Seed SD отражает разброс инициализаций на общих данных; SE — ошибку их среднего; file-bootstrap SD — разброс при повторной выборке целых файлов. Они не заменяют неопределённость другой production или модели транспорта."),
    P("21 октября — начало выбранного периода отчёта, а не переименование даты ранних расчётов. Архивная записка о жёсткости имеет заголовок 29.09.2026; её исходные оценки приведены далее только как исторический ориентир. [S0]"),
])

pages.append([
    H("Хронология: от ориентиров к проверяемому результату"),
    T(["Дата", "Сделано и получено", "Последствие для следующего шага"], [
        ["21.10", "Приняты 12 native DCM обучений и оценки. AUC четырёх семейств 0.906–0.910; 28 диагностических рисунков. Первый SMASH CPU smoke 4430110 остановлен по EACCES. [S1, S2]", "Не выбирать новую архитектуру по уже открытому test. Сначала восстановить SMASH lineage и общую популяцию."],
        ["22.10", "Восстановлены первые source matches; CUDA probe 4435668 выполнил перенос frozen DCM GraphSAGE на первые файлы трёх потенциалов. AUC около 0.90; SMASH fitting не было. [S2]", "Первые файлы подтвердили работу тракта, но не измерили контраст потенциалов на полной выборке."],
        ["23.10", "Диагностика разрыва student/native; полные binding-проверки zero/default/big, учёт пустых CSV и byte aliases. Разрешённые recovery после I/O отказов сохранены. [S3, S4]", "Сопоставление по имени файла признано недостаточным; нужны точные records и единое accounting."],
        ["24.10", "Полный frozen transfer: 432 уникальных непустых detector-файла; 1 296 prediction CSV проверены. Census 4440582: DCM повторяется; SMASH primary-события точных повторов не имеют. [S3, S4]", "Student AUC ≈0.98 относится к повторяющейся random-split задаче. Generator/PRNG independence ещё не доказана."],
        ["25–26.10", "Сертифицированы counters по raw primary: 568 файлов. B-free aggregate 4452505. Ретроспективная лестница T0–C4 и paired controls; затем поправки интерпретации v10. [S5, S6]", "Удалена опора на неподтверждённый B-join; недостаток статистики отделён от смещения response."],
        ["26–28.10", "Точные counters для [0.3,1) GeV; успешный recovery zeroSpot; aggregate 4453896. Canonical import, графики, PDF и независимый аудит закрыты 28.10. [S7]", "Основной знаменатель теперь внутри заявленного нижнего ToF reach; старое E<1 остаётся отдельным вариантом."],
        ["28–29.10", "Полный detector census и content grouping; 3-fold training. Девять исходов приняты, включая 3 resume retries. Оценки 4466339/4467086/4467091 завершены 29.10. [S8, S10]", "Классификация улучшилась, но C4 не замкнулся на truth. Объём и разбиение отличаются от transfer 24.10."],
        ["29.10", "Purity-scan получил NO-GO (13:54), затем отозван. В 14:09 сохранён draft per-band background correction 020e766; принятого результата ещё нет. [S9, S11]", "Ключевой вопрос смещён к модельной/генераторной зависимости оценки фона при замороженном cut."],
    ], [.12,.47,.41]),
    P("Даты означают дату события/проверки в сохранённом журнале. Git-коммит мог появиться позднее самого расчёта: в частности, материалы 21–23 октября были сведены в репозиторий 24 октября. Это не дополнительные повторные эксперименты."),
])

native_rows=[]
for key,label in [("student_sage","GraphSAGE"),("student_dynedge","DynamicEdgeConv"),("student_spectral_sage","Spectral GraphSAGE"),("student_spectral_dynedge","Spectral DynamicEdgeConv")]:
    v=D["native_dcm"]["models"][key]
    native_rows.append([label,pair(v["roc_auc"]["mean"],v["roc_auc"]["sample_sd"]),pair(v["mae_GeV"]["mean"],v["mae_GeV"]["sample_sd"],1000)])
pages.append([
    H("DCM: что удалось воспроизвести и почему ≈0.98 не стало benchmark"),
    P("Обучение с нуля воспроизвело student-качество около 0.98 на его задаче. При переносе архитектур в native pipeline изменились split, кандидаты, признаки, timing, метка и objective. Поэтому общий разрыв не выделяет чистый эффект одного из этих факторов. [S1, S4]"),
    T(["Native модель", "ROC-AUC ↑", "Energy MAE (MeV) ↓"],native_rows,[.47,.27,.26]),
    P("Mean ± SD трёх training seeds; reused DCM evaluation 988976: 222 251 кластер, 36 713 MC-positive, 91 081 представленный event identifier. Target энергии — максимум qualifying MC hit в кластере, а не энергия отдельного neutron track. [S1]"),
    F("01_dcm_census","Рис. 1. Слева — точные census counts без статистической ошибки: represented и distinct retained-hit contents по source-directory роли. Справа — AUC сохранённых student моделей/seed на all/seen/unseen кандидатах; точки показывают variation моделей, не confidence interval. 23 880 из 36 343 test-кандидатов повторяют training inputs/edges. [S4]"),
    P("Из 197 027 DCM event identifiers имеется 79 324 различных retained-hit contents; training — 2 209 из 64 138. Student pickle точно совпадает с multiset объединения трёх экспортов. AUC на seen-кандидатах 0.994–0.997, на unseen 0.934–0.947. Точных полных retained-hit повторов между native source directories нет; generator collision lineage этим не установлена."),
    P("Исключение одиночных кандидатов на фиксированных DCM весах даёт только +0.0057 ± 0.0031 AUC (парный mean ± seed SD). Это изменение популяции, не retraining. Доли причин всего AUC gap не измерены."),
])

census_rows=[]
for tag,label in [("zeroSpot","0"),("defaultSpot","18"),("bigSpot","90")]:
    v=D['census'][tag]
    census_rows.append([label,n(v['membership']['pairs']),n(v['membership']['nonempty']),n(v['counts']['contents']),n(v['counts']['groups']),n(v['counts']['truth_bound_groups'])])
pages.append([
    H("SMASH: binding, повторения и допустимое разбиение"),
    P("22 октября first-file перенос дал AUC около 0.90. Полный перенос, принятый 24 октября, использовал 195/186/51 непустой файл при 0/18/90 MeV. В ledger 568 исходных members отдельно учтены 100 header-only pairs и 36 exact aliases bigSpot. Сравнивать эти размеры с позднейшим full-census split как одну и ту же test-популяцию нельзя. [S2, S3]"),
    T(["S_pot (MeV)","Raw pairs","Непустых","Distinct byte contents","Групп после event grouping","Truth-bound групп"],census_rows,[.13,.15,.16,.19,.20,.17]),
    P("Таблица — полный census 28 октября, точные counts, без направления «лучше» и без статистической ошибки. Всего 584 byte-distinct contents объединены в 578 групп по content/event/direct-primary связям; 568 групп связаны с truth. Дополнительные detector exports не дают автоматически новые независимые события. [S8]"),
    F("02_crossfit","Рис. 2. Принятая схема трёх фолдов. Для каждого потенциала группы распределены по шести частям через фиксированное hash ranking. Test — две части, validation — следующая, train — остальные три. Unbound группы в test не входят. Это схема настроек протокола, а не измерение. [S8]"),
    P("На SMASH primary-уровне census 24 октября нашёл 2 345 993 различных представленных события без точных повторов, включая identity без vertex. На detector-уровне последующий census выявил множество копий и усечённых пересечений; эти два результата относятся к разным представлениям данных и не противоречат друг другу. [S4, S8]"),
    P("Каждый из 568 truth-bound test-файлов оценён только моделями, которые не использовали его для своего обучения или validation. Но frozen transfer уже открывал эти данные, поэтому это retrospective cross-fitting, а не новый blind holdout. Доказательство независимых PRNG streams и свежая production остаются отдельными задачами."),
])

pages.append([
    H("MC truth: определение жёсткости и физическая мотивация"),
    P("Основная величина: Rₐ = Nₐ(Ekin ≥ 2 GeV) / Nₐ(0.3 ≤ Ekin < 1 GeV), a = n,p; D = Rₙ/Rₚ. Контраст: 100 × [X(90)/X(0) − 1] в процентах. Граница 0.3 GeV принята человеком после v10; 1 и 2 GeV — унаследованные, не оптимизированные границы. [S7]"),
    F("03_truth_spectra","Рис. 3. MC-truth спектры n/p и отношения 90/0 в неизменном прямоугольном окне. Рисунок взят без изменения из проверенной canonical-сборки статьи. Полосы показывают знаменатель [0.3,1) и числитель ≥2 GeV; ToF bar — диапазон reference detector study, не измеренная efficiency. Погрешности спектров и ratios: 400 whole-file resamples. [S7]"),
    P("Около 36% нейтронов старого знаменателя Ekin<1 GeV лежат ниже 0.3 GeV. Поэтому старое определение оставлено отдельным вариантом. Высокая полоса всё ещё выходит за приблизительный ToF design range: перенос truth-контраста в измерение требует response calculation."),
    P("Окно — проекция луча из production vertex на z = 718.42 cm, −146.61 ≤ x ≤ −106.67 cm, |y| ≤ 77.34 cm. Оно не включает gaps и inactive regions двух модулей. Для протонов не моделируются магнитный транспорт и tracking efficiency; D является MC-truth диагностикой общей области эмиссии."),
    P("Ранний архивный ориентир: Rₙ +4.52 ± 0.45% и D +10.48 ± 0.76% при B-reweighting. У доступных raw exports B = −1; точный B-join не подтверждён. Эти архивные числа не используются как текущая основная оценка. Новая цепочка напрямую связывает raw bytes, counters, bootstrap и canonical inputs. [S0, S5]"),
])

pages.append([
    H("MC truth: сертифицированный результат и его границы"),
    F("04_truth_contrasts","Рис. 4. MC-truth контрасты при 0/18/90 MeV: точный основной диапазон и старый вариант. Рисунок из canonical JSON статьи; исходная выборка 568 файлов, 2 345 993 события. Бары — большая из file-bootstrap SD и event-clustered compound-Poisson SD. Ноль задаёт reference точно. Старый вариант в yield-панели не показан. [S7]"),
    T(["Величина", "18/0 (%)", "90/0 (%)"],[
        ["Rₙ", "+0.70 ± 0.57", "+4.71 ± 0.58"],
        ["Rₚ", "−1.87 ± 0.58", "−6.16 ± 0.56"],
        ["Rₙ/Rₚ", "+2.61 ± 0.79", "+11.59 ± 0.89"],
        ["n/p, Ekin ≥ 0.3 GeV", "+0.40 ± 0.29", "+0.24 ± 0.30"],
    ],[.44,.28,.28]),
    P("На полной выборке neutron hardness растёт, proton hardness падает; в двойном отношении контрасты складываются. Neutron 18/0 и интегральное n/p при этой точности не разрешены. Это статистическая картина данной симуляции без transport-model systematics и без измеренного detector response."),
    P("Дискретный MC-truth proxy — midrapidity n+p строго вне прямоугольника, |y−Ybeam/2|<0.5, Ybeam/2 = 1.0393. Совместные пять rank-классов и matching weights перестроены в каждой из 400 file-реплик. Центральные сдвиги endpoint-контрастов от такого matching <0.01 процентного пункта; это не проверка неизвестного B-распределения и не экспериментальная centrality."),
    P("Файловые и event-level ошибки измерены отдельно. Например, у Rₙ endpoint file SD = 0.53, event SD = 0.58 процентного пункта; в headline использовано 0.58. Для старого E<1 варианта текущая max-error политика даёт +4.39 ± 0.53%, а не прежнюю file-only ошибку 0.46. Это смена оценки ошибки, не новый прирост сигнала. [S5, S7]"),
])

pages.append([
    H("26 октября: первая полная ретроспективная лестница Rₙ"),
    P("Замороженные DCM weights/cuts применены к reused corrected-SMASH: 195/186/51 detector-файл. Ступени: T0 — доступные primary events; T1 — processed span; T2 — events с кандидатами; C1 — native-positive clusters с MC energy; C2 — score selection; C3 — predicted energy на этих positives; C4 — все выбранные clusters, включая фон. [S6]"),
    F("05_retrospective","Рис. 5. Историческая лестница 26.10 на общем subset; старый знаменатель E<1 GeV, числитель ≥2 GeV. Не подменяет основной [0.3,1) результат. Точки по seed, бары: 400 shared whole-file bootstrap SD. MC truth/MC-assisted T0–C3; reconstructed только C4. [S6]"),
    P("T1 = +5.08 ± 0.68%; C4 = +3.1 ± 1.4%, +2.8 ± 1.7%, +4.3 ± 1.6% по seed 42/123/456. Знак сохранился; G2 closure был not resolvable, поскольку парные ошибки превышали требуемую точность 25% от truth. Такой исход не доказывает closure."),
    P("Поправка v10: центральное C4/T1 нельзя интерпретировать без joint intervals. 68% bootstrap интервалы составляют [0.33,0.87], [0.18,0.85], [0.51,1.14]. Только background step seed 42 превышал 2 paired SD; механизм по всем seed не был установлен. G6 прошёл лишь с запасом 0.04 процентного пункта."),
    P("Инъекция +5% только в true-hard positives проверяла numerator signal path при фиксированном фоне. Она не проверяла весь спектр, low-band migration и связанный фон. Прогноз числа новых файлов предполагал нулевой bias; увеличение статистики само по себе не исправляет наблюдённое смещение response."),
])

contrast=TRAIN['classification_energy']['B1_minus_A1']
pages.append([
    H("28–29 октября: новое обучение на SMASH"),
    P("GraphSAGE H128/L4, 30 эпох, три фолда и seeds 42/123/456. Сигнал и graph view общие: native label, radius 3.6 cells, time window 1.0 ns, 10 features. A1 — frozen DCM с cut/calibration на fold validation; B1 — SMASH-trained с тем же validation tuning. B использует fold-train scaler, A — свой frozen DCM transform. Это сравнение двух адаптированных pipeline recipes, а не изоляция одного изменения весов. [S8]"),
    F("06_training_metrics","Рис. 6. На одной cross-fitted test population сравниваются A1 и B1. Точки: три training seeds; у каждого сначала усреднены per-fold metrics. Бары справа от точек — mean ± seed SD. ROC вычислялся per fold, без pooling несопоставимых score scales. Ни seed SD, ни SE не включают between-production uncertainty. [S8]"),
    T(["Метрика", "A1: DCM", "B1: SMASH", "B1 − A1 (paired SE)"],[
        ["ROC-AUC ↑", metric('A1','auc'), "0.9547; SD <0.0001",pair(contrast['auc']['mean'],contrast['auc']['se_of_mean'],signed=True)],
        ["Efficiency (%) ↑",metric('A1','efficiency',100),metric('B1','efficiency',100),pair(contrast['efficiency']['mean'],contrast['efficiency']['se_of_mean'],100,True)],
        ["Energy MAE (MeV) ↓",metric('A1','mae_GeV',1000),metric('B1','mae_GeV',1000),pair(contrast['mae_GeV']['mean'],contrast['mae_GeV']['se_of_mean'],1000,True)],
    ],[.28,.24,.24,.24]),
    P("Test: 568 файлов; 3 572 710 cluster candidates, 640 277 MC-positive. Cut выбирался только на validation для target count purity 0.70. Efficiency относится к MC-positive candidate clusters, не к generated neutrons. MAE сравнивает predicted energy с native max-hit target без score cut."),
    P("Парная SE описывает разброс трёх seed labels. Общая file-sampling ошибка этих сравнений и неизвестная систематика production в эту SE не входят; вывод о новой независимой production по ней не делается."),
    P("Прогнозы до обучения были осторожнее результата: B AUC ожидался 0.915–0.95, MAE 120–160 MeV. Classification head выбрал эпохи 21–29; два запуска достигли минимума на последней эпохе и отмечены budget-limited. Продления бюджета по test не было. [S8]"),
])

pages.append([
    H("Новая реконструкция: высокий AUC ещё не даёт closure"),
    F("07_closure","Рис. 7. Сверху: pooled out-of-fold C4 двух arms с T1 на той же processed population. Снизу: парные потери ступеней B1 по seed. Диапазон [0.3,1) / ≥2 GeV. Бары — SD 400 общих whole-file bootstrap draws; это статистика файлов, а не scatter обучения. C1–C3 используют MC-assisted определения. [S8]"),
    T(["Reconstructed C4 (%)", "seed 42", "seed 123", "seed 456"],[
        ["A1: DCM",*[c4('A1',i) for i in range(3)]],
        ["B1: SMASH",*[c4('B1',i) for i in range(3)]],
    ],[.28,.24,.24,.24]),
    P("MC truth T1 = +4.72 ± 0.57%; небольшой сдвиг относительно manuscript +4.71 ± 0.58% связан с processed-span population и file-only SD этой лестницы. Эти определения ошибок и популяций не переименованы друг в друга."),
    P("У B1 парные C4−T1 = −2.77 ± 0.93, −2.70 ± 0.94, −2.90 ± 0.91 процентного пункта: G2 fail по всем seed. Парная разность B1−A1 на C4: +0.80 ± 0.88, +0.12 ± 1.00, −0.86 ± 1.03 — не разрешена. Центральные retention 39–43% у B1 описательны; precision видна по самим C4 и paired gaps."),
    P("B1 снижает зависимость от seed: размах C4 0.20 процентного пункта против средней file SD 0.87 (G6 pass). У A1 — 1.53 против 1.02 (G6 fail). Самый большой центральный отдельный loss B1 — пропущенный фон, −1.11…−1.31 пункта, затем native target и selection. Это приоритет диагностики, а не доказательство единственного механизма по всем фолдам."),
])

pages.append([
    H("Что ещё ограничивает response и следующий gate"),
    F("08_bias_folds","Рис. 8. Слева: potential-dependent bias на MC-positive clusters в [1.75,3) GeV; три seed-точки на arm, file-bootstrap SD. Парная B1−A1 bootstrap для Δbias не вычислена, поэтому значимость улучшения по этой картинке не заявляется. Справа: B1 C4 по отдельным test folds и seed, file-bootstrap SD. [S8]"),
    P("Δbias(90−0) у A1 около −3.6 MeV при file SD 1.8–2.0 MeV, у B1 около −0.6 MeV при SD ≈0.9 MeV. В пределах этого окна средняя truth-energy разность +3.8 MeV. Меньший differential bias согласуется с лучшим energy response, но не заменяет замыкание интегрального Rₙ."),
    P("В fold 0 reconstructed B1 контраст около −0.57…+0.11%, тогда как folds 1/2 сохраняют положительный знак. Pooled результат скрывает эту variation. Стабильность относительно seed и стабильность относительно состава файлов — разные проверки."),
    P("На каждом этапе качество классификации, условный energy target и физический response проверяются отдельно. Ни малая seed variation, ни точная median calibration не доказывают пригодность Rₙ на новых данных. Для проверки обобщения по потенциалу нужен отдельный leave-one-potential-out control, по production — новые неиспользованные исходные события."),
    P("Приоритет следующей диагностики следует из ступеней, но не предрешает решение: low-band background, связь native label с primary neutron, затем selection и energy response. Полный response должен учитывать zero-candidate events, merging/splitting и энерго-зависимую эффективность. Efficiency среди cluster candidates этого не заменяет."),
])

pages.append([
    H("29 октября: следующий gate остановлен на preflight"),
    C("Purity-scan draft не принят. Протокол 61659b5 сохранён вместе с записью NO-GO 5a49497; его прогнозы не переписаны под результат проверки. Primary contrast scan на final F не выполнен. [S9]"),
    F('09_purity_preflight','Рис. 9. Ошибка разности C4(p)−C4(0.70) на selection S по preflight-записи. A1/B1 — paired SD seed-mean estimate, не сам контраст. Серая полоса — ранее ожидаемый сдвиг +0.3…+1.2 п.п., не confidence band. Ошибки самих SD в записи не приведены; F и альтернативные ΔC4 не использовались. [S9]'),
    P("Draft предлагал count-purity targets 0.60–0.95 с шагом 0.05, selection S = 286 файлов и final F = 282. Выбор p* опирался на парное one-SE правило. Проверка нашла ложную предпосылку: file sampling не сокращается достаточно между purity targets, потому что меняется сама низкоэнергетическая популяция."),
    P("По committed preflight на S, B1 теряет около 74% low-band clusters между targets 0.70 и 0.90. При target 0.70 purity по полосам около 0.56 в low и 0.81 в high; при 0.90 — около 0.80 и 0.93. Эти числа — диагностическая запись проверки без опубликованной sampling error, а не результат принятого full-scan опыта."),
    P("У B1 paired SD составляет 0.85/1.17/1.77/3.32 пункта при p=0.80/0.85/0.90/0.95; у A1 ещё больше. При прогнозе улучшения +0.3…+1.2 такой дизайн маломощен: review оценивает около 10–30% мощности даже на всех 568 файлах. Неудачный null не позволил бы заключить, что cut не помогает."),
    P("После NO-GO purity-scan отозван. Новый committed draft 020e766 (14:09) предлагает per-band background correction при фиксированном 0.70 cut: отдельно фон на событие и фон на hard-signal cluster, с общей validation-настройкой и проверкой модельной/генераторной зависимости. Корректированный Rₙ — вторичный результат, не обещание closure. В этом срезе принятого результата нового опыта ещё нет. [S11]"),
])

pages.append([
    H("Исполнение, отказы и статус доказательств"),
    T(["Этап", "Сохранённые jobs", "Итог"],[
        ["Первый SMASH audit", "4430110", "FAILED: EACCES; первичный отказ и вторичный отказ записи сохранены."],
        ["First-file transfer", "4435628 / 4435668", "CPU audit и CUDA probe завершены; это engineering scope."],
        ["Полный transfer", "4437503–4437505; 4437674–4437676", "Аудиты и GPU scoring приняты; 1 296 prediction CSV проверены."],
        ["Content census", "4440572 / 4440582", "Smoke и полный census приняты: primary и DCM identities."],
        ["Точный truth reach", "4453681; 4453689; 4453846; 4453896", "Первый zeroSpot упал; разрешённый recovery завершён. В aggregate входят только полные сертификаты."],
        ["Новый detector census", "4464520 / 4464531", "После разрешённой обработки EACCES три tag-проверки прошли. Code/input hashes закреплены."],
        ["SMASH training", "4466016 + 4466738/4467054/4467055", "6 прямых completion + 3 принятых resume outcomes. Старые cancelled executions не названы завершёнными."],
        ["Fold evaluations", "4466339 / 4467086 / 4467091", "COMPLETED 0:0; 192/189/187 test files приняты. Конец последней оценки 29.10 в 12:01 МСК."],
    ],[.23,.30,.47]),
    P("Свежая проверка squeue/sacct в 13:47:57 МСК подтверждает terminal состояния перечисленных HGND jobs; активных задач этой серии нет. Статусы прочих задач аккаунта не оценивались. Архивы заново не скачивались: статус отчёта опирается на Slurm и сохранённые acceptance/hash records. [S10]"),
    P("По итоговому журналу кампании обучение и evaluation заняли около 32.0 GPU·h: 19.4 завершённых training executions, 9.4 отменённых, 0.85 resume retries и 2.35 evaluation. Это учёт исполнения, а не новый расчёт physics sensitivity. [S8]"),
    P("Наблюдались EACCES на разных узлах при доступных позже файлах. Точная инфраструктурная причина не доказана; RA-09 остаётся open. В новой кампании retries ограничены, каждый логируется и не допускает незаметной подмены байтов. Три training executions были CANCELLED by 0 после примерно 3.1 часа; разрешённые resumes использовали проверенные зеркальные checkpoints. Состояние Slurm имеет приоритет над exit code 0:0."),
    H("Какие выводы были пересмотрены"),
    P("≈0.98 больше не трактуется как качество на новых collision contents; «полная acceptance» заменена прямоугольным analysis window; чистая причинная доля scaler в старом сравнении не выделена; not-resolvable closure не называется успешным; numerator-only injection не подтверждает весь spectral response; прогноз большего production не лечит фиксированный bias. История сохранена рядом с новыми трактовками. [S4, S6, S7]"),
])

source_rows=[
    ["S0", "hgnd_np_hardness_update.md", "Архивная записка, заголовок 29.09; исторический ориентир, содержит superseded claims."],
    ["S1", "student_pipeline_results_20261021.md", "12 native DCM runs; JSON evaluation_summary_20261021, диагностика."],
    ["S2", "smash_audit_status_20261021.md; smash_staged_transfer_results_20261022.md", "Первый отказ, paired relabelling и first-file transfer."],
    ["S3", "smash_full_transfer_results_20261024.md; smash_full_transfer_status_20261023.md", "Полная старая transfer population; verified metrics и source binding."],
    ["S4", "roc_auc_gap_20261023.md; v9_actions_20261024.md", "DCM census, student lineage/leakage; content_census_20261024 JSON."],
    ["S5", "v9_actions_20261024.md: B-free aggregation 4452505", "Raw counters, 568 файлов, B ignored, file-bootstrap errors."],
    ["S6", "reco_rn_retrospective_results_20261026.md", "Frozen protocol 5abd90e; summary/replicas; v10 addendum обязателен."],
    ["S7", "v10_actions_20261026.md; HGNDPaper/data/canonical_results.json", "4453896; validation_v10_20261028.json; number_audit_v10_20261028.md; paper e6afb98."],
    ["S8", "smash_train_protocol_20261028.md; smash_train_results_20261029.md", "results/smash_train_20261028/results_table.json, underlying analysis/heldout/census/split/acceptance."],
    ["S9", "rn_purity_scan_protocol_20261029.md", "Draft + NO-GO preflight 5a49497; selection-only ошибки, не primary scan result."],
    ["S10", "progress_20261021_20261029/slurm_status.json", "Read-only queue/accounting snapshot 29.10, 13:47:57 МСК."],
    ["S11", "020e766: rn_background_correction_protocol_20261029.md", "Новый draft; в пакете хранится точная копия committed blob, а не меняющийся рабочий файл."],
]
pages.append([
    H("Открытые задачи и карта источников"),
    P("1. Следовать отдельному согласованному gate фоновой поправки: проверить coefficients и их model/generator dependence до интерпретации corrected Rₙ; draft 020e766 не является уже полученным результатом. 2. По paired ladder решить, нужны ли target/selection corrections. 3. Подтвердить neutron one-to-one matching и общий energy/multiplicity response. 4. Проверить устойчивость по production/fold и leave-one-potential-out до заявления обобщения."),
    P("Внешние gates: реальные SMASH/DCM/transport run cards и версии; coefficients и отображение S_pot на E_sym; b sampling, generator seeds и export policy; точные active-module coordinates и proton magnetic response; immutable data/weight deposit, license/DOI; coauthor scope/authorship/funding approval. При отсутствии B физическая centrality-зависимость не измерена."),
    T(["ID", "Где искать", "Для какого утверждения"],source_rows,[.07,.47,.46]),
    P("Имена отчётов в колонке «Где искать» относительны HGNDRecoGNN/reports, пути results — корню code repository. Все 31 числовой/документальный источник зафиксированы SHA256 в соседнем data.json; отдельный validation.json хранит хэши отчёта, графиков, snapshot и проверки PDF. Report figures читают этот snapshot, а не повторно оценивают модели."),
    P("Документ обновляет статус работы, но не меняет manuscript scope A. Новые training/closure результаты остаются в исследовательском отчёте, пока отдельно не принято решение об их включении в статью. Публикация, отправка сообщений и новые jobs этим документом не выполняются."),
])

# Markdown and PDF share the same block content.
md=[]
for page in pages:
    for block in page:
        kind=block['type']
        if kind in ['title','h']:
            md.append(('# ' if kind=='title' else '## ')+block['text'])
        elif kind in ['p','callout']:
            md.append(('> ' if kind=='callout' else '')+block['text'])
        elif kind=='table':
            md.append('| '+' | '.join(block['head'])+' |\n| '+' | '.join(['---']*len(block['head']))+' |\n'+'\n'.join('| '+' | '.join(str(c).replace('|','/') for c in row)+' |' for row in block['rows']))
        elif kind=='figure':
            md.append(f"![{block['caption']}]({FIG / (block['name']+'.png')})\n\n{block['caption']}")
        md.append('')
    md.append('')
md.append('## Прямые ссылки на первичные записи\n')
for path in D['sources_sha256']:
    if path.endswith('.md'):
        md.append(f'- [{Path(path).name}]({(REPO/path).resolve()})')
MD.write_text('\n'.join(md)+'\n')

font_dir=Path('/opt/homebrew/Caskroom/miniconda/base/envs/pyg/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf')
for name,file in [('Report','DejaVuSans.ttf'),('Report-Bold','DejaVuSans-Bold.ttf'),('Report-Italic','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(font_dir/file)))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='Report-Bold',italic='Report-Italic',boldItalic='Report-Bold')
ink=colors.HexColor('#16364c');muted=colors.HexColor('#51616d')
styles={
    'p':ParagraphStyle('body',fontName='Report',fontSize=9.35,leading=13.0,spaceAfter=8,textColor=colors.HexColor('#202a31'),splitLongWords=True),
    'h':ParagraphStyle('heading',fontName='Report-Bold',fontSize=15.5,leading=20,spaceAfter=13,spaceBefore=2,textColor=ink),
    'title':ParagraphStyle('title',fontName='Report-Bold',fontSize=25,leading=31,spaceAfter=13,textColor=ink),
    'caption':ParagraphStyle('caption',fontName='Report',fontSize=8.0,leading=11,spaceAfter=11,textColor=muted),
    'cell':ParagraphStyle('cell',fontName='Report',fontSize=8.4,leading=11.1,textColor=colors.HexColor('#202a31')),
    'th':ParagraphStyle('th',fontName='Report-Bold',fontSize=8.4,leading=11.1,textColor=ink),
    'callout':ParagraphStyle('callout',fontName='Report-Bold',fontSize=10,leading=14,spaceAfter=12,textColor=ink,borderPadding=9,backColor=colors.HexColor('#edf3f6')),
}

def para(text,style='p'):
    return Paragraph(html.escape(str(text)),styles[style])

placements=[]
class VectorFigure(Flowable):
    def __init__(self,path):
        super().__init__();self.path=path
        p=PdfReader(path).pages[0]
        self.width=float(p.mediabox.width);self.height=float(p.mediabox.height)
    def draw(self):
        x,y=self.canv.absolutePosition(0,0)
        placements.append({'page':self.canv.getPageNumber()-1,'x':x,'y':y,'width':self.width,'height':self.height,'file':str(self.path)})

def page_frame(canvas,doc):
    canvas.saveState()
    canvas.setFont('Report',8)
    canvas.setFillColor(muted)
    canvas.drawString(44,A4[1]-28,'HGND  |  Прогресс 21–29 октября 2026  |  Внутренний отчёт')
    canvas.setStrokeColor(colors.HexColor('#cbd7df'));canvas.setLineWidth(.5)
    canvas.line(44,A4[1]-35,A4[0]-44,A4[1]-35)
    canvas.drawString(44,24,'Материалы: 29.10, 14:09 МСК  •  Slurm: 13:47:57  •  Внутренний отчёт')
    canvas.drawRightString(A4[0]-44,24,str(doc.page))
    canvas.restoreState()

base=HERE/'report_base.pdf'
doc=BaseDocTemplate(str(base),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=51,bottomMargin=42,title='HGND: прогресс и ход работ 21–29 октября 2026',author='HGND internal analysis',invariant=1)
frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id='normal',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
doc.addPageTemplates(PageTemplate(id='normal',frames=frame,onPage=page_frame))
story=[]
for i,page in enumerate(pages):
    if i:story.append(PageBreak())
    for b in page:
        kind=b['type']
        if kind in ['p','h','title','callout']:
            story.append(para(b['text'],kind))
        elif kind=='table':
            data=[[para(c,'th') for c in b['head']]]+[[para(c,'cell') for c in row] for row in b['rows']]
            widths=[v*doc.width for v in b['widths']] if b['widths'] else None
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('LINEBELOW',(0,0),(-1,0),.6,ink),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d7e0e5')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Report')]))
            story.extend([table,Spacer(1,10)])
        elif kind=='figure':
            story.append(VectorFigure(FIG/(b['name']+'.pdf')))
            story.extend([Spacer(1,5),para(b['caption'],'caption')])
def report_canvas(*args,**kwargs):
    # Avoid adding ReportLab's unused, unembedded Helvetica resource.
    kwargs['initialFontName']='Report'
    return Canvas(*args,**kwargs)

doc.build(story,canvasmaker=report_canvas)
reader=PdfReader(base);writer=PdfWriter()
for p in reader.pages:writer.add_page(p)
for p in placements:
    src=PdfReader(p['file']).pages[0]
    assert p['x']>=43 and p['y']>=41 and p['x']+p['width']<=A4[0]-43 and p['y']+p['height']<=A4[1]-50,p
    writer.pages[p['page']].merge_transformed_page(src,Transformation().translate(p['x'],p['y']),expand=False)
writer.add_metadata({'/Title':'HGND: прогресс и ход работ 21–29 октября 2026','/Author':'HGND internal analysis','/Subject':'MC-truth hardness, reconstruction, SMASH training and experiment chronology','/CreationDate':"D:20261029140901+03'00'",'/ModDate':"D:20261029140901+03'00'"})
with OUT.open('wb') as f:writer.write(f)
(HERE/'placements.json').write_text(json.dumps(placements,indent=2)+'\n')
base.unlink()
print('Built',len(writer.pages),'PDF pages;',len(placements),'vector figures;',MD.name)
