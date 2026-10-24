# Первый перенос фиксированной GraphSAGE DCM→SMASH

22 октября 2026. GPU smoke4435668 завершился COMPLETED/exit0 на cn-015
за1мин38с, Tesla V10032GB, PyTorch2.4.0+cu121. Все девять комбинаций
трёх SMASH potentials и training seeds42/123/456 завершены на CUDA.
SMASH training и calibration не выполнялись.

## Популяция и фиксированные условия

Результат относится к первому corrected detector file10943245/0001 каждого
потенциала. Это retrospective development engineering probe, выбранный до
model scores. Соответствие primary: zeroSpot0001→0001,
defaultSpot0001→0002, bigSpot0001→0001. CPU audit4435628 завершился
COMPLETED/exit0 за3мин34с; все1264/334/1216 eligible primary n/p records
согласованы по Row/Id, PDG и production vertex.

GraphSAGE weights/scalers — завершённая native DCM campaign
student_pipeline_20261020_56bcc48206, обучение/scaler fit на988975,
checkpoint и score cut на988974. Frozen cuts для seeds42/123/456:
0.46507/0.357785/0.397705. Эти значения наследуются из DCM validation
и не подбирались на SMASH. Input recipe:10features, radius3.6cells,
time window1.0ns, singleton candidates retained. MC не входит в forward.
CPU label audit использовал своё прежнее окно1.5ns; его clusters не были
поданы в DCM модель. Поэтому counts двух видов diagnostic различаются.

Timing augmentation: t=t_CSV+Normal(0,0.15ns). Это унаследованная конвенция
preprocessing; σ здесь не калибровалась по детекторным данным. Новый draw
воспроизводится из seed42 и hash относительного имени detector file.
Native/relaxed labels и transfer используют один observed-hit draw.
Побайтная node-local копия сохраняет этот draw, что проверено тестом.
Исторический RNG старых кешей не восстановлен. Физическое разрешение и
наличие прежнего smearing внутри CSV данным опытом не установлены.
Общая последовательность random draws у файлов с одинаковым относительным
именем не создаёт physical-event pairing между разными potentials.

## Пересчитанные метрики

<!-- table-provenance: scored on retrospective development first-file MC-labelled clusters, not independent test; model and scaler fitted on DCM train988975, checkpoint and cuts on validation988974. Counts are inventory quantities: no direction is better. -->

Каждая запись — среднее±sample SD по трём DCM training seeds на одной
фиксированной SMASH population. MC/source sampling uncertainty не оценена.
Clusters и MC positives одинаковы у трёх seeds; predictions не объединяются
в одну ROC. Эффективность ниже — доля выбранных MC-positive clusters.
Energy MAE считается на всех native MC-positive clusters без score cut.

| U_sym (MeV) | clusters (count) | MC-positive (count) | ROC-AUC ↑ | AP ↑ | cluster purity ↑ | cluster efficiency ↑ | native-target MAE (MeV) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6233 | 1116 | 0.899 ± 0.007 | 0.706 ± 0.007 | 0.706 ± 0.010 | 0.573 ± 0.012 | 160 ± 16 |
| 18 | 1611 | 329 | 0.909 ± 0.007 | 0.743 ± 0.006 | 0.750 ± 0.016 | 0.548 ± 0.008 | 174 ± 25 |
| 90 | 6302 | 1137 | 0.900 ± 0.007 | 0.699 ± 0.011 | 0.703 ± 0.007 | 0.547 ± 0.020 | 173 ± 15 |

Single-file composition и размеры выборок различаются между potentials.
Сравнения между этими строками не являются тестом U_sym response. Они не
позволяют объявить0.909 лучше0.899 или установить differential bias.
Высокий ROC-AUC не заменяет neutron matching и observable closure.

Отдельный paired label audit на прежних SMASH components с окном1.5ns
показал: снятие только Ekin>0.3GeV увеличивает число positive clusters
1111→1129 для0MeV,326→329 для18MeV и1133→1161 для90MeV. Из добавленных
18/3/28 components одиночными являются12/2/19. Energy targets прежних
positives не изменились. Это exact MC relabelling counts первых файлов,
не прирост reconstructed efficiency и не прогноз чувствительности к U_sym.
Эти counts не подставляются в transfer metrics с другим time window1.0ns.

![ROC по трём DCM training seeds](../results/smash_transfer_staged_20261022/figures/roc.png)

ROC reconstructed scores против native MC labels, Xe+Cs2.87GeV/A, без
добавленного centrality cut; исходный b-range не установлен. Каждый panel
показывает один first-file source и три training seeds. Диагональ — random
ranking reference. ROC-AUC приведён в таблице; MC-statistical band не
построен. Это engineering illustration, не independent-test figure.

![CDF остатков native energy target](../results/smash_transfer_staged_20261022/figures/energy_residual_cdf.png)

Empirical CDF E_pred−E_target на native MC-positive clusters без score cut,
по тем же sources/seeds. Все tails сохранены; histogram bins и physics
energy bands не выбирались. Target — maximum qualifying-hit initial
MCTrack kinetic energy. Он не является энергией индивидуального matched
neutron после transport. Sample SD в таблице показывает variation weights,
а не neutron energy resolution. Обе PNG figures200dpi визуально проверены;
автоматические data/annotation bounds assertions прошли.

## Что остаётся до полного SMASH измерения

Node-local input bundle407files/1.32GB позволил завершить first-file audit
и inference без прежнего compute EACCES. Это не объясняет и не закрывает
проблему исходного filesystem. Bundle hashes подтверждают побайтную копию.

Карта200 defaultSpot candidate names завершена4435670, COMPLETED/exit0
на cn-035 за17мин58с. Проверены все200 primary files. Все225844 eligible
detector n/p records из186 непустых files имеют однозначный exact match
по initial MCTrack PDG/Ekin/Px/Py/Pz/vX/vY/vZ, с неизменными Row/Id.
У186 detector files —186 различных primary source files; ambiguous
detector records0. Повторных source tracks/events между detector files
среди сопоставленных eligible records не найдено. Это не проверка всех
generated events без detector response. Все200 primary content hashes
совпали с independent read pass4435586; detector hashes совпали с bundle.
Сдвиг числового primary filename относительно detector filename меняется
от+1 до+14; одного постоянного offset для исправления имён недостаточно.

14 header-only vacs не имеют binding evidence и остаются unresolved.
Generated-event count и generation-job independence этой картой не
установлены. Полный результат и проверка:
`source_mapping_verified.json`; per-record mapping сохранён с SHA256.

Предыдущий4435627 FAILED за25с на обработке header-only CSV; причиной был
TypeError в нашем numeric finite check. Исправлен только этот случай,
покрыт новым test; выполнено одно ограниченное продолжение по протоколу.
14 vacs0187–0200 содержат только header. Остальные186 hits/vacs pairs
имеют186 различных content-hash pairs; точных непустых duplicates нет.
Это не исключает повторных/частично пересекающихся MC events — их должен
проверить explicit primary map. Header-only files не удалены из inventory.

До full transfer серии нужны complete mapping и binding остальных файлов,
проверка event overlap и geometry. Generated-event denominator остаётся
неизвестным: detector max Row не определяет число processed zero-hit events.
Generation-job-independent split и fresh final holdout не установлены.
Чувствительность к U_sym, generated-neutron efficiency и unfolding здесь
не рассчитаны. Новый SMASH training protocol пока не назначен.

## Воспроизведение и receipts

Canonical пересчёт: `results/smash_transfer_staged_20261022/metrics_verified.json`.
Все девять CSV hashes и три summary hashes проверены; метрики независимо
пересчитаны из сохранённых предсказаний. Кандидатные ключи/MC labels/targets
совпали между seeds. CSV roundoff проверяется с относительным допуском1e-6
и абсолютным1e-8; это проверка сериализации, не критерий physics agreement.

Команда локального пересчёта и figures:
`python scripts/summarize_smash_transfer.py --root results/smash_transfer_staged_20261022`.
Raw prediction archive сохранён локально и на scratch; его SHA256:
5f124965b1af15f51be1f8e06ff426fa06d2914f28b4a1a07b4fb6d63fd4bbd6.

Frozen source SHA256:
521a10aa4480868dc25614d742650e7cd8948c637807e73bb39e8b39b11a4af2.
Engineering inventory SHA256:
e1d4e92891e89ef635637d92ac8017ace32814dbe7b9c888db8f982c29786966.
Bundle manifest SHA256:
ad7bcfe7132b1d516458a47b2d001f28cc0b9e597e0b209786b1392943c43ea6.

Протокол: `smash_staged_transfer_protocol_20261022.md`.
Source mapping и реальные Slurm states: локальный `launch_status.json` и
submission/terminal receipts в том же results directory. Старый manuscript
и закрытая DCM development campaign не изменены.

Этот конечный этап закрыт: active stage jobs0. Следующий содержательный
переход — полный corrected-SMASH audit/transfer после binding остальных
zeroSpot/bigSpot файлов и согласования empty-file accounting. Обучение на
SMASH требует отдельного source grouping/split; fresh holdout отсутствует.
