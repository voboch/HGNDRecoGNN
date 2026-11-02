# Предзапускная проверка S-only census

02.11.2026. Авторизация — прямое поручение пользователя продолжать
эксперименты автономно; исходный запрос требует критиков перед запуском.

`/root/energy_technical_critic`: **GO для одного CPU-smoke A**.
FullS286 после принятого9-file smoke на том же source/plan и проверки
измеренного времени относительно45мин. Проверены real call path и
ограничения: pinned config/split/prepare/census; S/even/validation roles;
exact observation/native replay; direct-source proof; Mother0/nonprimary;
node-local work; compact immutable publication; full без smoke отказывается.
Графы/scalеры не строятся, модели не вызываются. Нечётные outcomes не читаются.

Критик независимо выполнил38focused tests за14.87с, один внешний PyG warning;
`bash -n slurm/label_energy.hpc.sbatch` PASS. Дополнительных blockers не нашёл.
Ограничение обеих фаз:4CPU×(15+45)мин = максимум4CPU·ч,0GPU·ч, последовательно.

Физический критик подтвердил возможность reuse exact direct-primary binding
без новой полной primary-редукции и уточнил значение `vacs.Ekin` по converter.
В origin ledger четыре disjoint категории; mixed и multi-object различаются.
Неизвестная secondary genealogy не останавливает корректно названный
кластерный опыт, но не позволяет объявить первичную эффективность или R_n.

Проверки основного агента перед фиксацией source:

-38focused tests PASS; ruff на изменённых Python-файлах PASS; SBATCH syntax PASS.
-Полный **активный** набор `python -m pytest tests -q -p no:cacheprovider`:
  **347passed,19warnings,41.71с**. Это путь `Makefile test`.
-`python -m compileall -q analysis scripts tests`: PASS.
-Голый `pytest` сначала попытался собрать архивные package copies из
  `results/` и `tmp/` и завершился import-file-mismatch при collection.
  Архивы не удалялись и не менялись; используется канонический active-tests
  путь, который прошёл полностью.

Last science/model-job:29.10, новый запрос02.11. Между ними пользовательская
пауза; инфраструктурной работы дольше рабочего дня не было. Нужный результат —
содержательный census влияния historical energy floor и основание no-floor
абляции. Простой путь — одна CPU-smoke, затем один full CPU-run.
Исходники и предсказания фиксируются commit до измерения.
