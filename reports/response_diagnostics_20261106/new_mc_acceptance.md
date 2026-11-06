# Приёмка новой продукции с dE/dx

6 ноября 2026. Спецификация будущей проверки, не результат на новой MC.
Предназначена для отделения neutron-method performance от полной n/p feasibility.

| Что фиксировать | Что проверить до physics scoring | Что блокирует интерпретацию |
|---|---|---|
| Production identity | generator, beam-energy convention, target, field, geometry, reconstruction/calibration versions | отсутствие версии или несогласованные условия |
| Event inventory | полные generated/processed event IDs, включая zero-track/zero-cluster события | denominator только по событиям с кандидатами |
| Primary/transport truth | PDG, vertex, four-momentum/units, track IDs и доступные mother links, transport-to-detector association | приравнивание прямого луча к charged-track acceptance |
| Tracking | charge/rigidity, fit/quality fields, valid subsystem links, MC match с правилами неоднозначности | отсутствующие links, clone/merge учёт по умолчанию |
| dE/dx | actual branch/getter и units, доля заполненных конечных значений по доступным track categories, distributions versus rigidity | constructor zero/missing value выдан за измерение |
| ToF | match IDs, timing/start/length, mass-squared/units, применимость calibration к geometry/run | совпадение имени calibration без подтверждения применимости |
| PID controls | доступные calibrant/control populations, эффективность и примесь как функции кинематики | purity выведена из MC truth без оценки применимости к data |
| Physics roles | заранее заданные fit/development/evaluation content groups и наблюдаемое/acceptance | одни outcomes выбирают поправку и подтверждают её перенос |

В старом proton runbook dE/dx определялся через GetdQdNLower, а не
несуществующий GetDedx; единицы были calibrated GEM charge, не MeV/cm.
Это проверяемая гипотеза о схеме новой продукции, не гарантия её API.
Исторические/отозванные diagnoses и проценты из runbook не переносятся
на новую MC. В частности, признаки отсутствующего dE/dx и отсутствующего
tracking link нужно измерять отдельно, а не выводить один из другого.

Порядок: inventory/schema, затем track/ToF/dE/dx occupancy и зависимости,
затем MC-assisted tracking/PID efficiency и contamination, затем proton spectra
и hybrid/full D с теми же event roles. Detector cuts и PID working point
получают отдельный development-only протокол после проверки полей.

До такой приёмки текущий ideal-MC-proton denominator сохраняет явную пометку:
нет magnetic transport, tracking/PID и detector efficiencies. Требования к
kappa_p из алгебраического budget не считаются измеренной характеристикой PID.
