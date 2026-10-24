# Аудит corrected SMASH перед переобучением

Разрешённый пользователем первый этап плана от 21 октября: аудит данных и
смены определения signal, без нового обучения или inference. Исходные данные
читаются из `data_fixed`; старые кеши не объявляются исправленной production
без manifest соответствия. Все результаты этого этапа — development diagnostics.

Ожидание до измерения: удаление `Ekin>0.3GeV` может добавить мягкие positive
clusters, но не изменить label или максимальный energy target прежних positive
clusters. Доля добавленных кластеров по U_sym неизвестна. По фиксированным
предсказаниям, cuts и energy bands реконструированная величина не меняется от
переименования MC labels. Направление эффекта после обучения неизвестно.

Входы: corrected detector exports для U_sym=0/18/90 MeV, пары hits/vacs с
локальными Row/Instance и Row/Id. SHA256 сохраняется для каждой пары;
повторные непустые файлы, пропавшие MC associations и неоднозначные ключи
останавливают аудит. События с нулём surviving candidates остаются в export
event inventory. Оно не включает generated events без detector export.

Кандидаты: штатный cut Edep>0.003 GeV, sigma_t=0.15 ns, radius=3.6 cells,
time edge=1.5 ns, singleton retained, native PDG0 half-event rejection.
Все значения наследуют действующий SMASH preprocessing и здесь не
оптимизируются. Новая детерминированная реализация шума seed 42 общая для
двух labels; это не восстановление случайных чисел исторического кеша.
Prompt matching сохраняет native ToF полосу ±0.3 ns и Side 5/6. Меняется только
нижняя граница Ekin в signal-hit rule.

Ledger отдельно содержит detector-export neutron IDs до hit cut, eligible
prompt IDs после cut и maximum-cardinality one-to-one candidate/object
matching, merging/splitting. Эти знаменатели не заменяют full generated
neutron inventory или detection efficiency. Первые три доступных primary
файла на потенциал дают ограниченный lineage probe по Row/Id, PDG и vertex;
совпадение этого probe не доказывает полную coverage или независимость jobs.

Бюджет: один CPU smoke, по одной паре на потенциал, максимум 10мин; после
COMPLETED/exit 0 — три CPU tasks, максимум 2 одновременно, до 2часов каждый.
GPU0, CPU4, --mem=0. Никаких автоматических повторов и новых training jobs.
Полный аудит не запускается при неуспехе smoke; failure сохраняется явно.
При нарушении source linkage дальнейшее обучение ждёт восстановления
provenance, а диагностика уже завершённого DCM продолжается независимо.

Физические границы energy bands пока не назначены. Новый finite-band API
использует [E_min,E_L) и [E_H,E_max), сохраняет zero-candidate events и
отказывается делить histogram bin по предположению о его форме. Legacy
counter defaults сохранены для воспроизводимости старых результатов.
