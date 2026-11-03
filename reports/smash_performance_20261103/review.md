# Проверка полного SMASH-отчёта, 03.11.2026

Scope: внутренний отчёт `reports/smash_performance_20261103.md`, 23 семейства
рисунков и PDF; базовый code HEAD `7683773`, branch `smash-train-20261028`.
Рукопись и canonical-входы статьи не менялись. Последний научный результат
принят 03.11 (S286 4481705); последнее модельное обучение/оценка - 29.10.
Сегодня выполнены приёмка census, план следующего arm и запрошенный отчёт.
Новых модельных запусков ради отчёта нет.

## Read-only verdicts

| Роль | Проверенное | Итог |
| --- | --- | --- |
| Technical | 43 input pins, 72 validation-артефакта, 27 head argmins, 27 cuts; описание cut/checkpoints/calibration и replay counts | GO после исправления PR endpoint |
| Methodology / physics | Full-primary/T1/candidate популяции; exact 0,3 и ΔE; MCTrack target; C2/C3/C4; ToF/graph/35 нс; незамыкание Rₙ | GO |
| Text | Весь Markdown; страницы 1/5/23 крупно; units, uncertainty, статус нового обучения | GO с двумя уточнениями, исправлены |
| Manager | Запрос покрыт; научный результат принят сегодня; нет инфраструктурного дрейфа; доставка после final page QA | GO |

Дополнительный statistics critic независимо подтвердил правильность общих
нормировок, сохранения хвостов, per-replica ratios и разделения seed/file SD.
Его итоговый проход прервался из-за лимита сервиса; полный финальный GO этой
дополнительной проверки не приписывается. Найденный им PR defect независимо
подтверждён technical critic и исправлен.

## Находки и исправления

1. **[B] PR-bootstrap endpoint:** при recall=1 использовалась центральная
   prevalence во всех replicas. Теперь endpoint каждой replica равен её
   `N_positive/N_all` с теми же file draws. Центральная PR-кривая, accepted
   exact AUC/AP и frozen cuts не изменились. Рисунок 05 и PDF пересобраны.
2. **[S] Парное сравнение разных весов:** в headline добавлены canonical
   B1−A1 ΔAUC/Δefficiency/ΔMAE, SE по трём общим seed, t, df=2 и tcrit=4,303.
3. **[S] Неоднозначное `t_label=0,3`:** заменено явным energy condition
   `Ekin>0,3 ГэВ`, отдельно от inclusive нижней границы Rₙ.
4. **[P] Selected vs all-positive:** утверждение о median response ≤0,84%
   явно отсылает к all-positive профилю рис.9; selected профиль - рис.10.
5. **[P] Читаемость:** legend type поднят до 7 pt, интервалы/units разнесены,
   S286/F282 расшифрованы, tail-table headers снабжены ГэВ. Две архивные
   design figures увеличены векторно в 1,55 раза для читаемости (письменное
   исключение из правила native-size inclusion; исходные SHA сохранены).
6. При собственной QA убраны искусственные baseline drops у steps и clipping
   ratio bands. Добавлены export assertions всех in-view data vertices и
   error-bar ends. Energy migration mesh рисуется только в объявленном окне;
   его знаменатель и сохранённые tails остались полными.

Итог: GO для внутреннего отчёта. Готовность статьи к публикации и будущий
no-floor training здесь не оценивались. Final PDF SHA и просмотр всех страниц
зафиксированы отдельно в `validation.json`.
