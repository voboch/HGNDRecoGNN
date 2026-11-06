"""Descriptive report of the accepted bounded response reductions."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
R=json.loads((HERE/'results.json').read_text())
assert hashlib.sha256((HERE/'counts.npz').read_bytes()).hexdigest()==R['counts_sha256']
Z=np.load(HERE/'counts.npz',allow_pickle=False)
P=['18/0','90/0','90/18'];colors=['#a55022','#174d78'];markers=['s','^']
plt.rcParams.update({'font.size':13,'axes.labelsize':14,'legend.fontsize':12,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
manifest={}
def save(fig,name):
 fig.text(.99,-.015,'SMASH Xe+Cs, 2.87 AGeV · simulation · S286, seed 42 · no B selection',ha='right',va='top',fontsize=10)
 fig.canvas.draw()
 for ext in ['png','pdf']:
  p=HERE/(name+'.'+ext);fig.savefig(p,dpi=180,bbox_inches='tight',pad_inches=.15);manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 plt.close(fig)

fig,ax=plt.subplots(figsize=(10.4,4.1),layout='constrained')
d=R['pairs']['90/0']
for a,label in enumerate(['A1: DCM pipeline','B1: SMASH pipeline']):
 y=np.array(d['adjacent_rung_delta_pp'])[0,a];err=np.array(d['adjacent_rung_file_SD_pp'])[0,a]
 ax.errorbar(np.arange(4)+(a-.5)*.10,y,yerr=err,fmt=markers[a],color=colors[a],capsize=3,label=label)
ax.axhline(0,color='.4',ls='--');ax.set_xticks(range(4),['C1 − T1','C2 − C1','C3 − C2','C4 − C3']);ax.set(ylabel='90/0 contrast change (pp)',xlabel='MC-assisted response transition')
fig.legend(*ax.get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False);save(fig,'01_stage_shifts')
fig,ax=plt.subplots(figsize=(10.4,4.1),layout='constrained')
for stage,label,col,m in [(4,'Uncorrected B1 C4','#333333','o'),(7,'B1 frozen M_hard',colors[1],'^')]:
 y=[];err=[]
 for p in P:
  d=R['pairs'][p]
  if stage==4:
   c=np.array(d['central_percent'])[0,1]
   hi,lo={'18/0':(1,0),'90/0':(2,0),'90/18':(2,1)}[p]
   rr=Z['replicas_Rn'];v=100*(rr[:,0,1,:,hi]/rr[:,0,1,:,lo]-1)
   y.append(c[4]-c[0]);err.append((v[:,4]-v[:,0]).std(ddof=1))
  else:
   y.append(np.array(d['corrected_minus_T1_pp'])[0,1]);err.append(np.array(d['corrected_minus_T1_SD_pp'])[0,1])
 ax.errorbar(np.arange(3)+(stage-5.5)*.035,y,yerr=err,fmt=m,color=col,capsize=3,label=label)
ax.axhline(0,color='.4',ls='--');ax.set_xticks(range(3),P);ax.set(xlabel='Potential pair (MeV)',ylabel='Reco minus truth contrast (pp)')
fig.legend(*ax.get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False);save(fig,'02_correction_residual')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
J=Z['joint'][:,0];W=Z['weights']
for ax,truth in zip(axes,[1,0]):
 for a,label in enumerate(['A1','B1']):
  sub=J[:,truth];den=sub.sum(axis=(-1,-2));num=sub[:,: ,1,:].sum(-1) if a==0 else sub[:,:,:,1].sum(-1)
  val=100*num.sum(0)/den.sum(0);rep=100*(W@num)/(W@den)
  ax.errorbar(np.arange(4)+(a-.5)*.08,val,yerr=rep.std(0,ddof=1),fmt=markers[a],color=colors[a],capsize=3,label=label)
 ax.set_xticks(range(4),['1','2–3','4–7','8+']);ax.set(xlabel='Observed hits per cluster',ylabel='MC-positive recall (%)' if truth else 'MC-negative selected (%)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False);save(fig,'03_selection_by_size')
fig,ax=plt.subplots(figsize=(10.4,4.1),layout='constrained')
for a,label in enumerate(['A1','B1']):
 y=[np.array(R['pairs'][p]['calibrated_minus_raw_C4_pp'])[0,a] for p in P];err=[np.array(R['pairs'][p]['calibrated_minus_raw_C4_SD_pp'])[0,a] for p in P]
 ax.errorbar(np.arange(3)+(a-.5)*.10,y,yerr=err,fmt=markers[a],color=colors[a],capsize=3,label=label)
ax.axhline(0,color='.4',ls='--');ax.set_xticks(range(3),P);ax.set(xlabel='Potential pair (MeV)',ylabel='Calibrated minus raw C4 contrast (pp)')
fig.legend(*ax.get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False);save(fig,'04_calibration_control')

s='''# Отклик реконструкции до новой dE/dx продукции\n\n6 ноября 2026. Расчёт завершён локально, без новых model calls, обучения,\nGPU/Slurm jobs, cut/calibration fits, odd report scoring или выбора окон.\nS286: 286 truth-bound even-part content, 1 798 225 кандидатов. Прежние TEST\npredictions относятся к каждой исходной cross-fit test-role. Это известная\nретроспективная популяция, не новый holdout и не no-floor validation.\n\nВся арифметика и SHA: [results.json](results.json),\n[per-file arrays и shared replicas](counts.npz),\n[зарегистрированный до исполнения протокол](protocol.md).\n\n## Изменения по этапам\n\nКонтраст 90/0, central ± paired whole-file bootstrap SD, п.п.\nT1 — primary truth в прежнем ray/processed-span окне; C1 — native-positive\nкластеры с MC target; C2 — выбранные positives с MC target; C3 — те же\nвыбранные positives с reconstructed calibrated energy; C4 — все selected.\nПереход T1→C1 сопоставляет разные физические объекты, не является\nсамостоятельной оценкой primary detection efficiency.\n\n|Arm|Seed|C1−T1|C2−C1|C3−C2|C4−C3|\n|---|---:|---:|---:|---:|---:|\n'''
for a,arm in enumerate(R['arms']):
 for si,seed in enumerate(R['seeds']):
  y=np.array(R['pairs']['90/0']['adjacent_rung_delta_pp'])[si,a];sd=np.array(R['pairs']['90/0']['adjacent_rung_file_SD_pp'])[si,a]
  s+='|'+arm+'|'+str(seed)+'|'+'|'.join(f'{v:.2f} ± {e:.2f}' for v,e in zip(y,sd))+'|\n'
s+='''\n![Stage changes](01_stage_shifts.png)\n\nMC-assisted reconstructed/primary comparison on SMASH Xe+Cs2.87AGeV, no B\nselection, inherited diagnostic0.3≤E<1 and E≥2GeV bands, not optimised here, seed42 as representative. Bars use\n400paired whole-file draws, potential×evenpart strata. All3seeds appear in\nthe table to establish whether stage directions repeat; they share files.\nAdjacent shifts telescope to C4−T1 exactly, but their errors correlate.\n\n## Замороженная фоновая поправка\n\nM_hard: high=NH, low=NL−cNH. c=(NL−SL)/NH взят из исходной validation\nчасти того же fold/seed, pooled over potentials, и воспроизведён против\nпринятого29October результата. Application — другая even TEST часть.\nКоэффициент не знает истинный потенциал. Во всех400replicas его\nнеопределённость пересчитывается теми же весами физических content, что\nи application counts. Это сохраняет covariance между двумя ролями файла.\n\nCentral ± paired file SD, п.п.; ноль residual означает numerical closure\nтолько для этой продукции и определения.\n\n|Pair|Arm|Seed|M_hard−C4|M_hard−T1|\n|---|---|---:|---:|---:|\n'''
for p in P:
 d=R['pairs'][p]
 for a,arm in enumerate(R['arms']):
  for si,seed in enumerate(R['seeds']):
   v=np.array(d['correction_minus_C4_pp'])[si,a];e=np.array(d['correction_minus_C4_SD_pp'])[si,a];rv=np.array(d['corrected_minus_T1_pp'])[si,a];re=np.array(d['corrected_minus_T1_SD_pp'])[si,a]
   s+=f'|{p}|{arm}|{seed}|{v:.2f} ± {e:.2f}|{rv:.2f} ± {re:.2f}|\n'
s+='''\n![Correction residual](02_correction_residual.png)\n\nB1 seed42 on the same S286 and frozen settings. Both residuals use paired\nfile SD against identical T1. Correction is a conditional internal transfer,\nnot an independently validated data correction. It addresses low-band\nMC-labelled background only and leaves high-band background and migration.\nTraining/cut/calibration selection, generator and geometry uncertainty are\nnot included in file SD. No conclusion of equivalence follows from zero\ninside the bars. All three pair contrasts correlate.\n\n## Где расходятся A1/B1\n\n![Selection by cluster size](03_selection_by_size.png)\n\nSame candidates, seed42, native MC labels, frozen cuts; bars: shared\nwhole-file SD. Left is MC-positive candidate recall, right is MC-negative\nselection probability, not per-primary efficiency or background species.\nThe two panels separate signal selection from negative-candidate rejection.\nDifferent weights/scalers/cuts are retained; this does not isolate their causes.
На показанном seed42 B1 recall выше во всех size bins. Отбор MC-negative
кандидатов также выше у B1 для1 и2–3хитов, но ниже для4–7 и8+.
Рост recall не означает равномерного улучшения rejection.\n\n![Same-head energy calibration control](04_calibration_control.png)\n\nSame S286, seed42. Raw A0 energy uses A1 cut, raw B0 uses B1 cut; exact\nscores/keys match. Thus this control changes only the energy map for each\nfixed selected set. Bars: paired file SD. Ordinary A0 at its own different cut\nis not used as the control. All-seed results are retained in JSON.
На показанном seed42 энергетическая карта A1 сдвигает18/0 и90/0 примерно
на−1п.п.; B1 почти не меняет18/0 и умеренно уменьшает остальные контрасты.
Направление и величина зависят от pipeline/pair; это не универсальный offset
и не проверка применимости calibration к экспериментальным данным.\n\n## Условные требования к протонному отклику\n\nДля D=Rn/Rp имеем dK_D/dln(eta_p)=−(100+K_D),\neta_p=kappa_p(a)/kappa_p(b). Ниже local slope, п.п. на единицу\n1% log-response variation; это единица представления, не предположение\nоб измеренной1% систематике PID. Ideal MC protons use the original forward\nray rectangle and generated energy, without magnetic transport/tracking/PID.\n\n|Pair|Seed|B1 hybrid D contrast (%), file SD|Local proton-response slope (pp per 1% log variation)|\n|---|---:|---:|---:|\n'''
for p in P:
 d=R['pairs'][p]
 for si,seed in enumerate(R['seeds']):
  k=np.array(d['hybrid_D_percent'])[si,1];e=np.array(d['hybrid_D_file_SD_percent'])[si,1];sl=np.array(d['D_shift_pp_per_1pct_log_proton_response'])[si,1]
  s+=f'|{p}|{seed}|{k:.2f} ± {e:.2f}|{sl:.3f}|\n'
s+='''\nS286 hybrid values do not replace the existing568-file hybrid report.\nA constant common proton response factor cancels in the potential contrast;\nan energy-dependent response can yield different effective factors because\nthe spectra differ. The slope quantifies a requirement, not real proton PID.\n\n## Что пока не измерено\n\nCSV не содержат raw hit features/scaler inputs, полной genealogy или\nprimary-to-hit denominator. Поэтому видовой состав material background,\nпричинная scaler/weights ablation и primary detection efficiency здесь не\nполучены. Пары native MC labels не подменяют происхождение shower.\n\nДля этих частей нужны отдельные native/raw inputs и конкретный CPU plan.\nРеальная proton tracking/PID проверка ждёт применимой продукции.\n[Требования к новой MC](new_mc_acceptance.md) подготовлены отдельно.\nNo-floor не повторялся, окна не оптимизировались, новых весов нет.\n'''
# Report measured conclusions without fitting a new choice to them.
d=R['pairs']['90/0'];c=np.array(d['central_percent']);sd=np.array(d['file_SD_percent'])
conclusion=f"""## Основные результаты

На этой S286 популяции primary90/0={c[0,1,0]:.2f} ± {sd[0,1,0]:.2f}%
(file SD). B1 C4 по трём seeds={c[:,1,4].min():.2f}–{c[:,1,4].max():.2f}%,
после M_hard={c[:,1,7].min():.2f}–{c[:,1,7].max():.2f}%. Центральные значения
сдвинулись к truth; corrected−truth residual остаётся около−2п.п.
с file SD около2,3п.п. Это не доказывает closure или разрешённый выигрыш.
Для18/0 коррекция сдвигает центральные значения дальше в отрицательную сторону,
для90/18 они переходят выше truth. Направление эффекта зависит от пары.

У B1 для90/0 C1−T1=−1,30п.п. во всехseeds, C2−C1=−1,22…−0,92п.п.,
C4−C3=−1,24…−0,70п.п. Переход энергии C3−C2 меняется от−0,04 до+0,47п.п.
Это условный stage ledger с коррелированными ошибками из таблицы ниже,
не независимые значимости и не разложение причин веса/scaler.

"""
s=s.replace('## Изменения по этапам',conclusion+'## Изменения по этапам',1)
s+='\n## Совместные решения и общая популяция энергетического сравнения\n\n'
s+='Числа на тех жеS286; MC-native positives. MAE относится к сохранённому MCTrack target. Ошибки MAE — paired-content file bootstrap SD, не training-seed uncertainty.\n\n'
s+='|Seed|A1-only positives|B1-only positives|Both selected positives|A1 MAE on both (MeV)|B1 MAE on both (MeV)|\n|---|---:|---:|---:|---:|---:|\n'
for si,seed in enumerate(R['seeds']):
 joint=Z['joint'][:,si,1].sum(axis=(0,1));err=Z['error_sums'][:,si,:,1];cs=err.sum(0);br=np.einsum('df,fac->dac',Z['weights'],err)
 mae=1000*cs[:,2]/cs[:,0];spread=(1000*br[:,:,2]/br[:,:,0]).std(0,ddof=1)
 s+=f"|{seed}|{joint[1,0]}|{joint[0,1]}|{joint[1,1]}|{mae[0]:.2f} ± {spread[0]:.2f}|{mae[1]:.2f} ± {spread[1]:.2f}|\n"
s+='\nТаблица отделяет различия выбора объектов от ошибки энергии на общем selected subset. Различия trained pipelines остаются смешанными: это не изолированная weights/scaler абляция.\n'
(HERE/'README.md').write_text(s)
(HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
