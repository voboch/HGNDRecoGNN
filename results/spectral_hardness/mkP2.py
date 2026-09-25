"""FIG 2 per protocol §2/§9.7: observables measured on DIFFERENT populations and
estimators must not be plotted as if commensurable -- so the panel is grouped by
population, with the population and its size stated on every row."""
import numpy as np
groups=[
 ("Primary neutrons, full production  (2.35 M events)",[
   ("R,  |y_cm| &lt; 0.5",              29.3, True),
   ("R,  |y_cm| &gt; 0.5  (spectator-rich)", 2.5, False),
   ("n/p,  4π  (confounded)",   23.3, True),
 ]),
 ("HGND acceptance, primary nucleons  (full production)",[
   ("n/p,  band",                     0.92,False),
   ("neutron yield,  band",           1.18,False),
   ("proton yield,  band",            1.67,False),
 ]),
 ("HGND-detected neutrons, schema-v2 caches",[
   ("R,  all detected",               9.9, True),
   ("R,  signal n, seed-42 test half", 6.4, True),
 ]),
]
rows=[r for _,g in groups for r in g]
W=690; L,R,T,Bm=248,104,34,42
rowh=28; grouph=22
H=T+len(rows)*rowh+len(groups)*grouph+Bm
pw=W-L-R; xmax=32.0
xm=lambda v:L+pw*(v/xmax)
o=[]
y=T
bars=[]
for gi,(gname,g) in enumerate(groups):
    o.append(f'<text class="grouplab" x="{L-8}" y="{y+11:.0f}" text-anchor="end">{gname}</text>')
    o.append(f'<line class="grule" x1="{L}" y1="{y+4:.0f}" x2="{L+pw}" y2="{y+4:.0f}"/>')
    y+=grouph
    for lab,sig,ordered in g:
        bars.append((y+rowh/2,lab,sig,ordered)); y+=rowh
ph=y-T
o.insert(0,f'<rect class="frame" x="{L}" y="{T}" width="{pw}" height="{ph}"/>')
for v in (0,5,10,15,20,25,30):
    x=xm(v)
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T+ph}" x2="{x:.1f}" y2="{T+ph-6}"/>')
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+6}"/>')
    o.append(f'<text class="ax" x="{x:.1f}" y="{T+ph+15}" text-anchor="middle">{v}</text>')
for v in range(1,32):
    if v%5==0: continue
    x=xm(v)
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T+ph}" x2="{x:.1f}" y2="{T+ph-3.5}"/>')
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+3.5}"/>')
for v,lab in ((3,"3σ"),(5,"5σ")):
    o.append(f'<line class="guide" x1="{xm(v):.1f}" y1="{T}" x2="{xm(v):.1f}" y2="{T+ph}"/>')
    o.append(f'<text class="guidelab" x="{xm(v):.1f}" y="{T-5}" text-anchor="middle">{lab}</text>')
for yc,lab,sig,ordered in bars:
    w=xm(min(sig,xmax))-L
    cls="ord" if ordered else "noord"
    o.append(f'<rect class="bar {cls}" x="{L}" y="{yc-5.5:.1f}" width="{w:.1f}" height="11"/>')
    lab=lab.replace("<","&lt;").replace(">","&gt;") if "&" not in lab else lab
    o.append(f'<text class="rowlab" x="{L-8}" y="{yc+3.5:.1f}" text-anchor="end">{lab}</text>')
    o.append(f'<text class="val" x="{L+w+6:.1f}" y="{yc+3.5:.1f}">{sig:.1f}σ{"" if ordered else "  not ordered"}</text>')
o.append(f'<text class="axt" x="{L+pw/2:.0f}" y="{H-10}" text-anchor="middle">statistical significance of the S&#8347;&#8340;&#8348; = 90 vs 0 MeV separation</text>')
o.append(f'<text class="lgh" x="{L+pw+8}" y="{T+12}">shading</text>')
o.append(f'<rect class="bar ord" x="{L+pw+8}" y="{T+19}" width="14" height="8"/>')
o.append(f'<text class="lg" x="{L+pw+26}" y="{T+26}">ordered</text>')
o.append(f'<rect class="bar noord" x="{L+pw+8}" y="{T+35}" width="14" height="8"/>')
o.append(f'<text class="lg" x="{L+pw+26}" y="{T+42}">not</text>')
print(f"{W}|{H}"); print("\n      ".join(o))
