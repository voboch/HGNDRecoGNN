"""FIG 1: (a) b distributions ratio to the reference -- the sampling mismatch;
(b) R vs centrality class -- the steep dependence that turns it into a signal.
Protocol: reference darkest and first, closed frame, inward ticks, no grid,
markers+stat bars, ratio panel 2.5:1 symmetric about unity, derived limits."""
import json,os,numpy as np
SP=os.environ["SP"]; J=json.load(open(SP+"/cent_fig.json"))
U=J["U"]; be=np.array(J["b_edges"]); NB=10
W=690; L,R,T=88,26,16; h1,h2=150,232
H=T+h1+h2+54; pw=W-L-R
ARMS=[("zeroSpot","ref","o","solid"),("defaultSpot","prior","s","dash"),("bigSpot","model","^","solid")]
o=[]
def mk(x,y,cls,sh,r=3.6):
    if sh=="o": return f'<circle class="mk {cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if sh=="s": return f'<rect class="mk open {cls}" x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}"/>'
    return f'<polygon class="mk {cls}" points="{x:.1f},{y-r-1:.1f} {x-r-1:.1f},{y+r:.1f} {x+r+1:.1f},{y+r:.1f}"/>'
def eb(x,y,dy,cls,c=3):
    return (f'<line class="er {cls}" x1="{x:.1f}" y1="{y-dy:.1f}" x2="{x:.1f}" y2="{y+dy:.1f}"/>'
            f'<line class="er {cls}" x1="{x-c:.1f}" y1="{y-dy:.1f}" x2="{x+c:.1f}" y2="{y-dy:.1f}"/>'
            f'<line class="er {cls}" x1="{x-c:.1f}" y1="{y+dy:.1f}" x2="{x+c:.1f}" y2="{y+dy:.1f}"/>')
# ---- panel (a): b-distribution ratio to reference ----
ref=np.array(J["b_hist"]["zeroSpot"],float)
xs=lambda v:L+pw*(v/14.0)
vals=[]
for ds,_,_,_ in ARMS:
    if ds=="zeroSpot": continue
    h=np.array(J["b_hist"][ds],float)
    for i in range(len(h)):
        if ref[i]>20 and h[i]>0:
            r=h[i]/ref[i]; se=r*np.sqrt(1/h[i]+1/ref[i]); vals+= [r-se,r+se]
half=max(abs(min(vals)-1),abs(max(vals)-1))*1.12
a_lo,a_hi=1-half,1+half
ya=lambda v:T+h1*(1-(v-a_lo)/(a_hi-a_lo))
o.append(f'<rect class="frame" x="{L}" y="{T}" width="{pw}" height="{h1}"/>')
for v in (0,2,4,6,8,10,12,14):
    x=xs(v)
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T+h1}" x2="{x:.1f}" y2="{T+h1-6}"/>')
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{T+6}"/>')
for v in (0.9,0.95,1.0,1.05,1.1,1.15):
    if not (a_lo<v<a_hi): continue
    y=ya(v)
    o.append(f'<line class="tk" x1="{L}" y1="{y:.1f}" x2="{L+6}" y2="{y:.1f}"/>')
    o.append(f'<line class="tk" x1="{L+pw}" y1="{y:.1f}" x2="{L+pw-6}" y2="{y:.1f}"/>')
    o.append(f'<text class="ax" x="{L-6}" y="{y+3.5:.1f}" text-anchor="end">{v:.2f}</text>')
o.append(f'<line class="unity" x1="{L}" y1="{ya(1.0):.1f}" x2="{L+pw}" y2="{ya(1.0):.1f}"/>')
for ds,cls,sh,_ in ARMS:
    if ds=="zeroSpot": continue
    h=np.array(J["b_hist"][ds],float)
    for i in range(len(h)):
        if ref[i]<=20 or h[i]<=0: continue
        r=h[i]/ref[i]; se=r*np.sqrt(1/h[i]+1/ref[i])
        if not (a_lo<r<a_hi): continue
        x=xs(0.5*(be[i]+be[i+1]))
        o.append(eb(x,ya(r),abs(ya(r)-ya(r+se)),cls)); o.append(mk(x,ya(r),cls,sh))
o.append(f'<text class="pan" x="{L+10}" y="{T+17}">(a)</text>')
o.append(f'<text class="sys" x="{L+pw-8}" y="{T+16}" text-anchor="end">dN/db ratio to S&#8347;&#8340;&#8348; = 0</text>')
o.append(f'<text class="axt" transform="translate(22,{T+h1/2}) rotate(-90)" text-anchor="middle">ratio</text>')
o.append(f'<text class="sys2" x="{L+10}" y="{T+h1-8}">S&#8347;&#8340;&#8348; cannot affect db sampling — differences are generation-level</text>')
# ---- panel (b): R vs centrality class ----
T2=T+h1
rv=[]
for ds in U:
    for d in J["R_com"][ds]:
        if np.isfinite(d["r"]): rv+=[d["r"]-d["se"],d["r"]+d["se"]]
b_lo,b_hi=0,max(rv)*1.10
yb=lambda v:T2+h2*(1-(v-b_lo)/(b_hi-b_lo))
xc=lambda i:L+pw*((i+0.5)/NB)
o.append(f'<rect class="frame" x="{L}" y="{T2}" width="{pw}" height="{h2}"/>')
for i in range(NB+1):
    x=L+pw*(i/NB)
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T2+h2}" x2="{x:.1f}" y2="{T2+h2-6}"/>')
    o.append(f'<line class="tk" x1="{x:.1f}" y1="{T2}" x2="{x:.1f}" y2="{T2+6}"/>')
for i in range(NB):
    o.append(f'<text class="ax" x="{xc(i):.1f}" y="{T2+h2+15}" text-anchor="middle">{i*10}</text>')
for v in (0,0.2,0.4,0.6,0.8):
    y=yb(v)
    o.append(f'<line class="tk" x1="{L}" y1="{y:.1f}" x2="{L+6}" y2="{y:.1f}"/>')
    o.append(f'<line class="tk" x1="{L+pw}" y1="{y:.1f}" x2="{L+pw-6}" y2="{y:.1f}"/>')
    o.append(f'<text class="ax" x="{L-6}" y="{y+3.5:.1f}" text-anchor="end">{v:.1f}</text>')
for ds,cls,sh,_ in ARMS:
    for i,d in enumerate(J["R_com"][ds]):
        if not np.isfinite(d["r"]): continue
        x=xc(i); y=yb(d["r"])
        o.append(eb(x,y,abs(yb(d["r"])-yb(d["r"]+d["se"])),cls)); o.append(mk(x,y,cls,sh))
o.append(f'<text class="pan" x="{L+10}" y="{T2+17}">(b)</text>')
lx=L+pw-188; ly=T2+22
o.append(f'<text class="lgh" x="{lx}" y="{ly}">S&#8347;&#8340;&#8348; [MeV]</text>')
for i,(ds,cls,sh,_) in enumerate(ARMS):
    yy=ly+15+i*15
    o.append(mk(lx+9,yy-3.5,cls,sh))
    tag=" (reference)" if ds=="zeroSpot" else ""
    o.append(f'<text class="lg" x="{lx+22}" y="{yy}">{U[ds]}{tag}</text>')
o.append(f'<text class="axt" transform="translate(22,{T2+h2/2}) rotate(-90)" text-anchor="middle">R = N(E&gt;2)/N(E&lt;1)</text>')
o.append(f'<text class="axt" x="{L+pw/2:.0f}" y="{H-14}" text-anchor="middle">centrality class  [%]   (0 = most central)</text>')
o.append(f'<text class="note" x="{L+pw}" y="{H-14}" text-anchor="end">matched in b; bars statistical</text>')
print(f"{W}|{H}"); print("\n      ".join(o))
