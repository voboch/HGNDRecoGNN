"""FIG 1 per plotting_style_protocol:
 §2 reference (S_pot=0) darkest, plotted FIRST, marker 'o'; arms firebrick 's'
    dashed and blue '^' solid. Colour is never the only channel.
 §4 ratio sub-panel, height ratio 2.5:1, shared x, unity line, SYMMETRIC y-range.
 §5 units in brackets; species+selection on the panel; b-window stated.
"""
import json, numpy as np
B=[b for b in json.load(open("/tmp/spec.json"))["bins"] if b["hi"]<=3.0]
W=690; L,R,T=92,26,16
h1,h2=250,100                      # 2.5:1 per §4
H=T+h1+h2+52
pw=W-L-R; xmax=3.0
xm=lambda v:L+pw*(v/xmax)
# Axis limits are DERIVED from the data, never hardcoded: a previous version
# fixed them at 0.008-3.0 for data spanning 0.056-21.3, putting 75 % of the
# points off the top of the panel. Containment is asserted below.
_vals=[b[k]["n"]/b[k]["ev"]/(b["hi"]-b["lo"]) for b in B for k in ("0","18","90")]
_dmin,_dmax=min(_vals),max(_vals)
# snap to a 1-3-10 sequence rather than full decades, so the data fills the
# panel instead of sitting in the middle of an over-wide axis
_seq=[m*10.0**e for e in range(-4,4) for m in (1,3)]
ylo=max([c for c in _seq if c<=_dmin/1.6])
yhi=min([c for c in _seq if c>=_dmax*1.3])
ya=lambda v:T+h1*(1-(np.log10(v)-np.log10(ylo))/(np.log10(yhi)-np.log10(ylo)))
assert ylo<_dmin and yhi>_dmax, f"panel (a) clips data: {_dmin:.3g}-{_dmax:.3g} vs axis {ylo:g}-{yhi:g}"
YMAJ=[c for c in _seq if ylo<=c<=yhi and abs(np.log10(c)-round(np.log10(c)))<1e-9]
YMIN=[c for c in [m*10.0**e for e in range(-4,4) for m in (2,3,4,5,6,7,8,9)] if ylo<c<yhi]
# §4 symmetric about unity, sized to contain the points and their error bars
_rv=[]
for b in B:
    y0=b["0"]["n"]/b["0"]["ev"]; y9=b["90"]["n"]/b["90"]["ev"]
    r=y9/y0; se=r*np.sqrt(1/b["0"]["n"]+1/b["90"]["n"])
    _rv += [r-se, r+se]
_half=max(abs(min(_rv)-1.0), abs(max(_rv)-1.0))*1.15
b_lo,b_hi=1.0-_half,1.0+_half
yb=lambda v:T+h1+h2*(1-(v-b_lo)/(b_hi-b_lo))
o=[]
def frame(y0,hh,yt,ym,fmt,xlab,ymin=None):
    out=[f'<rect class="frame" x="{L}" y="{y0}" width="{pw}" height="{hh}"/>']
    for v in (0,0.5,1.0,1.5,2.0,2.5,3.0):
        x=xm(v)
        out.append(f'<line class="tk" x1="{x:.1f}" y1="{y0+hh}" x2="{x:.1f}" y2="{y0+hh-6}"/>')
        out.append(f'<line class="tk" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+6}"/>')
        if xlab: out.append(f'<text class="ax" x="{x:.1f}" y="{y0+hh+15}" text-anchor="middle">{v:g}</text>')
    for v in np.arange(0.1,3.0,0.1):
        x=xm(v)
        out.append(f'<line class="tk" x1="{x:.1f}" y1="{y0+hh}" x2="{x:.1f}" y2="{y0+hh-3.5}"/>')
        out.append(f'<line class="tk" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+3.5}"/>')
    for v in yt:
        y=ym(v)
        out.append(f'<line class="tk" x1="{L}" y1="{y:.1f}" x2="{L+6}" y2="{y:.1f}"/>')
        out.append(f'<line class="tk" x1="{L+pw}" y1="{y:.1f}" x2="{L+pw-6}" y2="{y:.1f}"/>')
        out.append(f'<text class="ax" x="{L-6}" y="{y+3.5:.1f}" text-anchor="end">{fmt(v)}</text>')
    for v in (ymin or []):
        y=ym(v)
        out.append(f'<line class="tk" x1="{L}" y1="{y:.1f}" x2="{L+3.5}" y2="{y:.1f}"/>')
        out.append(f'<line class="tk" x1="{L+pw}" y1="{y:.1f}" x2="{L+pw-3.5}" y2="{y:.1f}"/>')
    return out
def mk(x,y,cls,shape,r=3.6):
    if shape=="o": return f'<circle class="mk {cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if shape=="s": return f'<rect class="mk open {cls}" x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r}" height="{2*r}"/>'
    return f'<polygon class="mk {cls}" points="{x:.1f},{y-r-1:.1f} {x-r-1:.1f},{y+r:.1f} {x+r+1:.1f},{y+r:.1f}"/>'
def eb(x,y,dy,cls,c=3):
    return (f'<line class="er {cls}" x1="{x:.1f}" y1="{y-dy:.1f}" x2="{x:.1f}" y2="{y+dy:.1f}"/>'
            f'<line class="er {cls}" x1="{x-c:.1f}" y1="{y-dy:.1f}" x2="{x+c:.1f}" y2="{y-dy:.1f}"/>'
            f'<line class="er {cls}" x1="{x-c:.1f}" y1="{y+dy:.1f}" x2="{x+c:.1f}" y2="{y+dy:.1f}"/>')

o+=frame(T,h1,YMAJ,ya,lambda v:(f"{v:g}" if v>=0.1 else f"{v:g}"),False,YMIN)
# §2: reference first and darkest
ARMS=[("0","ref","o","solid"),("18","prior","s","dash"),("90","model","^","solid")]
for key,cls,shape,ls in ARMS:
    pts=[]
    for b in B:
        y=b[key]["n"]/b[key]["ev"]/(b["hi"]-b["lo"])
        pts.append(f"{xm(b['lo']):.1f},{ya(y):.1f} {xm(b['hi']):.1f},{ya(y):.1f}")
    o.append(f'<polyline class="st {cls} {ls}" points="{" ".join(pts)}"/>')
    # markers at bin centres, sparse so they stay legible
    for b in B[::2]:
        y=b[key]["n"]/b[key]["ev"]/(b["hi"]-b["lo"])
        o.append(mk(xm(0.5*(b["lo"]+b["hi"])),ya(y),cls,shape))
o.append(f'<text class="pan" x="{L+10}" y="{T+18}">(a)</text>')
o.append(f'<text class="sys" x="{L+pw-8}" y="{T+17}" text-anchor="end">Xe + CsI, √s&#8345;&#8345; = 2.87 GeV &#183; primary neutrons</text>')
o.append(f'<text class="sys2" x="{L+pw-8}" y="{T+30}" text-anchor="end">|y&#8344;&#8344;| &lt; 0.5 &#183; no impact-parameter selection (B unavailable)</text>')
lx=L+pw-196; ly=T+48
o.append(f'<text class="lgh" x="{lx}" y="{ly-3}">S&#8347;&#8340;&#8348; [MeV]</text>')
for i,(key,cls,shape,ls) in enumerate(ARMS):
    yy=ly+14+i*15
    o.append(f'<line class="st {cls} {ls}" x1="{lx+2}" y1="{yy-3.5}" x2="{lx+26}" y2="{yy-3.5}"/>')
    o.append(mk(lx+14,yy-3.5,cls,shape))
    tag = " (reference)" if key=="0" else ""
    o.append(f'<text class="lg" x="{lx+32}" y="{yy}">{key}{tag}</text>')
o.append(f'<text class="axt" transform="translate(22,{T+h1/2}) rotate(-90)" text-anchor="middle">(1/N&#8337;&#8341;) dN/dE&#8342;&#8347;&#8342;  [GeV&#8315;&#185;]  (log)</text>')
# ratio panel
_step=0.05 if _half>0.07 else 0.02
_maj=[round(1.0+k*_step,3) for k in range(-3,4) if b_lo< 1.0+k*_step <b_hi]
_min=[round(1.0+k*_step/2,3) for k in range(-7,8) if b_lo< 1.0+k*_step/2 <b_hi]
o+=frame(T+h1,h2,_maj,yb,lambda v:f"{v:.2f}",True,_min)
for _r in _rv:
    assert b_lo<=_r<=b_hi, f"ratio panel clips a point/error bar: {_r:.4f} outside {b_lo:.3f}-{b_hi:.3f}"
o.append(f'<line class="unity" x1="{L}" y1="{yb(1.0):.1f}" x2="{L+pw}" y2="{yb(1.0):.1f}"/>')
for b in B:
    x=xm(0.5*(b["lo"]+b["hi"]))
    y0=b["0"]["n"]/b["0"]["ev"]; y9=b["90"]["n"]/b["90"]["ev"]
    r=y9/y0; se=r*np.sqrt(1/b["0"]["n"]+1/b["90"]["n"])
    o.append(eb(x,yb(r),abs(yb(r)-yb(r+se)),"model")); o.append(mk(x,yb(r),"model","^"))
o.append(f'<line class="cross" x1="{xm(0.47):.1f}" y1="{T+h1+5}" x2="{xm(0.47):.1f}" y2="{T+h1+h2-5}"/>')
o.append(f'<text class="crosslab" x="{xm(0.47)+5:.1f}" y="{T+h1+14}">0.47 GeV</text>')
o.append(f'<text class="pan" x="{L+10}" y="{T+h1+16}">(b)</text>')
o.append(f'<text class="axt" transform="translate(22,{T+h1+h2/2}) rotate(-90)" text-anchor="middle">90 / 0</text>')
o.append(f'<text class="axt" x="{L+pw/2:.0f}" y="{H-14}" text-anchor="middle">neutron E&#8342;&#8347;&#8342;  [GeV]</text>')
o.append(f'<text class="note" x="{L+pw}" y="{H-14}" text-anchor="end">bars: statistical, per-particle (clustering factor 1.0 &#177; 0.1)</text>')
print(f"{W}|{H}"); print("\n      ".join(o))
