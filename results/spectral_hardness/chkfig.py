import json, numpy as np, re
B=[b for b in json.load(open("/tmp/spec.json"))["bins"] if b["hi"]<=3.0]
vals=[b[k]["n"]/b[k]["ev"]/(b["hi"]-b["lo"]) for b in B for k in ("0","18","90")]
dmin,dmax=min(vals),max(vals)
ylo=10**np.floor(np.log10(dmin/1.6)); yhi=10**np.ceil(np.log10(dmax*1.3))
print("data      : %.4f .. %.4f" % (dmin,dmax))
print("new axis  : %g .. %g   contains all: %s" % (ylo,yhi,ylo<dmin and yhi>dmax))
raw=open("/tmp/figP1.out").read().split("\n",1)
W,H=[int(x) for x in raw[0].split("|")]
svg=raw[1]
T,h1,h2=16,250,100
ys=[float(m) for m in re.findall(r'cy="([\d.]+)"',svg)]
for pl in re.findall(r'points="([^"]+)"',svg):
    for pair in pl.split():
        if "," in pair:
            ys.append(float(pair.split(",")[1]))
lo,hi=T-2,T+h1+h2+2
out=[y for y in ys if not (lo<=y<=hi)]
print("\nfigure vertical extent in use: %d .. %d  (viewBox height %d)" % (lo,hi,H))
print("plotted y coords: %d   outside the two panels: %d" % (len(ys),len(out)))
if out: print("  offenders:", sorted(out)[:5])
else:   print("  -> all marks and step lines are inside the panels")
