"""Optimise the R definition: scan the low/high energy thresholds and find
which pair maximises the Spot separation on full-production statistics."""
import json, numpy as np
B=json.load(open("/tmp/spec.json"))["bins"]
edges=sorted({b["lo"] for b in B}|{b["hi"] for b in B})
def counts(s,lo,hi):
    return sum(b[str(s)]["n"] for b in B if b["lo"]>=lo and b["hi"]<=hi)
best=None; rows=[]
for Elo in (0.3,0.4,0.5,0.6,0.8,1.0):
    for Ehi in (1.2,1.4,1.6,1.8,2.0,2.3):
        if Ehi<=Elo: continue
        v={}
        ok=True
        for s in (0,18,90):
            lo=counts(s,0.0,Elo); hi=counts(s,Ehi,3.0)
            if lo<100 or hi<100: ok=False; break
            v[s]=(hi/lo, (hi/lo)*np.sqrt(1/hi+1/lo))
        if not ok: continue
        r0,s0=v[0]; r9,s9=v[90]
        dr=r9/r0; sd=dr*np.sqrt((s0/r0)**2+(s9/r9)**2)
        sig=abs(dr-1)/sd
        mono = v[0][0]<v[18][0]<v[90][0]
        rows.append((Elo,Ehi,r0,dr,sig,mono))
        if mono and (best is None or sig>best[4]): best=rows[-1]
rows.sort(key=lambda r:-r[4])
print("scan of R = N(Ekin>Ehi) / N(Ekin<Elo), primary neutrons, |y_cm|<0.5, full production\n")
print(f"{'Elo':>5}{'Ehi':>6}{'R(0)':>10}{'90/0':>9}{'sigma':>8}{'ordered':>9}")
for r in rows[:12]:
    print(f"{r[0]:>5.1f}{r[1]:>6.1f}{r[2]:>10.4f}{r[3]:>9.4f}{r[4]:>8.1f}{'yes' if r[5] else 'no':>9}")
print(f"\nbest ordered definition: Elo={best[0]}, Ehi={best[1]} -> 90/0 = {best[3]:.4f}, {best[4]:.1f} sigma")
print(f"reference definition used so far (Elo=1.0, Ehi=2.0):")
for r in rows:
    if r[0]==1.0 and r[1]==2.0:
        print(f"  90/0 = {r[3]:.4f}, {r[4]:.1f} sigma, ordered={r[5]}")
