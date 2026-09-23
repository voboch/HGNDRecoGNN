import numpy as np
m=0.93827
for T in (2.5,3.8):
    E=T+m; p=np.sqrt(E*E-m*m)
    s=2*m*(T+2*m); rs=np.sqrt(s)
    ylab=0.5*np.log((E+p)/(E-p))
    print(f"T_beam = {T}A GeV  ->  sqrt(s_NN) = {rs:.3f} GeV,  y_beam(lab) = {ylab:.4f},  y_cm shift = {ylab/2:.4f}")
print("\ndataset label says 2.87 GeV")
