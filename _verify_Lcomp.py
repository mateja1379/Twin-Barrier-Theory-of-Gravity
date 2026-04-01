import numpy as np
k = 1.0
print("Action integral fraction captured by L_comp:")
for L_comp in [1,2,3,5,10,20]:
    frac = (1 - np.exp(-4*k*L_comp)) / (1 - np.exp(-4*k*20))
    e2ky_max = np.exp(2*k*L_comp)
    print(f"  L_comp={L_comp:2d}: {frac*100:.6f}% captured, e^(2kL)={e2ky_max:.2e}")
