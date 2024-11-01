import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from data import *

k1 = [690,
      700,
      710,
      720,
      730,
      740]
pk = [1440,
      1180,
      950,
      665,
      413,
      177]

def p(k, B):
    return B*(587.5/popt[1] - (k-popt[0])) + 719
# initial_guess1 = [-25]
def pfx(k):
    return -25.58*k + (25.58*(587.5/popt[1] + popt[0]) + 719)

popt2, pcov2 = curve_fit(p, k1, pk, p0 = -25)
print(popt2) ## yields -25 
k1_ = np.linspace(680,750,600)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(k1, pk, color = 'red')
ax.plot(k1_, pfx(k1_), color = 'blue')

ax.set_xlabel('Rotation position k')
ax.set_ylabel('Pixel position')
ax.set_title(r'dependence of p on k')

plt.savefig('./pk.png')
print('#2.6:')
os.system("kitty +kitten icat ./pk.png")



p1_ = pfx(k1_)
l1_ = wl(k1_, *popt)

def lmbd(p,k):
    return popt[1]*((k-popt[0])+(p-719)/25.58)




