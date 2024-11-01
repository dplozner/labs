import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os

## TO FIND: ---> lambda (p, k) = b ((k - k_0) + (p - p_central)/B)
##      step 1:     lambda (k) = b(k-k_0) --> curve fit find parameters k_0 and b
##      step 2:     p (k) = B * k + displacement --> curve fit find parameters B and displacement
##      step 3:     for k in Balmer series: k => lambda(p(k), k) ---> RESULTS WE LOOK FOR!

wlk = [388.8,
       447.1,
       471.3,
       492.1,
       501.5,
       504.7,
       587.5,
       667.8]
k = [218,
     356,
     419,
     473,
     498,
     506,
     716,
     926]
def wl(k, k_0, b):
    return b*(k-k_0)
initial_guess = [68, 0.37]

popt, pcov = curve_fit(wl, k, wlk, p0 = initial_guess)
k_0_uncertainty = np.sqrt(pcov[0, 0])
b_uncertainty = np.sqrt(pcov[1, 1])
print(popt, k_0_uncertainty, b_uncertainty) ##yields -780 and 0.39
k_ = np.linspace(200,1000,800)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(k, wlk, color = 'red')
ax.plot(k_, wl(k_, *popt), color = 'blue')

ax.set_xlabel('Rotation position k')
ax.set_ylabel('Wavelength [nm]')
ax.set_title(r'dependence of $\lambda_c$ on k')

plt.savefig('./wlk.png')
print('#2.6:')
os.system("kitty +kitten icat ./wlk.png")


