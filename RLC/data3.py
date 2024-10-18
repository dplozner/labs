import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os


frq = [1e3, 5e3, 10e3]
ang_freq = [i*2*np.pi for i in frq]
phi_deg = [-35.1, 2.85, 56.8]
phi_rad = [i*2*np.pi/360 for i in phi_deg]

def model_phi(omega, R, L, C):
    return np.arctan((omega*L - 1/(omega*C))/R)

be_my_guess = [1e3, 10e-3, 100e-9]
# popt, pcov = curve_fit(model_phi, ang_freq, phi_rad, p0 = be_my_guess)
ang_frq_range = np.linspace(2000, 70000, 1000)
phi_fit = model_phi(ang_frq_range, *be_my_guess)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(ang_freq, phi_rad, color = 'blue', label = 'measured points')
ax.plot(ang_frq_range, phi_fit, color = 'red', label = 'curve fit')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$\varphi$')
ax.legend()
fig.savefig('/tmp/RLCgraph2.png')
os.system('kitty +kitten icat /tmp/RLCgraph2.png')

