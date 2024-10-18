import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os 
from scipy.optimize import curve_fit


data = pd.read_csv('./RLCdata.csv', delimiter = ';')[:6]
C_m = data.iloc[:,-3]
L_m = data.iloc[:,-2]
frq_m = data.iloc[:,-1]
ang_frq = [i*2*np.pi for i in frq_m]


# L_vals = np.linspace(1e-3, 20e-3, 100)  # Inductance from 1 mH to 20 mH
# C_vals = np.linspace(10e-9, 1000e-9, 100)  # Capacitance from 10 nF to 1000 nF
#
# # Create a meshgrid of L and C values
# L, C = np.meshgrid(L_vals, C_vals)
#
# # Calculate the resonant frequency omega_0 for each L and C pair
# omega_0 = 1 / np.sqrt(L * C)
#
# # Logarithmic scaling for omega_0 to make the gradient more visible
# log_omega_0 = np.log10(omega_0)
#
# # Plotting
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(L*1e3, C*1e9, log_omega_0, 100, cmap='viridis')
# cbar = plt.colorbar(contour)
# cbar.set_label(r'$\log_{10}(\omega_0)$ (log(rad/s))')
#
# # Labels and title
# plt.xlabel('Inductance L (mH)')
# plt.ylabel('Capacitance C (nF)')
# plt.title(r'Logarithmic Scale of $\omega_0$ as a function of L and C')
#
# # Show plot
# plt.grid(False)

from mpl_toolkits.mplot3d import Axes3D

# Define ranges for L and C (in H and F, respectively)
L_vals = np.linspace(1e-3, 20e-3, 100)  # Inductance from 1 mH to 20 mH
C_vals = np.linspace(10e-9, 1000e-9, 100)  # Capacitance from 10 nF to 1000 nF

# Create a meshgrid of L and C values
L, C = np.meshgrid(L_vals, C_vals)

# Calculate the resonant frequency omega_0 for each L and C pair
omega_0 = 1 / np.sqrt(L * C)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


# Plot the surface
ax.plot_surface(L*1e3, C*1e9, omega_0, alpha = 0.7, cmap = 'Blues_r')

# Add a color bar
# cbar = fig.colorbar(surf, ax=ax, pad=0.1)
# cbar.set_label(r'$\omega_0$ (rad/s)')

# Labels and title
ax.set_xlabel('Inductance L (mH)')
ax.set_ylabel('Capacitance C (nF)')
ax.set_zlabel(r'$\omega_0$ (rad/s)')
ax.set_title(r'Resonant Frequency $\omega_0$ as a function of L and C')

plt.savefig('/tmp/RLCgraph3.png')
os.system("kitty +kitten icat /tmp/RLCgraph3.png")
