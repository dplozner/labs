import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from data import *
from data2 import *


k_Balmer = [324, #blue
            457, #light blue
            898] #red

b = popt[1]
k_0 = popt[0]
p_0 = 719
p_ = [int(p(k, *popt2)) for k in k_Balmer]
# print(p_)

def lambda_(p,k):
    return b*((k-k_0) + (p-p_0)/B)

print([float(lmbd(pfx(s),s)) for s in k_Balmer])
print(wl(k_Balmer, *popt)) ## yields [-25.43142857 746.62097892] some of those are very precisely correct


# Constants from fit
sigma_b = np.sqrt(pcov[1, 1])
sigma_k0 = np.sqrt(pcov[0, 0])
print(f'''sigma_b = {sigma_b} nm
sigma_k0 = {sigma_k0}''')
# Calculate wavelengths and uncertainties for each k value in the Balmer series
lambda_Balmer = [wl(k, *popt) for k in k_Balmer]
sigma_lambda = [
    np.sqrt((k - k_0)**2 * sigma_b**2 + b**2 * sigma_k0**2) for k in k_Balmer
]

# Print results
for i, k_val in enumerate(k_Balmer):
    print(f"k: {k_val}, λ: {lambda_Balmer[i]:.2f} ± {sigma_lambda[i]:.2f} nm")


# Given measurements and uncertainties
n_values = [5, 4, 3]  # principal quantum numbers

# Constants
m = 2  # lower energy level for Balmer series
sigma_lambda = [4.42,4.65,5.51]
# Convert wavelengths to meters for Rydberg calculation
wavelengths_m = [wl * 1e-9 for wl in lambda_Balmer]  # in meters
uncertainties_m = [unc * 1e-9 for unc in sigma_lambda]  # uncertainties in meters

# Calculate Rydberg constant (R_H) for each wavelength
R_H_values = []
R_H_uncertainties = []
for i, wl in enumerate(wavelengths_m):
    n = n_values[i]
    term_diff = (1 / m**2) - (1 / n**2)
    R_H = 1 / (wl * term_diff)
    R_H_values.append(R_H)
    
    # Uncertainty in R_H
    dR_H_dlambda = -1 / (wl**2 * term_diff)
    R_H_uncertainty = abs(dR_H_dlambda * uncertainties_m[i])
    R_H_uncertainties.append(R_H_uncertainty)

# Calculate average R_H and combined uncertainty
R_H_avg = np.mean(R_H_values)
R_H_avg_uncertainty = np.sqrt(sum(unc**2 for unc in R_H_uncertainties) / len(R_H_uncertainties))

# Predicted wavelength for n=6
n_pred = 6
term_diff_pred = (1 / m**2) - (1 / n_pred**2)
lambda_pred = 1 / (R_H_avg * term_diff_pred)  # in meters
lambda_pred_nm = lambda_pred * 1e9  # convert to nm
print('#2.7')
print(f'''Rydberg constant: {R_H_avg:.2f} +/- {R_H_avg_uncertainty:.2f}''')

# Calculate the partial derivative of lambda with respect to R_H
partial_lambda_R_H = -1 / (R_H_avg**2 * term_diff_pred)

# Calculate the uncertainty in the predicted wavelength
sigma_lambda_pred = abs(partial_lambda_R_H * R_H_avg_uncertainty)

print(f'''lambda purple: {lambda_pred_nm:.2f} +/- {sigma_lambda_pred * 1e9:.2f}''')

print(f'''The limit of the Balmer series is: {4e9/R_H_avg:.2f}''')

