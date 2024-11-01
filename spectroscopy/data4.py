
import numpy as np

# Constants and measured values
b = 0.39  # slope from calibration in nm per k-unit
k0 = -780  # intercept from calibration in k-units
sigma_b = 0.0026  # uncertainty in b (nm per k-unit)
sigma_k0 = 8.57  # uncertainty in k0 (k-units)
sigma_k = 1  # absolute uncertainty in k

# Measured k-values and corresponding Balmer wavelengths (example values)
k_Balmer = [324, 457, 898]  # knob values for Balmer lines
lambda_Balmer = [433.07, 485.24, 658.22]  # experimental wavelengths in nm

# Calculate uncertainties in lambda for each k-value
lambda_uncertainties = []

for k in k_Balmer:
    # Each term in the uncertainty formula
    term_b = (k - k0) ** 2 * sigma_b ** 2
    term_k0 = b ** 2 * sigma_k0 ** 2
    term_k = b ** 2 * sigma_k ** 2
    
    # Total uncertainty in lambda
    sigma_lambda = np.sqrt(term_b + term_k0 + term_k)
    lambda_uncertainties.append(float(sigma_lambda))

# Display results
for val in lambda_uncertainties:
    print(f'{val:.2f}')
