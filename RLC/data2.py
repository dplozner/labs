import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit


print(f'''#3.2.1:
already fullfilled in the lab''')

data = pd.read_csv('./RLCdata.csv', delimiter = ';')
freq = data.iloc[:,0]
ang = data.iloc[:,1]
vol = data.iloc[:,2]
ang_freq = [i*2*np.pi for i in freq]

C = data.iloc[:,-3][:6]
L = data.iloc[:,-2][:6]
f = data.iloc[:,-1][:6]
ang_f = [i*2*np.pi for i in f]

print(f'''#3.2.2:
omega_0 corresponds to roughly {ang_freq[8]} from the graph
omega_0_th is equal to 1/sqrt(LC) = {1/np.sqrt(10e-3*100e-9)}
''')

def model(omega, U_in, R, L, C):
    return U_in/np.sqrt(1 + (omega*R*L + 1/(omega*C))**2)

initial_guess = [1.0, 1e3, 10e-3, 100e-9]

popt, pcov = curve_fit(model, ang_freq, vol, p0 = initial_guess)
omega_range = np.linspace(18000, 48000, 100000)
U_in_fit, R_fit, L_fit, C_fit = popt
U_out_fit = model(omega_range, *popt)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel(r'$\omega$')
ax1.set_ylabel(r'$U_{out}$(mV)', color = 'blue')

ax1.scatter(ang_freq, vol, label = r'measured points', color = 'blue')
ax1.plot(omega_range, U_out_fit, label = r'fit', color = 'red')
ax1.legend()


fig.savefig('/tmp/graph2.png')
os.system('kitty +kitten icat /tmp/graph2.png')

U_max = max(U_out_fit)
omega_0_exp = omega_range[np.where(U_out_fit == max(U_out_fit))][0]
omega_low, omega_up = omega_range[np.where(np.isclose(U_out_fit, U_max/np.sqrt(2)))]
delta_omega = omega_up -omega_low
Q_exp = omega_0_exp/delta_omega
Q_th = 1e3*np.sqrt(100e-9/10e-3)

print(f'''#3.2.3:
omega_0 from fitted graph: {omega_0_exp}
omega_0_th is equal to 1/sqrt(LC) = {1/np.sqrt(10e-3*100e-9)}
Quality factor experimental = {Q_exp}
Quality factor theoretical = {Q_th}
''')





