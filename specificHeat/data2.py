import data as dt
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os

## "only" task is to get value of c_Al and c_Al


cu_data = pd.read_csv("./Messung Al.csv")
startIndex = 30
timeAl = cu_data['Zeit'].tolist()[startIndex:]
timeAl = [float(i) for i in timeAl]
chAlA = cu_data['Kanal A'].tolist()[startIndex:] #mV PT100
chAlA = [float(i) for i in chAlA]
chAlB = cu_data['Kanal B'].tolist()[startIndex:] #V
chAlB = [float(i) for i in chAlB]

AlT = [float(i)/0.4 for i in chAlA]
AlP = [float(i)*1.49 for i in chAlB]

reducer = 1
timeAl = [timeAl[i] for i in range(len(timeAl)) if i%reducer==0]
AlT = [AlT[i] for i in range(len(AlT)) if i%reducer==0]
AlP = [AlP[i] for i in range(len(AlP)) if i%reducer==0]

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111)
# ax.set_xticks(tickList, labels=[x for x in tickList])
ax1 = ax.twinx()
ax.plot(timeAl, AlT, color='red')
ax1.plot(timeAl, AlP, color='blue')

ax.set_ylabel(r'$T (^{\circ}C)$', color='red')
# ax.set_ylim(top=60)
ax1.set_ylabel(r'$P (W)$', color='blue')
# ax1.set_ylim(top=(60-25)/(42.5-25) * 20)
ax.set_xlabel(r'$t (s)$')
ax.set_title('Temperature and Power over time')



plt.savefig('./tmp2.png')
os.system("kitty +kitten icat ./tmp2.png")


def C_tot(time, T, P):
    indexF = 0
    for index in range(len(P)):
        if P[index]>1:
            indexF = index
    index1 = 0
    for index in range(len(P)):
        if P[index]>1:
            index1 = index
            break
    indexG = 0
    for index in range(len(time)):
        if(time[index1] + 600 - time[index]) < 0.001:
            indexG = index


    P = 20 ## Watt
    dt = time[indexG] - time[index1] ## secondi
    dT = T[indexG] - T[index1] ## °C
    F = sum([i*(time[2]-time[1]) for i in T[index1:indexG]]) ## integral naturally expressed
    def model(a, b, x):
        return a*x+b
    initial_guess = [-0.001, 45]
    popt, pcov = curve_fit(model, time[indexF:indexG], T[indexF:indexG], p0=initial_guess)
    print(popt)
    T_prime = -0.0011 #(T[indexG] - T[indexF])/(time[indexG] - time[indexF]) ##LOOOL##
    print(f'''({P}·{dt}·{dT})/({dT**2}-{T_prime}·{F})''')
    return [(P * dt * dT)/(dT**2 - T_prime*F), popt, time[indexF:indexG]]
M = 26.98153860 ## Atomic weight (grams/mol)
m = 466 ## grams
C_Al = (C_tot(timeAl, AlT, AlP)[0] - dt.C_water) * M/m
print(C_tot(timeAl, AlT, AlP)[0])
print(C_Al)

# b, a = C_tot(timeAl, AlT, AlP)[1]
# t = C_tot(timeAl, AlT, AlP)[2]
# ax.plot(t, [i*float(-0.0011)+float(b-1) for i in t])
#
# plt.savefig('./tmp2.png')
# os.system("kitty +kitten icat ./tmp2.png")
