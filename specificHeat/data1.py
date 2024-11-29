import data as dt
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os

## "only" task is to get value of c_Al and c_Cu


cu_data = pd.read_csv("./Messung Cu.csv")
startIndex = 1
timeCu = cu_data['Zeit'].tolist()[startIndex:]
timeCu = [float(i) for i in timeCu]
chCuA = cu_data['Kanal A'].tolist()[startIndex:] #mV PT100
chCuA = [float(i) for i in chCuA]
chCuB = cu_data['Kanal B'].tolist()[startIndex:] #V
chCuB = [float(i) for i in chCuB]

CuT = [float(i)/0.4 for i in chCuA]
CuP = [float(i)*1.49 for i in chCuB]

reducer = 1
timeCu = [timeCu[i] for i in range(len(timeCu)) if i%reducer==0]
CuT = [CuT[i] for i in range(len(CuT)) if i%reducer==0]
CuP = [CuP[i] for i in range(len(CuP)) if i%reducer==0]

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111)
# ax.set_xticks(tickList, labels=[x for x in tickList])
ax1 = ax.twinx()
ax.plot(timeCu, CuT, color='red')
ax1.plot(timeCu, CuP, color='blue')

ax.set_ylabel(r'$T (^{\circ}C)$', color='red')
# ax.set_ylim(top=60)
ax1.set_ylabel(r'$P (W)$', color='blue')
# ax1.set_ylim(top=(60-25)/(42.5-25) * 20)
ax.set_xlabel(r'$t (s)$')
ax.set_title('Temperature and Power over time')



plt.savefig('./tmp1.png')
os.system("kitty +kitten icat ./tmp1.png")


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
    T_prime = (T[indexG] - T[indexF])/(time[indexG] - time[indexF])
    print(f'''({P}·{dt}·{dT})/({dT**2}-{T_prime}·{F})''')
    return (P * dt * dT)/(dT**2 - T_prime*F)
M = 63.5460 ## Atomic weight (grams/mol)
m = 1532 ## grams
C_Cu = (C_tot(timeCu, CuT, CuP) - dt.C_water) * M/m
print(C_tot(timeCu, CuT, CuP))
print(C_Cu)
