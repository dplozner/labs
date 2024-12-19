import data as dt
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import *
import os

## "only" task is to get value of c_Al and c_Cu


cu_data = pd.read_csv("./Messung Cu.csv")
startIndex = 1
timeCu = cu_data['Zeit'].tolist()[startIndex:]
timeCu = [float(i) for i in timeCu]
timeCu = [ufloat(i, 0.0001) for i in timeCu]
chCuA = cu_data['Kanal A'].tolist()[startIndex:] #mV PT100
chCuA = [float(i) for i in chCuA]
chCuB = cu_data['Kanal B'].tolist()[startIndex:] #V
chCuB = [float(i) for i in chCuB]

CuT = [ufloat(i, 0.1)/0.4 for i in chCuA]
CuP = [ufloat(i, 0.01)*1.49 for i in chCuB]

reducer = 1
timeCu = [timeCu[i] for i in range(len(timeCu)) if i%reducer==0]
CuT = [CuT[i] for i in range(len(CuT)) if i%reducer==0]
CuP = [CuP[i] for i in range(len(CuP)) if i%reducer==0]

indexF = 0
for index in range(len(CuP)):
    if CuP[index].nominal_value>1:
        indexF = index
index1 = 0
for index in range(len(CuP)):
    if CuP[index].nominal_value>1:
        index1 = index
        break
indexG = 0
for index in range(len(timeCu)):
    if(timeCu[indexF].nominal_value + 600 - timeCu[index].nominal_value) < 0.001:
        indexG = index
        break

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111)
# ax.set_xticks(tickList, labels=[x for x in tickList])
ax1 = ax.twinx()
ax.plot([i.nominal_value for i in timeCu], [i.nominal_value for i in CuT], color='red', alpha=0.5)
ax1.plot([i.nominal_value for i in timeCu], [i.nominal_value for i in CuP], color='blue')
ax.scatter([timeCu[index1].nominal_value, timeCu[indexF + 150].nominal_value, timeCu[indexG].nominal_value],
           [CuT[index1].nominal_value, CuT[indexF + 150].nominal_value, CuT[indexG].nominal_value], color='green')
ax.set_ylabel(r'$T (^{\circ}C)$', color='red')
# ax.set_ylim(top=60)
ax1.set_ylabel(r'$P (W)$', color='blue')
# ax1.set_ylim(top=(60-25)/(42.5-25) * 20)
ax.set_xlabel(r'$t (s)$')
ax.set_title('Temperature and Power over time')



plt.savefig('./tmp1.png')
os.system("kitty +kitten icat ./tmp1.png")


def C_tot(time, T, P):
    P = ufloat(np.mean(chCuB[index1:indexF])*1.49, np.std(chCuB[index1:indexF])) ## Watt
    dt = time[indexF] - time[index1] ## secondi
    dT = T[indexG] - T[index1] ## °C
    F_list = [i*(time[2]-time[1]) for i in T[index1:indexG]] ## integral naturally expressed
    F = 0
    for i in F_list:
        F += i
    T_prime = (T[indexG] - T[indexF + 150])/(time[indexG] - time[indexF + 150])
    print(f'''({P}·{dt}·{dT})/({dT**2}-{T_prime}·{F})''')
    print(
            f'''power: {P}
            time using power: {dt}
            temperature differential {dT}
            slope of T(t) {T_prime}
            integral of T(t) {F}
            ''')
    return (P * dt * dT)/(dT**2 - T_prime*F)
M = 63.5460 ## Atomic weight (grams/mol)
m = 1532 ## grams
C_Cu = (C_tot(timeCu, CuT, CuP) ) * M/m
print(C_tot(timeCu, CuT, CuP).nominal_value, C_tot(timeCu, CuT, CuP).std_dev)
print(C_Cu.nominal_value, C_Cu.std_dev)
