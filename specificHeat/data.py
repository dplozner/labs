import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import *
import os

## "only" task is to get value of c_Al and c_Cu


water_data = pd.read_csv("./Messung Wasser geschlossen.csv")
startIndex = 700
timeW = water_data['Zeit'].tolist()[startIndex:]
timeW = [ufloat(float(i), 0.0001) for i in timeW]
chwA = water_data['Kanal A'].tolist()[startIndex:] #mV PT100
chwA = [float(i) for i in chwA]
chwB = water_data['Kanal B'].tolist()[startIndex:] #V
chwB = [float(i) for i in chwB]

wT = [ufloat(i, 0.001)/0.4 for i in chwA]
wP = [ufloat(i, 0.001)*1.49 for i in chwB]

reducer = 1
timeW = [timeW[i] for i in range(len(timeW)) if i%reducer==0]
wT = [wT[i] for i in range(len(wT)) if i%reducer==0]
wP = [wP[i] for i in range(len(wP)) if i%reducer==0]


indexF = 0
for index in range(len(wP)):
    if wP[index].nominal_value>1:
        indexF = index
index1 = 0
for index in range(len(wP)):
    if wP[index].nominal_value>1:
        index1 = index
        break
indexG = 0
for index in range(len(timeW)):
    if(timeW[indexF].nominal_value + 600 - timeW[index].nominal_value) < 0.001:
        indexG = index
        break
fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111)
# ax.set_xticks(tickList, labels=[x for x in tickList])
ax1 = ax.twinx()
ax.plot([i.nominal_value for i in timeW], [i.nominal_value for i in wT], color='red', alpha=0.5)
ax1.plot([i.nominal_value for i in timeW], [i.nominal_value for i in wP], color='blue')

# ax.scatter([timeW[index1].nominal_value, timeW[indexF].nominal_value, timeW[indexG].nominal_value],
#           [wT[index1].nominal_value, wT[indexF].nominal_value, wT[indexG].nominal_value], color='green')
# ax.xticks(np.append(plt.gca().get_xticks(), timeW[index1].nominal_value), labels=[*plt.gca().get_xticklabels(), r'$t_0$'])
ax.set_ylabel(r'$T (^{\circ}C)$', color='red')
# ax.set_ylim(top=60)
ax1.set_ylabel(r'$P (W)$', color='blue')
# ax1.set_ylim(top=(60-25)/(42.5-25) * 20)
ax.set_xlabel(r'$t (s)$')
ax.set_title('Temperature and Power over time')



def C_tot(time, T):

    P = ufloat(np.mean(chwB[index1:indexF])*1.49, np.std(chwB[index1:indexF])) ## Watt sigma_W = 0.1 W
    dt = time[indexF] - time[index1] ## secondi sigma_t = 0.001
    dT = T[indexG] - T[index1 + 150] ## °C sigma_T = 1°C
    F_list = [i*(time[2]-time[1]) for i in T[index1:indexG]] ## integral naturally expressed
    F = 0
    for i in F_list:
        F += i
    F += 20000
    T_prime = (T[indexG] - T[indexF + 950])/(time[indexG] - time[indexF + 950])
    print(
            f'''
            power: {P}
            time using power: {dt}
            temperature differential {dT}
            slope of T(t) {T_prime}
            integral of T(t) {F}
            ''')
    return (P * dt * dT)/(dT**2 - T_prime*F)
C_water = C_tot(timeW, wT)
print(C_water.nominal_value, C_water.std_dev)

plt.savefig('./tmp0.png')
os.system("kitty +kitten icat ./tmp0.png")
