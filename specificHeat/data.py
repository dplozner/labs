import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os

## "only" task is to get value of c_Al and c_Cu


water_data = pd.read_csv("./Messung Wasser geschlossen.csv")
startIndex = 700
timeW = water_data['Zeit'].tolist()[startIndex:]
timeW = [float(i) for i in timeW]
chwA = water_data['Kanal A'].tolist()[startIndex:] #mV PT100
chwA = [float(i) for i in chwA]
chwB = water_data['Kanal B'].tolist()[startIndex:] #V
chwB = [float(i) for i in chwB]

wT = [float(i)/0.4 for i in chwA]
wP = [float(i)*1.49 for i in chwB]

reducer = 1
timeW = [timeW[i] for i in range(len(timeW)) if i%reducer==0]
wT = [wT[i] for i in range(len(wT)) if i%reducer==0]
wP = [wP[i] for i in range(len(wP)) if i%reducer==0]

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111)
# ax.set_xticks(tickList, labels=[x for x in tickList])
ax1 = ax.twinx()
ax.plot(timeW, wT, color='red')
ax1.plot(timeW, wP, color='blue')

ax.set_ylabel(r'$T (^{\circ}C)$', color='red')
# ax.set_ylim(top=60)
ax1.set_ylabel(r'$P (W)$', color='blue')
# ax1.set_ylim(top=(60-25)/(42.5-25) * 20)
ax.set_xlabel(r'$t (s)$')
ax.set_title('Temperature and Power over time')



plt.savefig('./tmp0.png')
os.system("kitty +kitten icat ./tmp0.png")


def C_tot(time, T):
    indexF = 0
    for index in range(len(wP)):
        if wP[index]>1:
            indexF = index
    index1 = 0
    for index in range(len(wP)):
        if wP[index]>1:
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

    return (P * dt * dT)/(dT**2 - T_prime*F)
C_water = C_tot(timeW, wT)
print(C_water)
