import data as dt
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import *
import os

## "only" task is to get value of c_Al and c_Al


cu_data = pd.read_csv("./Messung Al.csv")
startIndex = 30
timeAl = cu_data['Zeit'].tolist()[startIndex:]
timeAl = [float(i) for i in timeAl]
timeAl = [ufloat(i, 0.0001) for i in timeAl]
chAlA = cu_data['Kanal A'].tolist()[startIndex:] #mV PT100
chAlA = [float(i) for i in chAlA]
chAlB = cu_data['Kanal B'].tolist()[startIndex:] #V
chAlB = [float(i) for i in chAlB]

AlT = [ufloat(i, 0.1)/0.4 for i in chAlA]
AlP = [ufloat(i, 0.01)*1.49 for i in chAlB]

reducer = 1
timeAl = [timeAl[i] for i in range(len(timeAl)) if i%reducer==0]
AlT = [AlT[i] for i in range(len(AlT)) if i%reducer==0]
AlP = [AlP[i] for i in range(len(AlP)) if i%reducer==0]


indexF = 0
for index in range(len(AlP)):
    if AlP[index].nominal_value>1:
        indexF = index
index1 = 0
for index in range(len(AlP)):
    if AlP[index].nominal_value>1:
        index1 = index
        break
indexG = 0
for index in range(len(timeAl)):
    if(timeAl[indexF].nominal_value + 600 - timeAl[index].nominal_value) < 0.001:
        indexG = index
        break

fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(111)
# ax.set_xticks(tickList, labels=[x for x in tickList])
ax1 = ax.twinx()
ax.plot([i.nominal_value for i in timeAl], [i.nominal_value for i in AlT], color='red', alpha=0.5)
ax1.plot([i.nominal_value for i in timeAl], [i.nominal_value for i in AlP], color='blue')
ax.scatter([timeAl[index1].nominal_value, timeAl[indexF].nominal_value, timeAl[indexG].nominal_value],
           [AlT[index1].nominal_value, AlT[indexF].nominal_value, AlT[indexG].nominal_value], color='green')
ax.set_ylabel(r'$T (^{\circ}C)$', color='red')
# ax.set_ylim(top=60)
ax1.set_ylabel(r'$P (W)$', color='blue')
# ax1.set_ylim(top=(60-25)/(42.5-25) * 20)
ax.set_xlabel(r'$t (s)$')
ax.set_title('Temperature and Power over time')



plt.savefig('./tmp2.png')
os.system("kitty +kitten icat ./tmp2.png")


def C_tot(time, T, P):

    P = ufloat(np.mean(chAlB[index1:indexF])*1.49, np.std(chAlB[index1:indexF])) ## Watt
    dt = time[indexF] - time[index1] ## secondi
    dT = T[indexG] - T[index1] ## °C
    F_list = [i*(time[2]-time[1]) for i in T[index1:indexG]] ## integral naturally expressed
    F = 0
    for i in F_list:
        F += i
    # def model(a, b, x):
    #     return a*x+b
    # initial_guess = [-0.001, 45]
    # popt, pcov = curve_fit(model, time[indexF:indexG], T[indexF:indexG], p0=initial_guess)
    # print(popt)
    T_prime = (T[indexG] - T[indexF])/(time[indexG] - time[indexF]) 
    # print(f'''({P}·{dt}·{dT})/({dT**2}-{T_prime}·{F})''')
    print(
            f'''power: {P}
            time using power: {dt}
            temperature differential {dT}
            slope of T(t) {T_prime}
            integral of T(t) {F}
            ''')
    return [(P * dt * dT)/(dT**2 - T_prime*F), 'porcodio', 'diocane']
M = 26.98153860 ## Atomic weight (grams/mol)
m = 466 ## grams
C_Al = (C_tot(timeAl, AlT, AlP)[0] - dt.C_water) * M/m
print(C_tot(timeAl, AlT, AlP)[0].nominal_value, C_tot(timeAl, AlT, AlP)[0].std_dev)
print(C_Al.nominal_value, C_Al.std_dev)

# b, a = C_tot(timeAl, AlT, AlP)[1]
# t = C_tot(timeAl, AlT, AlP)[2]
# ax.plot(t, [i*float(-0.0011)+float(b-1) for i in t])
#
# plt.savefig('./tmp2.png')
# os.system("kitty +kitten icat ./tmp2.png")
