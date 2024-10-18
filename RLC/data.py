import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os

# USING AMPLITUDES CUZ WHY NOT????
U_in_10hz = pd.read_csv('./data/ALL0001/F0001CH1.CSV').iloc[:,-2]
U_out_10hz = pd.read_csv('./data/ALL0001/F0001CH2.CSV').iloc[:,-2]
time10 = pd.read_csv('./data/ALL0001/F0001CH1.CSV').iloc[:,3]
max_U_in_10hz = max(U_in_10hz)
max_U_out_10hz = max(U_out_10hz)


U_in_100hz = pd.read_csv('./data/ALL0002/F0002CH1.CSV').iloc[:,-2]
U_out_100hz = pd.read_csv('./data/ALL0002/F0002CH2.CSV').iloc[:,-2]
time100 = pd.read_csv('./data/ALL0002/F0002CH1.CSV').iloc[:,3]
max_U_in_100hz = max(U_in_100hz)
max_U_out_100hz = max(U_out_100hz)


U_in_1khz = pd.read_csv('./data/ALL0003/F0003CH1.CSV').iloc[:,-2]
U_out_1khz = pd.read_csv('./data/ALL0003/F0003CH2.CSV').iloc[:,-2]
time1k = pd.read_csv('./data/ALL0003/F0003CH1.CSV').iloc[:,3]
max_U_in_1khz = max(U_in_1khz)
max_U_out_1khz = max(U_out_1khz)


U_in_10khz = pd.read_csv('./data/ALL0005/F0005CH1.CSV').iloc[:,-2]
U_out_10khz = pd.read_csv('./data/ALL0005/F0005CH2.CSV').iloc[:,-2]
time10k = pd.read_csv('./data/ALL0005/F0005CH1.CSV').iloc[:,3]
max_U_in_10khz = max(U_in_10khz)
max_U_out_10khz = max(U_out_10khz)


U_in_100khz = pd.read_csv('./data/ALL0006/F0006CH1.CSV').iloc[:,-2]
U_out_100khz = pd.read_csv('./data/ALL0006/F0006CH2.CSV').iloc[:,-2]
time100k = pd.read_csv('./data/ALL0006/F0006CH1.CSV').iloc[:,3]
max_U_in_100khz = max(U_in_100khz)
max_U_out_100khz = max(U_out_100khz)

# Data (same as before)
UOI = [max_U_out_10hz/max_U_in_10hz, max_U_out_100hz/max_U_in_100hz, max_U_out_1khz/max_U_in_1khz, max_U_out_10khz/max_U_in_10khz, max_U_out_100khz/max_U_in_100khz]
f = [10, 100, 1e3, 10e3, 100e3]
ang_frq = [i*2*np.pi for i in f]

ang_frequencies = np.linspace(0, 1e6, 100000) * 2*np.pi
def UOI_th(omega):
    return 1/(np.sqrt(omega**2 * 1e3**2 * 100e-9**2 + 1))
# def UOI_th_slope(omega):
#     return -(omega*1e3**2*100e-9**2)/(np.sqrt(omega**2 * 1e3**2 * 100e-9**2 + 1))

# Scale UOI for better fitting
UOI_scaled = [i*10 for i in UOI]

# plt.plot(ang_frq, UOI_scaled, 'o', label='Data Points')
# plt.plot(ang_frequencies, UOI_th(ang_frequencies), color='r', label = 'theoretical function')
# plt.axhline(y=1/np.sqrt(2), color='green', linestyle='--', label=r'$\frac{1}{\sqrt{2}}$')
# plt.xscale('log')
#
# # Add labels and legend
# plt.title('#3.1:')
# plt.xlabel('$\log{\omega}$')
# plt.ylabel('Amplitude Ratio')
# plt.legend()
#
# # Save and display the plot
# plt.savefig('/tmp/transferFx.png')
# os.system("echo '#3.1'")
# os.system("kitty +kitten icat /tmp/transferFx.png")
# os.system("rm /tmp/transferFx.png")

print(f'''#3.1:
partial result of the graph of 3.5''')

print(f'''#3.2: from the graph:
---> omega_gr = ~~{10e3}
---> omega_gr_th = {1/(1e3*100e-9)}''')

print(f'''#3.3:
the curve seems to have the maximal slope precisely where the graph intersects the line y = 1/sqrt(2)
for very small omega max_U_out/max_U_in = 1
for very big omega max_U_out/max_U_in = 0''')

print(f'''#3.4:
the slope of the graph is roughly {(UOI[3]-UOI[2])/np.log(f[3]/f[2])}
the slope of the theoretical graph is equal to: {(UOI_th(f[3])-UOI_th(f[2]))/(np.log(f[3]/f[2]))}''')



def findPhase(ch1, ch2, time, max_in, max_out):
    t0 = time[time == 0].index[0]  
    t0 = t0 + 5
    time = time[t0:]
    ch1 = ch1[t0:]
    ch2 = ch2[t0:]
    t1 = -1
    t2 = -1

    for t in range(len(time)):
        if abs(ch1.iloc[t] - max_in) < 0.01:  
            t1 = time.iloc[t]  
            break
            
    for t in range(len(time)):
        if abs(ch2.iloc[t] - max_out) < 0.01:  
            t2 = time.iloc[t]  
            break
        
    return t1-t2

phi10 = findPhase(U_in_10hz, U_out_10hz, time10, max_U_in_10hz, max_U_out_10hz)
phi100 = findPhase(U_in_100hz, U_out_100hz, time100, max_U_in_100hz, max_U_out_100hz)
phi1k = findPhase(U_in_1khz, U_out_1khz, time1k, max_U_in_1khz, max_U_out_1khz)
phi10k = findPhase(U_in_10khz, U_out_10khz, time10k, max_U_in_10khz, max_U_out_10khz)
phi100k = findPhase(U_in_100khz, U_out_100khz, time100k, max_U_in_100khz, max_U_out_100khz)
phi = [phi10, phi100, phi1k, phi10k, phi100k]
phi = [phi[i]*ang_frq[i] for i in range(len(phi))]
phi = [i if i<np.pi else i-2*np.pi for i in phi]


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(ang_frq, UOI_scaled, 'o', label='Data points')
ax.plot(ang_frequencies, UOI_th(ang_frequencies), color='r', label = 'Thor. prediction')
ax.axhline(y=1/np.sqrt(2), color='green', linestyle='--', label=r'$\frac{U_{out}}{U_{in}} = \frac{1}{\sqrt{2}}$')
ax.set_xscale('log')
ax.set_title(r'(Transfer func./ phase shift $U_{in}$-$U_{out}$) over angular frequency')

# Add labels and legend
ax.set_xlabel(r'Angular frequency ($\log{\omega}$)')
ax.set_ylabel(r'Amplitude ratio ($\frac{U_{out}}{U_{in}}$)', color = 'red')
ax.legend(bbox_to_anchor = [0.65, 0.6])

ax2 = ax.twinx()
ax2.plot(ang_frq, phi, 'o', color='#a502f7')
ax2.set_ylabel(r'Phase shift($\varphi$)', color = '#a502f7')

ax2.tick_params(axis='y')
fig.tight_layout()
plt.savefig('/tmp/transferFx.png')
plt.show()
# os.system("echo '#3.5:'")
# os.system("kitty +kitten icat /tmp/transferFx.png")
print(f'''
the phase shift at the cut-off frequency is very small (0)''')

I10 = U_in_10hz-U_out_10hz
max_I_10 = max(I10)
I100 = U_in_100hz-U_out_100hz
max_I_100 = max(I100)
I1k = U_in_1khz-U_out_1khz
max_I_1k = max(I1k)
I10k = U_in_10khz-U_out_10khz
max_I_10k = max(I10k)
I100k = U_in_100khz-U_out_100khz
max_I_100k = max(I100k)

dt = [findPhase(I10, U_out_10hz, time10, max_I_10, max_U_out_10hz),findPhase(I100, U_out_100hz, time100, max_I_100, max_U_out_100hz),findPhase(I1k, U_out_1khz, time1k, max_I_1k, max_U_out_1khz),findPhase(I10k, U_out_10khz, time10k, max_I_10k, max_U_out_10khz),findPhase(I100k, U_out_100khz, time100k, max_I_100k, max_U_out_100khz)]

phi2 = [dt[i]*ang_frq[i] for i in range(len(dt))] 
phi2 = [float(i* 360/(2*np.pi)) for i in phi2]

print(f'''#3.6:
the phase shift between I_C and U_C doesn't seem to be constant as it should, maybe there are mistakes in the data collection:
here are the results:
{phi2}''')

print('for f = 100Hz')
# Parameters
fs = 1/0.00002  # Sampling frequency
t = time1k

# Create two signals with a phase difference
f = 1e3 # Frequency of the signals
signal1 = I1k
signal2 = U_out_1khz

# Compute the FFT
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2)

# Get the frequency bins
frequencies = np.fft.fftfreq(len(t), 1/fs)

# Get the phase of each signal
phase_signal1 = np.angle(fft_signal1)
phase_signal2 = np.angle(fft_signal2)

# Find the phase difference at the frequency of interest
# Assuming we are interested in the first frequency component
index = np.argmax(np.abs(fft_signal1))  # Find the index of the dominant frequency
phase_diff = phase_signal2[index] - phase_signal1[index]

# Normalize the phase difference to be within -pi to pi
phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

# Output the phase difference
print(f"Phase difference: {phase_diff} radians")
print(f"Phase difference: {np.degrees(phase_diff)} degrees")

print(f'''There are definitely inconsistent results!
Remark: We found out that the DSO was also struggling with the measuring og the phase''')
