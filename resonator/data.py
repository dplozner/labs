import numpy as n
import os 
import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_csv("data.csv", delimiter=";")
f = df["Frequency [Hz]"]
f = [i for i in f]
X = df["Amplitude [mV]"]
X = [i for i in X]
phi = df["Phase [mV/cyc.]"]
phi = [i for i in phi]

### task 1: fit amplitude/frequency to find max --> resonance freq
###         fit phase/frequency


