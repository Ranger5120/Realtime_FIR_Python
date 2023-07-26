# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:33:48 2022

@author: xiaha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import firfilter


ECG_wave = np.loadtxt('ECG.dat')
plt.plot(ECG_wave)
plt.show()
iSampleRate = 1000

(bandstop,N) = firfilter.firfilterdesign.bandstopdesign(1000, 50)
(highpass,N2) = firfilter.firfilterdesign.highpassdesign(1000, 0.5)

filtered_wave1 = np.zeros(len(ECG_wave))
filtered_wave2 = np.zeros(len(ECG_wave))


coe_band = firfilter.FIRfilter(bandstop)
coe_high = firfilter.FIRfilter(highpass)

for i in range(0, len(ECG_wave)-1):
    filtered_wave1[i] = coe_high.dofilter(ECG_wave[i])
    filtered_wave2[i] = coe_band.dofilter(filtered_wave1[i])

plt.figure(1)
plt.plot(filtered_wave2)


template = filtered_wave2[500:1000]
plt.subplot(211)
plt.plot(template)
fir_coeff = template[::-1]
plt.subplot(212)
plt.plot(fir_coeff)

matched_wave = np.zeros(len(filtered_wave2))
matched = firfilter.FIRfilter(fir_coeff)

for k in range(0, len(filtered_wave2)-1):
    matched_wave[k] = matched.dofilter(filtered_wave2[k])

iSampleCount = matched_wave.shape[0]          
t = np.linspace(0,iSampleCount/iSampleRate,iSampleCount)    
plt.figure(2)

plt.plot(t,matched_wave*matched_wave)

plt.show()