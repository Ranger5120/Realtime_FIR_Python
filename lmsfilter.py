# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 20:43:31 2022

@author: xiaha
"""
import numpy as np
import pylab as pl
import firfilter as fir

fs = 1000
LEARNING_RATE = 0.001
fnoise = 50
(cofficients,NTAPS) =fir.firfilterdesign.bandstopdesign(fs, fnoise)


ecg = np.loadtxt("ECG.dat")
pl.figure(1)
pl.plot(ecg)

f = fir.FIRfilter(np.zeros(NTAPS))
y = f.doFilterAdaptive(ecg,fnoise,LEARNING_RATE)

'''
for i in range(len(ecg)):
    ref_noise = np.sin(2.0 * np.pi * fnoise/fs * i );
    canceller = f.dofilter(ref_noise)
    output_signal = ecg[i] - canceller
    f.lms(output_signal,LEARNING_RATE)
    y[i] = output_signal
'''
pl.figure(2)
pl.plot(y)
pl.show()

template = y[200:2000]
pl.title('single heart beats')
pl.plot(template)
pl.savefig("single_hear_tbeat.svg", dpi=600, format="svg")
pl.show()
