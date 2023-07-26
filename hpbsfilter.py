import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import firfilter

ECG_wave = np.loadtxt('ECG.dat')

plt.title('original signal')
plt.plot(ECG_wave)
plt.savefig("original_signal.svg", dpi=600, format="svg")
plt.show()

template2 = ECG_wave[200:2000]
plt.title('single heart beats')
plt.plot(template2)
plt.savefig("single_heartbeats.svg", dpi=600, format="svg")
plt.show()


(bandstop,N) = firfilter.firfilterdesign.bandstopdesign(1000, 50)
(highpass,N2) = firfilter.firfilterdesign.highpassdesign(1000, 0.5)

filtered_wave1 = np.zeros(len(ECG_wave))
filtered_wave2 = np.zeros(len(ECG_wave))


coe_band = firfilter.FIRfilter(bandstop)
coe_high = firfilter.FIRfilter(highpass)

for i in range(0, len(ECG_wave)-1):
    filtered_wave1[i] = coe_high.dofilter(ECG_wave[i])
    filtered_wave2[i] = coe_band.dofilter(filtered_wave1[i])
    
'''    
ax1 = plt.figure(1)   
plt.title('bandstop 50Hz and highpass 0.5Hz')    

ax1.set_xlabel("Time(s)")
ax1.plot(filtered_wave2)
'''

iSampleCount = filtered_wave2.shape[0]           
t = np.linspace(0,iSampleCount/1000,iSampleCount)   

# ax1 = plt.subplot(111)          
# ax1.set_title("bandstop 50Hz and highpass 0.5Hz")
# ax1.set_xlabel("Time(s)")
# ax1.set_ylabel("")
# ax1.plot(t,filtered_wave2)



plt.title('Result Signal')
plt.plot(filtered_wave2)
plt.savefig("filtered_signal.svg", dpi=600, format="svg")
plt.show()

template = filtered_wave2[200:2000]
plt.title('single heart beats')
plt.plot(template)
plt.savefig("single_heartbeats_filtered.svg", dpi=600, format="svg")
plt.show()
