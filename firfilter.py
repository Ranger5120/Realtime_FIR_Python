import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

class FIRfilter:
    def __init__(self, _coefficients):
        self.coefficients = _coefficients
        self.ntaps = len(_coefficients)
        self.buffer = np.zeros(self.ntaps)

    def dofilter(self,v):
        for j in range (self.ntaps-1):
            self.buffer[self.ntaps-j-1]  = self.buffer[self.ntaps-j-2]
        self.buffer[0] = v
        return np.inner(self.buffer,self.coefficients)
        
  
    def doFilterAdaptive(self,signal,fnoise,LEARNING_RATE):
        ecg = signal
        fs = 1000
        mu = LEARNING_RATE
        y = np.empty(len(ecg))
        for i in range(len(ecg)):
            ref_noise = np.sin(2.0 * np.pi * fnoise/fs * i )
            v = ref_noise
            for k in range (self.ntaps-1):
                self.buffer[self.ntaps-k-1]  = self.buffer[self.ntaps-k-2]
            self.buffer[0] = v
            canceller = np.inner(self.buffer,self.coefficients)
            output_signal = ecg[i] - canceller
            y[i] = output_signal  
            for j in range(self.ntaps):
                self.coefficients[j]=self.coefficients[j] + output_signal * mu * self.buffer[j] 
        return y
               
class firfilterdesign:
    def bandstopdesign(fs1,cutoff1):
        fs = fs1
        cutoff_fre = cutoff1
        f_pl = ((cutoff_fre-10)/fs)*2*np.pi
        f_sl = ((cutoff_fre-5)/fs)*2*np.pi
        f_sh = ((cutoff_fre+5)/fs)*2*np.pi

        w_d = abs(f_sl - f_pl)
        N = (6.6*np.pi)/w_d
        if N%2 == 0:
            N = N+1

        coefficient = N
        a = (N-1)/2
        n = np.arange(-a, a+1)

        M=int(N)

        k1=int(((cutoff_fre-5)/fs)*M)
        k2=int(((cutoff_fre+5)/fs)*M)
        X = np.ones(M)

        X[k1:k2+1]=0
        X[M-k2:M-k1+1]=0

        x=np.fft.ifft(X)
        x=np.real(x)

        h=np.zeros(M)

        h[0:int(M/2)]=x[int(M/2):M]
        h[int(M/2):M]=x[0:int(M/2)]
        '''
        plt.plot(h)
        plt.show()
        '''
        h_c = h*np.hamming(len(n)-1)
        '''
        fft_wave = abs(np.fft.fft(h_c))
        
        plt.plot(fft_wave)
        plt.show()
        '''
        return h_c,M

    def highpassdesign(fs1,cutoff):
       fs = fs1
       cutoff_fre = cutoff

       f_s = (cutoff_fre/fs)*2*np.pi
       f_p = ((cutoff_fre+5)/fs)*2*np.pi

       w_d = abs(f_s - f_p)
       N = (6.6*np.pi)/w_d
       if N % 2 == 0:
           N = N+1

       a = (N-1)/2
       n = np.arange(-a, a+1)

       M = int(N)

       k1 = int(((cutoff_fre + 5) / fs) * M)

       X = np.ones(M)
       X[0:k1+1] = 0
       
       x = np.fft.ifft(X)
       x = np.real(x)
       '''
       plt.plot(X)
       plt.show()
       '''
       h = np.zeros(M)

       h[0:int(M/2)] = x[int(M/2):M]
       h[int(M/2):M] = x[0:int(M/2)]
       '''
       plt.plot(h)
       plt.show()
       '''
       h_c = h * np.hamming(len(n) - 1)
       '''
       fft_wave = abs(np.fft.fft(h_c))

       plt.plot(fft_wave)
       plt.show()
       '''
       return h_c,M
