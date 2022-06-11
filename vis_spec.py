#visualize a spectrogram

# Importing libraries using import keyword.
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import torch


file = r"C:\Users\Can\Desktop\dummy_data\male\eng5-male0.wav"




SAMPLE_RATE = 8000
N_FFT_8khz = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH_8khz = (SAMPLE_RATE * 16) // 1000
sr1, audio_8khz = wavfile.read(file)
audio_8khz = signal.resample(audio_8khz, int(audio_8khz.shape[0]/sr1*8000))
spec_8khz = torch.stft(input=torch.from_numpy(audio_8khz).unsqueeze(0), n_fft=N_FFT_8khz, hop_length=HOP_LENGTH_8khz, normalized=True)
print(f"spec_8khz.shape: {spec_8khz.shape}")

SAMPLE_RATE = 16000
N_FFT_16khz = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH_16khz = (SAMPLE_RATE * 16) // 1000

sr2, audio_16khz = wavfile.read(file)
spec_16khz = torch.stft(input=torch.from_numpy(audio_16khz).unsqueeze(0).type(torch.FloatTensor), n_fft=N_FFT_16khz, hop_length=HOP_LENGTH_16khz, normalized=True)
print(f"spec_16khz.shape: {spec_16khz.shape}")


#import the pyplot and wavfile modules 

import matplotlib.pyplot as plot

from scipy.io import wavfile

 

# Read the wav file (mono)

#samplingFrequency, signalData = wavfile.read('y.wav')

# Plot the signal read from wav file
plot.subplot(211)
plot.plot(audio_8khz[0:200])
#plot.specgram(audio_8khz[0:8000], Fs=8000) #,Fs=16000)
plot.xlabel('Time')
plot.ylabel('Frequency')

plot.subplot(212)
plot.plot(audio_16khz[0:400])
#plot.specgram(audio_16khz[0:16000], Fs=16000) #,Fs=16000)
plot.xlabel('Time')
plot.ylabel('Frequency')
plot.show()



"""
# Matplotlib.pyplot.specgram() function to
# generate spectrogram
plt.subplot(221)
plt.specgram(audio_8khz, Fs=6, cmap="rainbow")
plt.title('Spectrogram of 8khz sample')
plt.xlabel("DATA")
plt.ylabel("TIME")

plt.subplot(222)
plt.specgram(audio_16khz, Fs=6, cmap="rainbow")
plt.title('Spectrogram of 16khz sample')
plt.xlabel("DATA")
plt.ylabel("TIME")

plt.show()
"""
