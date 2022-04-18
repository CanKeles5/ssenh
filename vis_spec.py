#visualize a spectrogram

# Importing libraries using import keyword.
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import torch


file = r"C:\Users\Can\Desktop\dummy_data\male\eng5-male0.wav"

# Generating an array of values
sr1, audio_8khz = wavfile.read(file)
audio_8khz = signal.resample(audio_8khz, int(audio_8khz.shape[0]/sr1*8000))
spec_8khz = torch.stft(input=torch.from_numpy(audio_8khz).unsqueeze(0), n_fft=64, hop_length=16, normalized=True)

sr2, audio_16khz = wavfile.read(file)
spec_16khz = torch.stft(input=torch.from_numpy(audio_16khz).unsqueeze(0).type(torch.FloatTensor), n_fft=64, hop_length=16, normalized=True)


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
