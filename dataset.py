#Dataset

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import numpy as np
import glob
from scipy import signal
import random
import matplotlib.pyplot as plt

random.seed(10)


SAMPLE_RATE = 16000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 


class SpeechDataset(Dataset):
    """
    A dataset class with audio that cuts them/paddes them to a specified length, applies a Short-tome Fourier transform,
    normalizes and leads to a tensor.
    """
    def __init__(self, noisy_files, clean_files, n_fft=64, hop_length=16, task="enhancement"):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        
        # stft parameters
        self.n_fft = N_FFT
        self.hop_length = HOP_LENGTH
        
        self.len_ = len(self.noisy_files)
        
        # fixed len
        self.max_len = 165000
        
        self.task = task
    
    def __len__(self):
        return self.len_
    
    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform
    
    def __getitem__(self, index):
        
        print(f"self.task: {self.task}")
        
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])
        
        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        
        if self.task == "enhancement":
            x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
            
        elif self.task == "denoise":
            rand_vec = torch.add(torch.rand(x_clean.shape)*2, -1)/10
            x_noisy = torch.add(x_clean.clone(), rand_vec)
            x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
            
        elif self.task == "upsample":
            audio_8khz = signal.decimate(x_clean[0], 2)
            
            x_noisy_stft = torch.stft(input=torch.from_numpy(np.expand_dims(audio_8khz.copy(), axis=0)), n_fft=self.n_fft // 2, hop_length=self.hop_length // 2, normalized=True)
            zeros = torch.zeros(x_noisy_stft.shape[0], x_noisy_stft.shape[1]-1, x_noisy_stft.shape[2], x_noisy_stft.shape[3])
            x_noisy_stft = torch.cat((zeros, x_noisy_stft), dim=1)
        
        
        # Short-time Fourier transform
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)
        
        #print(f"Task {self.task}.")
        #print(f"x_noisy_stft.shape: {x_noisy_stft.shape}")
        #print(f"x_clean_stft.shape: {x_clean_stft.shape}")
        
        return x_noisy_stft, x_clean_stft
        
    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]
        
        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)
        
        return output


def create_dataloader(input_folder, target_folder, task="enhancement", percentage=1.0):
    INPUT_PATHS = glob.glob(input_folder + "\*.wav")
    TARGET_PATHS = glob.glob(target_folder + "\*.wav")
    
    if percentage != 1.0:
        INPUT_PATHS = random.sample(INPUT_PATHS, int(percentage*len(INPUT_PATHS)))
        TARGET_PATHS = random.sample(TARGET_PATHS, int(percentage*len(TARGET_PATHS)))
    
    input_files = sorted(list(INPUT_PATHS))
    target_files = sorted(list(TARGET_PATHS))
    
    print("Number of Training files: ",len(input_files))
    
    dataset = SpeechDataset(input_files, target_files, n_fft=N_FFT, hop_length=HOP_LENGTH, task=task)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    return dataloader

