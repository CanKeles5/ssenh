#Dataset

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import numpy as np
import glob

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
        # load to tensors and normalization
        x_clean = self.load_sample(self.clean_files[index])
        x_noisy = self.load_sample(self.noisy_files[index])
        
        # padding/cutting
        x_clean = self._prepare_sample(x_clean)
        x_noisy = self._prepare_sample(x_noisy)
        
        """
        if self.task == "upsample":
            pass
        elif self.task == "denoise":
            pass
        elif self.task == "fif":
            pass
        """
        
        # Short-time Fourier transform
        x_noisy_stft = torch.stft(input=x_noisy, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, normalized=True)
        x_clean_stft = torch.stft(input=x_clean, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, normalized=True)
        
        return x_noisy_stft, x_clean_stft
        
    def _prepare_sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]
        
        output = np.zeros((1, self.max_len), dtype='float32')
        output[0, -current_len:] = waveform[0, :self.max_len]
        output = torch.from_numpy(output)
        
        return output


def create_dataloader(input_folder, target_folder):
    INPUT_PATHS = glob.glob(input_folder + "\*.wav")
    TARGET_PATHS = glob.glob(target_folder + "\*.wav")
    
    input_files = sorted(list(INPUT_PATHS))
    target_files = sorted(list(TARGET_PATHS))
    
    print("No. of Training files: ",len(input_files))
    
    dataset = SpeechDataset(input_files, target_files, n_fft=64, hop_length=16)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    return dataloader

