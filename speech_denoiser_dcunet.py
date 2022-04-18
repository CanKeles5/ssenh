# -*- coding: utf-8 -*-


noise_class = "white"
training_type =  "Noise2Noise"

import os
basepath = str(noise_class)+"_"+training_type
os.makedirs(basepath,exist_ok=True)
os.makedirs(basepath+"\Weights",exist_ok=True)
os.makedirs(basepath+"\Samples",exist_ok=True)

# Commented out IPython magic to ensure Python compatibility.
import time
import pickle
import warnings
import gc
import copy

import utils

from metrics import AudioMetrics
from metrics import AudioMetrics2

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from tqdm import tqdm, tqdm_notebook
from matplotlib import colors, pyplot as plt
#from pypesq import pesq

import models
import train
import dataset


np.random.seed(999)
torch.manual_seed(999)

# If running on Cuda set these 2 for determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""### Checking whether the GPU is available ###"""

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')
       
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

### Set Audio backend as Soundfile for windows and Sox for Linux ###

torchaudio.set_audio_backend("soundfile")
print("TorchAudio backend used:\t{}".format(torchaudio.get_audio_backend()))

"""### The sampling frequency and the selected values for the Short-time Fourier transform. ###"""

SAMPLE_RATE = 16000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

"""### The declaration of datasets and dataloaders ###"""

test_loader = dataset.create_dataloader(r"C:\Users\Can\Desktop\dummy_data\male", r"C:\Users\Can\Desktop\dummy_data\male")
train_loader = dataset.create_dataloader(r"C:\Users\Can\Desktop\dummy_data\male", r"C:\Users\Can\Desktop\dummy_data\male")

# For testing purpose
test_loader_single_unshuffled = test_loader

"""## Training New Model ##"""

# # clear cache
gc.collect()
torch.cuda.empty_cache()

dcunet20 = models.DCUnetCanA2A(N_FFT, HOP_LENGTH, DEVICE=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(dcunet20.parameters())

#model_checkpoint = torch.load("/content/dc20_model_2.pth")
#opt_checkpoint = torch.load("/content/dc20_opt_2.pth")
#dcunet20.load_state_dict(model_checkpoint)
#optimizer.load_state_dict(opt_checkpoint)

loss_fn = wsdr_fn
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

print(f"number of parameters: {utils.count_parameters(dcunet20)}")

# specify paths and uncomment to resume training from a given point
# model_checkpoint = torch.load(path_to_model)
# opt_checkpoint = torch.load(path_to_opt)
# dcunet20.load_state_dict(model_checkpoint)
# optimizer.load_state_dict(opt_checkpoint)

basepath = "/content/white_Noise2Noise"

train_losses, test_losses = train.train(dcunet20, train_loader, test_loader, loss_fn, optimizer, scheduler, 4)

"""## Using pretrained weights to run denoising inference ##

#### Select the model weight .pth file ####
"""

model_weights_path = "/content/white_Noise2Noise/Weights/dc20_model_2.pth"

dcunet20 = DCUnetCanA2A(N_FFT, HOP_LENGTH).to('cuda:0')
optimizer = torch.optim.Adam(dcunet20.parameters())

checkpoint = torch.load(model_weights_path,
                        map_location=torch.device('cuda:0')
                       )

"""#### Select the testing audio folders for inference ####"""

from pathlib import Path

test_noisy_files = sorted(list(Path("/content/wav16k/min/cv/mix_single_reverb").rglob('*.wav')))
test_clean_files = sorted(list(Path("/content/wav16k/min/cv/s1_anechoic").rglob('*.wav')))

test_dataset = dataset.SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH, task="upsample")

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)

dcunet20.load_state_dict(checkpoint)

"""#### Enter the index of the file in the Test Set folder to Denoise and evaluate metrics waveforms (Indexing starts from 0) ####"""

index = 800

dcunet20.eval()
test_loader_single_unshuffled_iter = iter(test_loader_single_unshuffled)

x_n, x_c = next(test_loader_single_unshuffled_iter)
for _ in range(index):
    x_n, x_c = next(test_loader_single_unshuffled_iter)

x_est = dcunet20(x_n.to('cuda:0'), is_istft=True)

x_est_np = x_est[0].view(-1).detach().cpu().numpy()
x_c_np = torch.istft(torch.squeeze(x_c[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
x_n_np = torch.istft(torch.squeeze(x_n[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()

"""#### Metrics ####"""

metrics = AudioMetrics(x_c_np, x_est_np, SAMPLE_RATE)
print(metrics.display())

"""#### Visualization of denoising the audio in /Samples folder ####

#### Noisy audio waveform ####
"""

plt.plot(x_n_np)

"""#### Model denoised audio waveform ####"""

plt.plot(x_est_np)

"""#### True clean audio waveform ####"""

plt.plot(x_c_np)

"""#### Save Recently Denoised Speech Files ####"""

utils.save_audio_file(np_array=x_est_np,file_path=Path("/content/denoised16khz.wav"), sample_rate=16000, bit_precision=16)
utils.save_audio_file(np_array=x_c_np,file_path=Path("/content/clean.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
utils.save_audio_file(np_array=x_n_np,file_path=Path("/content/noisy.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)

