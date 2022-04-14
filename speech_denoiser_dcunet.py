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

import noise_addition_utils #only using the save audio function, possibly can be removed

from metrics import AudioMetrics
from metrics import AudioMetrics2

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from tqdm import tqdm, tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
#from pypesq import pesq

import models
import train
import dataset

# not everything is smooth in sklearn, to conveniently output images in colab
# we will ignore warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

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



TRAIN_INPUT_DIR = r"C:\Users\Can\Desktop\dummy_data\male"
#TRAIN_TARGET_DIR = "/content/wav16k/min/tr/s1_anechoic"
TRAIN_TARGET_DIR = TRAIN_INPUT_DIR

VAL_INPUT_DIR = r"C:\Users\Can\Desktop\dummy_data\male"
#VAL_TARGET_DIR = "/content/wav16k/min/cv/s1_anechoic"
VAL_TARGET_DIR = VAL_INPUT_DIR

import glob
TRAIN_INPUT_PATHS = glob.glob(TRAIN_INPUT_DIR + "\*.wav")
TRAIN_TARGET_PATHS = glob.glob(TRAIN_TARGET_DIR + "\*.wav")

VAL_INPUT_PATHS = glob.glob(VAL_INPUT_DIR + "\*.wav")
VAL_TARGET_PATHS = glob.glob(VAL_TARGET_DIR + "\*.wav")

train_input_files = sorted(list(TRAIN_INPUT_PATHS))
train_target_files = sorted(list(TRAIN_TARGET_PATHS))

test_noisy_files = sorted(list(VAL_INPUT_PATHS))
test_clean_files = sorted(list(VAL_TARGET_PATHS))

print("No. of Training files:",len(train_input_files))
print("No. of Testing files:",len(test_noisy_files))

test_dataset = dataset.SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)
train_dataset = dataset.SpeechDataset(train_input_files, train_target_files, N_FFT, HOP_LENGTH)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# For testing purpose
test_loader_single_unshuffled = DataLoader(test_dataset, batch_size=1, shuffle=False)

"""### Average Test Set Metrics ###"""

def test_set_metrics(test_loader, model):
    metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI"]
    overall_metrics = [[] for i in range(len(metric_names))]
    
    for i,(noisy,clean) in enumerate(test_loader):
        x_est = model(noisy.to(DEVICE), is_istft=True)
        x_est_np = x_est[0].view(-1).detach().cpu().numpy()
        x_c_np = torch.istft(torch.squeeze(clean[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
        metrics = AudioMetrics(x_c_np, x_est_np, SAMPLE_RATE)
        
        overall_metrics[0].append(metrics.CSIG)
        overall_metrics[1].append(metrics.CBAK)
        overall_metrics[2].append(metrics.COVL)
        overall_metrics[3].append(metrics.PESQ)
        overall_metrics[4].append(metrics.SSNR)
        overall_metrics[5].append(metrics.STOI)
    
    metrics_dict = dict()
    for i in range(len(metric_names)):
        metrics_dict[metric_names[i]] ={'mean': np.mean(overall_metrics[i]), 'std_dev': np.std(overall_metrics[i])} 
    
    return metrics_dict


"""### Loss function ###"""

#from pesq import pesq
from scipy import interpolate

def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old  = np.linspace(0, duration, original.shape[0])
        time_new  = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original


def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    y_true = torch.istft(y_true_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)
    x_ = torch.squeeze(x_, 1)
    x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)


    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)

wonky_samples = []

def getMetricsonLoader(loader, net, use_net=True):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    metric_names = ["PESQ-WB","PESQ-NB","SNR","SSNR","STOI"]
    overall_metrics = [[] for i in range(5)]
    for i, data in enumerate(loader):
        if (i+1)%10==0:
            end_str = "\n"
        else:
            end_str = ","
        #print(i,end=end_str)
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            noisy = data[0]
            clean = data[1]
            if use_net: # Forward of net returns the istft version
                x_est = net(noisy.to(DEVICE), is_istft=True)
                x_est_np = x_est.view(-1).detach().cpu().numpy()
            else:
                x_est_np = torch.istft(torch.squeeze(noisy, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
            x_clean_np = torch.istft(torch.squeeze(clean, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().cpu().numpy()
            
        
            metrics = AudioMetrics2(x_clean_np, x_est_np, 16000)
            
            ref_wb = resample(x_clean_np, 16000, 16000)
            deg_wb = resample(x_est_np, 16000, 16000)
            pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')
            
            ref_nb = resample(x_clean_np, 16000, 8000)
            deg_nb = resample(x_est_np, 16000, 8000)
            pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')

            #print(new_scores)
            #print(metrics.PESQ, metrics.STOI)

            overall_metrics[0].append(pesq_wb)
            overall_metrics[1].append(pesq_nb)
            overall_metrics[2].append(metrics.SNR)
            overall_metrics[3].append(metrics.SSNR)
            overall_metrics[4].append(metrics.STOI)
    print()
    print("Sample metrics computed")
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] =  np.mean(overall_metrics[i])
        temp["STD"]  =  np.std(overall_metrics[i])
        temp["Min"]  =  min(overall_metrics[i])
        temp["Max"]  =  max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data",addon)
    for i in range(5):
        print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])))
    return results

"""## Training New Model ##"""

# # clear cache
gc.collect()
torch.cuda.empty_cache()


dcunet20 = models.DCUnetA2A(N_FFT, HOP_LENGTH, DEVICE=DEVICE).to(DEVICE)
optimizer = torch.optim.Adam(dcunet20.parameters())

#model_checkpoint = torch.load("/content/dc20_model_2.pth")
#opt_checkpoint = torch.load("/content/dc20_opt_2.pth")
#dcunet20.load_state_dict(model_checkpoint)
#optimizer.load_state_dict(opt_checkpoint)

loss_fn = wsdr_fn
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"number of parameters: {count_parameters(dcunet20)}")

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

test_dataset = dataset.SpeechDataset(test_noisy_files, test_clean_files, N_FFT, HOP_LENGTH)

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

noise_addition_utils.save_audio_file(np_array=x_est_np,file_path=Path("/content/denoised16khz.wav"), sample_rate=16000, bit_precision=16)
noise_addition_utils.save_audio_file(np_array=x_c_np,file_path=Path("/content/clean.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)
noise_addition_utils.save_audio_file(np_array=x_n_np,file_path=Path("/content/noisy.wav"), sample_rate=SAMPLE_RATE, bit_precision=16)

