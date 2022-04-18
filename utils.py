import numpy as np
import torch
import torchaudio


def save_audio_file(np_array=np.array([0.5]*1000),file_path='./sample_audio.wav', sample_rate=48000, bit_precision=16):
    np_array = np.reshape(np_array, (1,-1))
    torch_tensor = torch.from_numpy(np_array)
    torchaudio.save(file_path, torch_tensor, sample_rate, bits_per_sample=bit_precision)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
