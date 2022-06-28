"""
Main file
"""

import torch
import yaml

import dataset
import models
import train
import metrics
import utils


SAMPLE_RATE = 16000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000


if __name__ == "__main__":
    
    with open(r"C:\Users\Can\Desktop\ssenh\config.yml") as f:
        conf = yaml.safe_load(f)
    
    #get the paths for data from the config file
    val_loader = dataset.create_dataloader(r"C:\Users\Can\Desktop\dummy_data\male", r"C:\Users\Can\Desktop\dummy_data\male", task=conf['task'])
    train_loader = dataset.create_dataloader(r"C:\Users\Can\Desktop\dummy_data\male", r"C:\Users\Can\Desktop\dummy_data\male", task=conf['task'])
    
    dcunet20 = models.DCUnetA2A(N_FFT, HOP_LENGTH, DEVICE=conf['device'])
    optimizer = torch.optim.Adam(dcunet20.parameters())
    
    loss_fn = metrics.wsdr_fn
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    print(f"Number of parameters: {utils.count_parameters(dcunet20)}")
    
    #basepath = r"C:\Users\Can\Desktop\ssenh\white_Noise2Noise"
    
    train_losses, test_losses = train.train(dcunet20, train_loader, val_loader, loss_fn, optimizer, scheduler, 1)
    


