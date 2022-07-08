from tqdm import tqdm
import gc
import torch

import metrics
import utils


DEVICE = "cpu"
basepath = r"C:\Users\Can\Desktop\ssenh\white_Noise2Noise"

SAMPLE_RATE = 16000
N_FFT = (SAMPLE_RATE * 64) // 1000
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000

#Train


def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.0
    counter = 0
    
    for noisy_x, clean_x in train_loader:
        
        if counter % 100 == 0:
          print(f"Processing batch {counter}.")

        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        
        #noisy_x = torch.istft(torch.squeeze(noisy_x[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().numpy()
        #clean_x = torch.istft(torch.squeeze(clean_x[0], 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True).view(-1).detach().numpy()
        
        #_array=x_c_np,file_path=Path("/content/res/clean.wav"), sample_rate=16000, bit_precision=16
        #utils.save_audio_file(np_array=noisy_x, file_path=r"C:\Users\Can\Desktop\16khz.wav", sample_rate=8000, bit_precision=16)
        #utils.save_audio_file(np_array=clean_x, file_path=r"C:\Users\Can\Desktop\8khz.wav", sample_rate=16000, bit_precision=16)
        
        #return None
        
        # zero  gradients
        net.zero_grad()

        # get the output from the model
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        
        loss.backward()
        optimizer.step()
        
        train_ep_loss += loss.item() 
        
        counter += 1

    train_ep_loss /= counter
    
    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss


"""### Description of the validation of epochs ###"""

def val_epoch(net, val_loader, loss_fn):
    net.eval()
    val_ep_loss = 0.0
    counter = 0
    
    for noisy_x, clean_x in val_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)
        
        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        val_ep_loss += loss.item()
        
        counter += 1

    val_ep_loss /= counter
    
    #print("Actual compute done...testing now")
    
    #testmet = metrics.getMetricsonLoader(test_loader,net,use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return val_ep_loss



"""### To understand whether the network is being trained or not, we will output a train and test loss. ###"""

def train(net, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs):
    
    train_losses = []
    val_losses = []
    
    for e in tqdm(range(epochs)):

        # first evaluating for comparison        
        
        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        val_loss = 0
        scheduler.step()
        
        with torch.no_grad():
            val_loss = val_epoch(net, val_loader, loss_fn)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        with open(basepath + "/results.txt","a") as f:
            f.write("Epoch :"+str(e+1) + "\n" + str(val_losses))
            f.write("\n")
        
        print("OPed to txt")
        
        print("Saving model....")
        
        torch.save(net.state_dict(), basepath +'\Weights\dc20_model_'+str(e+1)+'.pth')
        torch.save(optimizer.state_dict(), basepath+'\Weights\dc20_opt_'+str(e+1)+'.pth')
        
        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

        #print("Epoch: {}/{}...".format(e+1, epochs),
        #              "Loss: {:.6f}...".format(train_loss),
        #              "Test Loss: {:.6f}".format(test_loss))
        
        print(f"Epoch {e} : Train loss = {train_loss}, Validation loss = {val_loss}")
    
    return train_loss, val_loss


