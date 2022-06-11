from metrics_utils import *
import numpy as np
from scipy.io import wavfile
from scipy import interpolate
from scipy.linalg import solve_toeplitz,toeplitz
#import pesq as pypesq
from pystoi import stoi
import random
import torch

SAMPLE_RATE = 16000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

DEVICE = "cpu"

# Expected input, 2 numpy arrays, one for the reference clean audio, the other for the degraded audio, and sampling rate (should be same)
# The way we'd use these metrics would be to compute the values on clean compared to noisy and then clean compared to our denoising results

class AudioMetrics():
    def __init__(self, target_speech, input_speech, fs): 
        if len(target_speech) != len(input_speech):
            raise AudioMetricsException("Signal lengths don't match!")
    
        self.min_cutoff = 0.01
        self.clip_values = (-self.min_cutoff, self.min_cutoff)

        # The SSNR and composite metrics fail when comparing silence
        # The minimum value of the signal is clipped to 0.001 or -0.001 to overcome that. For reference, in a non-silence case, the minimum value was around 40 (???? Find correct value)
        # For PESQ and STOI, results are identical regardless of wether or not 0 is present

        # The Metrics are as follows:
        # SSNR : Segmented Signal to noise ratio - Capped from [-10,35] (higher is better)
        # PESQ : Perceptable Estimation of Speech Quality - Capped from [-0.5, 4.5]
        # STOI : Short Term Objective Intelligibilty of Speech - From 0 to 1
        # CSIG : Quality of Speech Signal. Ranges from 1 to 5 (Higher is better)
        # CBAK : Quality of Background intrusiveness. Ranges from 1 to 5 (Higher is better - less intrusive)
        # COVL : Overall Quality measure. Ranges from 1 to 5 (Higher is better)
        # CSIG,CBAK and COVL are computed using PESQ and some other metrics like LLR and WSS
        
        clean_speech = np.zeros(shape=target_speech.shape)
        processed_speech = np.zeros(shape=input_speech.shape)

        for index, data in np.ndenumerate(target_speech):
            # If value less than min_cutoff difference from 0, then clip
            '''
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    clean_speech[index] = self.clip_values[0]
                else:
                    clean_speech[index] = self.clip_values[1]
            '''
            if data==0:
                clean_speech[index] = 0.01
            else:
                clean_speech[index] = data
                
        for index, data in np.ndenumerate(input_speech):
            '''
            # If value less than min_cutoff difference from 0, then clip
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    processed_speech[index] = self.clip_values[0]
                else:
                    processed_speech[index] = self.clip_values[1]
            '''
            if data==0:
                processed_speech[index] = 0.01
            else:
                processed_speech[index] = data
             
                
        #print('clean speech: ', clean_speech)
        #print('processed speech : ', processed_speech)
        self.SNR = snr(target_speech, input_speech)
        self.SSNR = SNRseg(target_speech, input_speech,fs)
        self.PESQ = 0 #pesq_score(clean_speech, processed_speech, fs, force_resample=True)
        self.STOI = stoi_score(clean_speech, processed_speech, fs)
        self.CSIG, self.CBAK, self.COVL = composite(clean_speech, processed_speech, fs)

    def display(self):
        fstring = "{} : {:.3f}"
        metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR"]
        for name in metric_names:
            metric_value = eval("self."+name)
            print(fstring.format(name,metric_value))
        
        
class AudioMetrics2():
    def __init__(self, target_speech, input_speech, fs): 
        if len(target_speech) != len(input_speech):
            raise AudioMetricsException("Signal lengths don't match!")
    
        self.min_cutoff = 0.01
        self.clip_values = (-self.min_cutoff, self.min_cutoff)

        # The SSNR and composite metrics fail when comparing silence
        # The minimum value of the signal is clipped to 0.001 or -0.001 to overcome that. For reference, in a non-silence case, the minimum value was around 40 (???? Find correct value)
        # For PESQ and STOI, results are identical regardless of wether or not 0 is present

        # The Metrics are as follows:
        # SSNR : Segmented Signal to noise ratio - Capped from [-10,35] (higher is better)
        # PESQ : Perceptable Estimation of Speech Quality - Capped from [-0.5, 4.5]
        # STOI : Short Term Objective Intelligibilty of Speech - From 0 to 1
        # CSIG : Quality of Speech Signal. Ranges from 1 to 5 (Higher is better)
        # CBAK : Quality of Background intrusiveness. Ranges from 1 to 5 (Higher is better - less intrusive)
        # COVL : Overall Quality measure. Ranges from 1 to 5 (Higher is better)
        # CSIG,CBAK and COVL are computed using PESQ and some other metrics like LLR and WSS
        
        clean_speech = np.zeros(shape=target_speech.shape)
        processed_speech = np.zeros(shape=input_speech.shape)

        for index, data in np.ndenumerate(target_speech):
            # If value less than min_cutoff difference from 0, then clip
            '''
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    clean_speech[index] = self.clip_values[0]
                else:
                    clean_speech[index] = self.clip_values[1]
            '''
            if data==0:
                clean_speech[index] = 0.01
            else:
                clean_speech[index] = data
                
        for index, data in np.ndenumerate(input_speech):
            '''
            # If value less than min_cutoff difference from 0, then clip
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    processed_speech[index] = self.clip_values[0]
                else:
                    processed_speech[index] = self.clip_values[1]
            '''
            if data==0:
                processed_speech[index] = 0.01
            else:
                processed_speech[index] = data
             
                
        #print('clean speech: ', clean_speech)
        #print('processed speech : ', processed_speech)
        self.SNR = snr(target_speech, input_speech)
        self.SSNR = SNRseg(target_speech, input_speech,fs)
        self.STOI = stoi_score(clean_speech, processed_speech, fs)

# Formula Reference: http://www.irisa.fr/armor/lesmembres/Mohamed/Thesis/node94.html

def snr(reference, test):
    numerator = 0.0
    denominator = 0.0
    for i in range(len(reference)):
        numerator += reference[i]**2
        denominator += (reference[i] - test[i])**2
    return 10*np.log10(numerator/denominator)


# Reference : https://github.com/schmiph2/pysepm

def SNRseg(clean_speech, processed_speech,fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    MIN_SNR     = -10 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1] # remove last frame -> not valid
    return np.mean(segmental_snr)



def composite(clean_speech, processed_speech, fs):
    wss_dist=wss(clean_speech, processed_speech, fs)
    llr_mean=llr(clean_speech, processed_speech, fs,used_for_composite=True)
    segSNR=SNRseg(clean_speech, processed_speech, fs)
    pesq_mos,mos_lqo = 0,0 # pesq(clean_speech, processed_speech,fs)    
    if fs >= 16e3:
        used_pesq_val = mos_lqo
    else:
        used_pesq_val = pesq_mos    

    Csig = 3.093 - 1.029*llr_mean + 0.603*used_pesq_val-0.009*wss_dist
    Csig = np.max((1,Csig))  
    Csig = np.min((5, Csig)) # limit values to [1, 5]
    Cbak = 1.634 + 0.478 *used_pesq_val - 0.007*wss_dist + 0.063*segSNR
    Cbak = np.max((1, Cbak))
    Cbak = np.min((5,Cbak)) # limit values to [1, 5]
    Covl = 1.594 + 0.805*used_pesq_val - 0.512*llr_mean - 0.007*wss_dist
    Covl = np.max((1, Covl))
    Covl = np.min((5, Covl)) # limit values to [1, 5]
    return Csig,Cbak,Covl

def pesq_score(clean_speech, processed_speech, fs, force_resample=False):
    if fs!=8000 or fs!=16000:
        if force_resample:
            clean_speech = resample(clean_speech, fs, 16000)
            processed_speech = resample(processed_speech, fs, 16000)
            fs = 16000
        else:
            raise(AudioMetricsException("Invalid sampling rate for PESQ! Need 8000 or 16000Hz but got "+str(fs)+"Hz"))
    if fs==16000:
        score = 0 #pypesq.pesq(16000, clean_speech, processed_speech, 'wb')
        score = min(score,4.5)
        score = max(-0.5,score)
        return(score)
    else:
        score = 0 #pypesq.pesq(16000, clean_speech, processed_speech, 'nb')
        score = min(score,4.5)
        score = max(-0.5,score)
        return(score)

# Original paper http://cas.et.tudelft.nl/pubs/Taal2010.pdf
# Says to resample to 10kHz if not already at that frequency. I've kept options to adjust
def stoi_score(clean_speech, processed_speech, fs, force_resample=True, force_10k=True):
    if fs!=10000 and force_10k==True:
        if force_resample:
            clean_speech = resample(clean_speech, fs, 10000)
            processed_speech = resample(processed_speech, fs, 10000)
            fs = 10000
        else:
            raise(AudioMetricsException("Forced 10kHz sample rate for STOI. Got "+str(fs)+"Hz"))
    return stoi(clean_speech, processed_speech, 10000, extended=False)

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


