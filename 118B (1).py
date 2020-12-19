#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install neurodsp')


# In[128]:


import numpy as np
import sklearn
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import io
import pickle
import scipy.signal as signal 
from neurodsp.spectral import compute_spectrum
import pywt
import numpy as np


# In[3]:


eeg_fs = 250 ##250 Hz


# ## functions provided to us by starternotebook

# In[4]:


## Create DF for each of these, columns are channels, each row is a trial run
def getDF(epochs, labels, times, chans):
    data_dict = {}
    for i, label in enumerate(labels): 
        start_time = times[i][0]
        if 'start_time' not in data_dict: 
            data_dict['start_time'] = list()
        data_dict['start_time'].append(start_time)
        
        if 'event_type' not in data_dict:
            data_dict['event_type'] = list()
        data_dict['event_type'].append(label)
        
        for ch in range(len(chans)): 
            if chans[ch] not in data_dict:
                data_dict[chans[ch]] = list() 
            data_dict[chans[ch]].append(epochs[i][ch])
        
    return pd.DataFrame(data_dict)


# In[5]:


# Extract data from raw dataframes for constructing trial-by-trial dataframe
def getEpochedDF(eeg_df, event_df, trial_duration_ms=4000):
    epochs = []
    epoch_times = []
    labels = []
    start_df = eeg_df[eeg_df['EventStart'] == 1]
    for i, event_type in enumerate(event_df["EventType"].values): 
        labels.append(event_type)
        start_time = start_df.iloc[i]["time"]
        end_time = int(start_time + trial_duration_ms)
        epoch_times.append((start_time, end_time))
        sub_df = eeg_df[(eeg_df['time'] > start_time) & (eeg_df['time'] <= end_time)]
        eeg_dat = []
        for ch in all_chans: 
            eeg_dat.append(sub_df[ch].values) ##Change to np.mean(sub_df[ch].values) to get meaned channel data
        epochs.append(np.array(eeg_dat))

    # Create dataframe from the data extracted previously
    eeg_epoch_df = getDF(epochs, labels, epoch_times, all_chans)
    return eeg_epoch_df


# In[6]:


# PSD plotting
def plotPSD(freq, psd, fs=eeg_fs, pre_cut_off_freq=0, post_cut_off_freq=120, label=None):
    '''
    Inputs 
    - freq: the list of frequencies corresponding to the PSDs
    - psd: the list of psds that represent the power of each frequency
    - pre_cut_off_freq: the lowerbound of the frequencies to show
    - post_cut_off_freq: the upperbound of the frequencies to show
    - label: a text label to assign this plot (in case multiple plots want to be drawn)
    
    Outputs: 
    - None, except a plot will appear. plot.show() is not called at the end, so you can call this again to plot on the same axes. 
    '''
    # Label the axes
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('log(PSD)')
    
    # Calculate the frequency point that corresponds with the desired cut off frequencies
    pre_cut = int(len(freq)*(pre_cut_off_freq / freq[-1]))
    post_cut = int(len(freq)*(post_cut_off_freq / freq[-1]))
    
    # Plot
    plt.plot(freq[pre_cut:post_cut], np.log(psd[pre_cut:post_cut]), label=label)

# Get Frequencies and PSDs from EEG data - this is the raw PSD method. 
def getFreqPSDFromEEG(eeg_data, fs=eeg_fs):
    # Use scipy's signal.periodogram to do the conversion to PSDs
    freq, psd = signal.periodogram(eeg_data, fs=int(fs), scaling='spectrum')
    return freq, psd

# Get Frequencies and mean PSDs from EEG data - this yeilds smoother PSDs because it averages the PSDs made from sliding windows. 
def getMeanFreqPSD(eeg_data, fs=eeg_fs):
    freq_mean, psd_mean = compute_spectrum(eeg_data, fs, method='welch', avg_type='mean', nperseg=fs*2)
    return freq_mean, psd_mean

# Plot PSD from EEG data (combines the a PSD calculator function and the plotting function)
def plotPSD_fromEEG(eeg_data, fs=eeg_fs, pre_cut_off_freq=0, post_cut_off_freq=120, label=None):
    freq, psd = getMeanFreqPSD(eeg_data, fs=fs)
    plotPSD(freq, psd, fs, pre_cut_off_freq, post_cut_off_freq, label)


# In[7]:


# Spectrogram plotting
def plotSpectrogram_fromEEG(eeg_data, fs=eeg_fs, pre_cut_off_freq=0, post_cut_off_freq=120):
    f, t, Sxx = signal.spectrogram(eeg_data, fs=fs)
    # Calculate the frequency point that corresponds with the desired cut off frequencies
    pre_cut = int(len(f)*(pre_cut_off_freq / f[-1]))
    post_cut = int(len(f)*(post_cut_off_freq / f[-1]))
    plt.pcolormesh(t, f[pre_cut:post_cut], Sxx[pre_cut:post_cut], shading='gouraud')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (sec)")


# # Data Exploration - Begin actual analysis

# In[8]:


#GET EPOCHED DATA
#contains start time of event, and event type with corresponding eeg signals over the time span 
#start_time marks the 4s time period in which the subject needs to imagine left or right hand movement (marked by event_type)
epoch_train = pd.read_pickle('epoched_train.pkl')
epoch_test = pd.read_pickle('epoched_test.pkl')


# In[9]:


epoch_unfiltered = epoch_train.loc[:,['event_type','C3','Cz','C4']]


# In[10]:


# Get PSD averages for each channel for each event type (0=left or 1=right)
eeg_chans = ["C3", "Cz", "C4"] # 10-20 system 
eog_chans = ["EOG:ch01", "EOG:ch02", "EOG:ch03"] 
all_chans = eeg_chans + eog_chans
event_types = {0:"left", 1:"right"}
psd_averages_by_type = {}

for event_type in event_types.keys(): 
    psds_only_one_type={}
    freqs_only_one_type={}
    for i, row in epoch_train[epoch_train["event_type"] == event_type].iterrows(): 
        for ch in eeg_chans: 
            if ch not in psds_only_one_type: 
                psds_only_one_type[ch] = list()
                freqs_only_one_type[ch] = list()
            f, p = getMeanFreqPSD(row[ch])
            psds_only_one_type[ch].append(p)
            freqs_only_one_type[ch].append(f)
    avg_psds_one_type = {}
    for ch in eeg_chans:
        psds_only_one_type[ch] = np.array(psds_only_one_type[ch])
        avg_psds_one_type[ch] = np.mean(psds_only_one_type[ch], axis=0)
    psd_averages_by_type[event_type] = dict(avg_psds_one_type)


# In[11]:


#Looking at these, channel Cz experiences a peak in PSD between 5 and 8ish frequencies for left handed events
#C3 experiences a pretty extreme peaks between frequencies 6 and 15 for left handed events
#C4 experienes extreme peaks between frequencies between around 7 and 15 for right handed events, as well as between 15-25 
#Possibly focus on these intervals and channels as possible predictors for left/right handed movement

# View Average PSDs
for event_type in event_types.keys(): 
    for ch in eeg_chans[:]: 
        plotPSD(freqs_only_one_type[eeg_chans[0]][0], psd_averages_by_type[event_type][ch],pre_cut_off_freq=0, post_cut_off_freq=30, label=ch)

    plt.legend()
    plt.title("event type: " + event_types[event_type])
    plt.show()


# In[20]:


#signal length per channel per trial, dimensions of all training trials, dimensions of all testing trials (does not contain classification)
epoch_train.iloc[0]['C3'].shape, epoch_train.shape, epoch_test.shape 
epoch_train.shape


# ## Begin data filtering - Bandpass...this turned out to be unnecessary 

# In[21]:


''' Key: 
      BPTruncated(Tr/Tst): BandPass Truncated Train
      EBTr: Epoch Bandpass Train
      EBTst: Epoch Bandpass Tst
'''

##apply highpass filtering on signals for each trial (save in new df)
from scipy import signal

sos = signal.butter(2, (7,30), 'bandpass', fs=250, output='sos')

##train signals 
epoch_BPTruncatedTr = pd.DataFrame(index=range(3680),columns=['C3','Cz','C4']) ##3680 = length of 0th dimension for epoch data

for i in range(epoch_train.shape[0]):
  for j in ['C3','Cz','C4']:
    epoch_BPTruncatedTr.iloc[i].loc[j] = signal.sosfilt(sos, epoch_train.iloc[i].loc[j])

##test signals 

epoch_BPTruncatedTst = pd.DataFrame(index=range(3680),columns=['C3','Cz','C4'])

for i in range(epoch_test.shape[0]):
  for j in ['C3', 'Cz', 'C4']:
    epoch_BPTruncatedTst.iloc[i].loc[j] = signal.sosfilt(sos, epoch_test.iloc[i].loc[j])

##create new bandpass filtered dataframe for training (we will work with these dfs)
copy_EBTr = epoch_train.copy(deep = True)
copy_EBTr.drop(labels=['C3','Cz','C4'], axis="columns", inplace=True)
copy_EBTr[['C3','Cz','C4']] = epoch_BPTruncatedTr[['C3','Cz','C4']]
EBTr = copy_EBTr.copy(deep = True)
EBTr.head() #we can drop the EOG if needed

##create newbandpass filtered datarame for testing
copy_EBTst = epoch_test.copy(deep = True)
copy_EBTst.drop(labels = ['C3', 'Cz', 'C4'], axis = "columns", inplace = True)
copy_EBTst[['C3','Cz','C4']] = epoch_BPTruncatedTst[['C3','Cz','C4']]
EBTst = copy_EBTst.copy(deep = True)
EBTr


# In[22]:


## visualize the effect of highpass/lowpass/bandpass filtering on a single trial for a single channel

sos_high = signal.butter(2,30, 'highpass',fs=250, output='sos') ##create high signal transform
sos_low = signal.butter(2,30, 'lowpass', fs = 250, output = 'sos') ##create low signal transform
sos_band = signal.butter(2,(7,30), 'bandpass', fs = 250, output = 'sos') ##bandpass
sig = epoch_train.iloc[0,3] #sample data from single trial single channel epoch
filtered_high = signal.sosfilt(sos_high, sig)
filtered_low = signal.sosfilt(sos_low,sig)
filtered_band = signal.sosfilt(sos_band, sig)

f, (ax1, ax2, ax3,ax4) = plt.subplots(4, sharex=True, sharey=True, figsize = (15,8))
ax1.plot(epoch_train.iloc[0,3], color='b', label='Orig', alpha = 0.5)
ax1.legend(loc="upper right")
ax2.plot(filtered_high, color = 'r', label = 'high', alpha = 0.5)
ax2.legend(loc = 'upper right')
ax3.plot(filtered_low, color = 'g', label = 'low', alpha = 0.5)
ax3.legend(loc = 'upper right')
ax4.plot(filtered_band, color = 'y', label = 'band', alpha = 0.5)
ax4.legend(loc = 'upper right')


# ## Begin Applying Power Transformations on Epoched Signals

# In[23]:


epoch_channelsTr = pd.concat([EBTr.loc[:,['C3','Cz','C4']],EBTr.loc[:,'event_type']], axis = 1) ##using banded training df
epoch_channelsTst = EBTst.loc[:,['C3','Cz','C4']]
epoch_channelsTr.shape, epoch_channelsTst.shape
epoch_channelsTr.shape[0]


# ## custom functions used to extract magnitude from epoched data, and signal differences

# In[24]:


## magnitude/difference functions

#take magnitude of every channel entry
def get_magnitude(epoch_df, sig_strings):
  index= range(epoch_df.shape[0])
  dummy = pd.DataFrame(index = index, columns = sig_strings)
  for i in index: 
    for j in sig_strings:
      dummy.iloc[i][j]  = np.linalg.norm(epoch_df.loc[i][j])
  magnitude_df = pd.concat([epoch_df.drop(sig_strings, axis = 1),dummy], axis =1)
  return magnitude_df

## subtract channel signals and take magnitude
## C3 - C4, C3 - CZ, CZ - C4

#get pairwise differences and magnitude 
def signal_difference(epoch_df):
    sig_strings = ['C3','Cz','C4']
    sig_strings2 = ['C3-C4', 'C3-Cz', 'Cz - C4']
    channel_df = epoch_df.copy(deep = True)
    channel_df[sig_strings2[0]] = channel_df.loc[:,sig_strings[0]] - channel_df.loc[:,sig_strings[1]]
    channel_df[sig_strings2[1]] = channel_df.loc[:,sig_strings[0]] - channel_df.loc[:,sig_strings[2]]
    channel_df[sig_strings2[2]] = channel_df.loc[:,sig_strings[1]] - channel_df.loc[:,sig_strings[2]]
    magnitude = get_magnitude(channel_df, sig_strings2).drop(sig_strings, axis =1)
    
    return magnitude




train_magnitude = get_magnitude(epoch_channelsTr, ['C3','Cz', 'C4'])
train_signalDiff = signal_difference(epoch_channelsTr)

train_unfiltered_mag = get_magnitude(epoch_unfiltered,['C3','Cz','C4']) ######
train_unfiltered_sigdiff = signal_difference(epoch_unfiltered)          #####


# In[25]:


#see if data is normal or log transformed data is normal for C4
#other analysis tells us C3/Cz are normal

fig,ax = plt.subplots(nrows = 2, ncols = 2, figsize = (15,6))
x= range(train_magnitude.shape[0])
ax[0][0].hist(train_magnitude.iloc[:,3], alpha = 0.5)
ax[0][0].set_title('train_mag_C4')
ax[1][0].hist(np.log(np.array(train_magnitude.iloc[:,3], dtype=int)), alpha = 0.5)
ax[0][1].hist(train_signalDiff.iloc[:,3], alpha = 0.5, color = 'r')
ax[0][1].set_title('signal_diff_C4')
ax[1][1].hist(np.log(np.array(train_signalDiff.iloc[:,3], dtype=int)), alpha = 0.5, color = 'r')


#logged data seems approximately normal, let's log the magnitudes and work with this to better transform using StandardScaler(), as well as the unlogged dat
#we'll run a goodness of fit test on the PSDs perhaps


# In[26]:


#Log the magnitudes and signal_diff magnitudes
train_cols = train_magnitude.columns[1:4]
trainmag_as_int = np.array(train_magnitude.loc[:,train_cols], dtype=int)
train_logMag = pd.concat([pd.DataFrame(np.log(trainmag_as_int), columns = train_cols), train_magnitude.event_type], axis =1)

diff_cols = train_signalDiff.columns[1:4]

train_diff_int = np.array(train_signalDiff.loc[:,diff_cols], dtype=int)
train_diff_logmag = pd.concat([pd.DataFrame(np.log(train_diff_int), columns = diff_cols), train_signalDiff.event_type], axis =1)

####################
train_uncols = train_unfiltered_mag.columns[1:4]
trainmag_unf_int = np.array(train_unfiltered_mag.loc[:,train_uncols], dtype=int)
train_unf_log = pd.concat([pd.DataFrame(np.log(trainmag_unf_int), columns = train_uncols), train_unfiltered_mag.event_type], axis =1)

unf_diff = train_unfiltered_sigdiff.columns[1:4]
train_un_diff_int = np.array(train_unfiltered_sigdiff.loc[:,unf_diff], dtype = int)
train_undiff_log = pd.concat([pd.DataFrame(np.log(train_un_diff_int), columns = unf_diff), train_unfiltered_sigdiff.event_type], axis =1)

train_unfiltered_mag = get_magnitude(epoch_unfiltered,['C3','Cz','C4']) ######
train_unfiltered_sigdiff = signal_difference(epoch_unfiltered)   


# In[27]:


'''Magnitude Data we can use - Cleaned and ready to go'''

train_unfiltered_mag;
train_unfiltered_sigdiff;
train_unf_log;
train_undiff_log;


train_magnitude; 
train_signalDiff;
train_logMag;
train_diff_logmag;


# In[28]:


#Using this functions

'''
# Get Frequencies and mean PSDs from EEG data - this yeilds smoother PSDs because it averages the PSDs made from sliding windows. 
def getMeanFreqPSD(eeg_data, fs=eeg_fs):
    freq_mean, psd_mean = compute_spectrum(eeg_data, fs, method='welch', avg_type='mean', nperseg=fs*2)
    return freq_mean, psd_mean
'''

epoch_channelsTr.head() ##using these cleaned sets
epoch_channelsTst.head()
epoch_channelsTr.head()


# In[29]:


#total averages

psdavg_0 = psd_averages_by_type.get(0)  #dict
psdavg_1 = psd_averages_by_type.get(1) #dict


# ## custom functions to extract magnitude of PSDS and create 1hz log-magged PSDs, also to apply DWT and CWT 
# -note this code was reformatted so as to NOT take the logarithm on the PSDS before taking their magnitude, we discussed this in our discussion/conclusion

# In[ ]:




epoch_channelsTr0 = epoch_channelsTr[epoch_channelsTr.event_type == 0]
epoch_channelsTr1 = epoch_channelsTr[epoch_channelsTr.event_type ==1]

mean_psd_trial = getMeanFreqPSD(epoch_channelsTr0.iloc[0].loc['C3'])
trial_frequencies = mean_psd_trial[0] ##associated frequencies
log_trial_psd = np.log(mean_psd_trial[1])

lower_bound = min(trial_frequencies).astype(int)
upper_bound = max(trial_frequencies).astype(int)

#create a function for extracting the power (magnitude) of the mean psds for a single trial/channel by interval
#returns an array of three magnitudes

#can reformat this so that we split intervals into 2hz intervals
## now that this is iterating a bunch of times, its super slow woops
def extract_psd_mag(channel_array): 
    freq_and_psds = getMeanFreqPSD(channel_array)
    freq = freq_and_psds[0]
    psds = freq_and_psds[1]
    
    inter_arr = []
    ##so this should approximate whats going on in the previous code, but i dont trust it 100%
    for i in range(lower_bound, upper_bound):
        on = np.where(np.logical_or(freq == i, freq == i+0.5)) #(0, 0.5)
        on_slice = np.arange(on[0][0], on[0][1]+1)
        inter_arr.append(np.linalg.norm(psds[on_slice]))
        i+=1
    return np.array(inter_arr)
def extract_psd_logmag(channel_array): 
    freq_and_psds = getMeanFreqPSD(channel_array)
    freq = freq_and_psds[0]
    psds = freq_and_psds[1]
    
    inter_arr = []
    ##so this should approximate whats going on in the previous code, but i dont trust it 100%
    for i in range(lower_bound, upper_bound):
        on = np.where(np.logical_or(freq == i, freq == i+0.5)) #(0, 0.5)
        on_slice = np.arange(on[0][0], on[0][1]+1)
        inter_arr.append(np.log(np.linalg.norm(log_psds[on_slice])))
        i+=1
    return np.array(inter_arr)


    
##create a function that iterates through an only-channel dataframe and outputs the psd magnitudes for each channel and each interval
inter_length = upper_bound #if we were doing 2 hz intervals it would be upper-lower/2 rounded up
def create_interval_psds(epoch_df_no_label):
  #create empty data frame N-samples x (3chan x 125 intervals)
    channel_list = epoch_df_no_label.columns
    int_columns = []
    for i in channel_list:
        for j in range(inter_length):
            int_columns.append(i+'_inter_' + str(j+1))
            
    interval_df = pd.DataFrame(index = range(epoch_df_no_label.shape[0]), columns = int_columns) 

    for i,p in zip(channel_list,[0,inter_length,inter_length*2]): #so for [(c3, 0), (cz,3), (c4,6) change to [0,inter_length,inter_length*2]
        for j in range(epoch_df_no_label.shape[0]):
            psd_extract = extract_psd_mag(epoch_df_no_label.iloc[j].loc[i])
            for k in range(inter_length):
                interval_df.head()
                interval_df.iloc[j].iloc[k+p] = psd_extract[k]
  
    return interval_df
def create_interval_logpsds(epoch_df_no_label):
  #create empty data frame N-samples x (3chan x 125 intervals)
    channel_list = epoch_df_no_label.columns
    int_columns = []
    for i in channel_list:
        for j in range(inter_length):
            int_columns.append(i+'_inter_' + str(j+1))
            
    interval_df = pd.DataFrame(index = range(epoch_df_no_label.shape[0]), columns = int_columns) 

    for i,p in zip(channel_list,[0,inter_length,inter_length*2]): #so for [(c3, 0), (cz,3), (c4,6) change to [0,inter_length,inter_length*2]
        for j in range(epoch_df_no_label.shape[0]):
            psd_extract = extract_psd_logmag(epoch_df_no_label.iloc[j].loc[i])
            for k in range(inter_length):
                interval_df.head()
                interval_df.iloc[j].iloc[k+p] = psd_extract[k]
  
    return interval_df


##Now we need to create a N_sample x 10 dataframe, 3 features for each channel C3:(psd_mag1,psd_mag2,psd_mag3)

######################################## begin CWT/DWT transform functions
def extract_cwt_mag(channel_array):
    dt = 0.004  # 100 Hz sampling
    frequencies = pywt.scale2frequency('morl', np.arange(6,30)) / dt
    frequency_len = len(frequencies)
    wavelet_arr, freqs = pywt.cwt(data = sig, scales = frequencies, wavelet = 'morl')
    mag_arr =  []
    for i in range(frequency_len):
        mag_arr.append(np.log(np.linalg.norm(wavelet_arr[i])))
    return mag_arr

def create_interval_cwt(epoch_df_no_label):
  #create empty data frame N-samples x (3chan x 125 intervals)
    dt = .004 ##250 hz
    frequency_len  = len(pywt.scale2frequency('morl', np.arange(6,30))/dt) ##constant
    channel_list = epoch_df_no_label.columns
    int_columns = []
    for i in channel_list:
        for j in range(frequency_len):
            int_columns.append(i+'cwt' + str(j+1))
            
    interval_df = pd.DataFrame(index = range(epoch_df_no_label.shape[0]), columns = int_columns) 

    for i,p in zip(channel_list,[0,frequency_len,frequency_len*2]): 
        for j in range(epoch_df_no_label.shape[0]):
            cwt_extract = extract_cwt_mag(epoch_df_no_label.iloc[j].loc[i])
            for k in range(frequency_len):
                interval_df.iloc[j].iloc[k+p] = cwt_extract[k]
  
    return interval_df

def extract_dwt_mag(channel_array):
    a,b = pywt.dwt(data = channel_array, wavelet = 'db4', mode='symmetric', axis=-1)
    mags = [np.log(np.linalg.norm(a)), np.log(np.linalg.norm(b))]
    return mags

dwt_length = 2 ##constant
def create_interval_dwt(epoch_df_no_label):
    channel_list = epoch_df_no_label.columns
    int_columns = []
    for i in channel_list:
        for j in range(dwt_length): 
            int_columns.append(i+'dwt' + str(j+1))
    interval_df = pd.DataFrame(index = range(epoch_df_no_label.shape[0]), columns = int_columns) 
    for i,p in zip(channel_list,[0,dwt_length, dwt_length*2]): 
        for j in range(epoch_df_no_label.shape[0]):
            dwt_extract = extract_dwt_mag(epoch_df_no_label.iloc[j].loc[i])
            for k in range(dwt_length):
                interval_df.iloc[j].iloc[k+p] = dwt_extract[k]
    return interval_df
def extract_multiple_dwt_mag(channel_array):
    coefs_approx_list = pywt.wavedec(data = channel_array, wavelet ='db4', mode = 'symmetric', level = 5, axis = -1,)
    mags = []
    for i in range(len(coefs_approx_list)):
        mags.append(np.log(np.linalg.norm(coefs_approx_list[i])))
    return mags ##48 length list
mult_dwt_length = 5
def create_interval_dwt_multiple(epoch_df_no_label):
    channel_list = epoch_df_no_label.columns
    int_columns = []
    for i in channel_list:
        for j in range(mult_dwt_length):
            if(j%2 == 0):
                int_columns.append(i + "approx" + str(j+1))
            else:
                int_columns.append(i + "detail" + str(j+1))
    interval_df = pd.DataFrame(index = range(epoch_df_no_label.shape[0]), columns = int_columns)
    for i,p in zip(channel_list,[0,mult_dwt_length,mult_dwt_length*2]):
        for j in range(epoch_df_no_label.shape[0]):
            mult_dwt_extract = extract_multiple_dwt_mag(epoch_df_no_label.iloc[j].loc[i])
            for k in range(mult_dwt_length):
                interval_df.iloc[j].iloc[k+p] = mult_dwt_extract[k]
    return interval_df
##################################
    


# ### CREATE CWT/DWT DF 

# In[ ]:


## unfiltered


# In[481]:


CWT_unfilt = create_interval_cwt(epoch_unfiltered.loc[:,['C3','C4','Cz']])


# In[483]:


DWT_unfilt = create_interval_dwt(epoch_unfiltered.loc[:,['C3','C4','Cz']])


# In[484]:


DWT_unfil_mult = create_interval_dwt_multiple(epoch_unfiltered.loc[:,['C3','Cz','C4']])


# In[485]:


WT_unfilt_combined = pd.concat([CWT_unfilt,DWT_unfilt],axis = 1)


# In[488]:


WT_unfilt_combined = pd.concat([WT_combined, epoch_unfiltered.loc[:,'event_type']], axis = 1)


# In[489]:


WT_unfilt_combined = WT_unfilt_combined.astype(float)
WT_unfilt_combined['event_type'] = WT_unfilt_combined['event_type'].astype(int)


# In[492]:


WT_mult_unfilt_combined = pd.concat([CWT_unfilt, DWT_unfil_mult], axis = 1)
WT_mult_unfilt_combined = pd.concat([WT_mult_unfilt_combined, epoch_unfiltered.loc[:,'event_type']], axis = 1)
WT_mult_unfilt_combined = WT_mult_unfilt_combined.astype(float)
WT_mult_unfilt_combined['event_type'] = WT_mult_unfilt_combined['event_type'].astype(int) 


# In[479]:


## FILTERED BELOW


# In[ ]:


##Create Separate CWT/DWT dfs
CWT = create_interval_cwt(EBTr.loc[:,['C3','C4','Cz']])


# In[323]:


DWT = create_interval_dwt(EBTr.loc[:,['C3','Cz','C4']])


# In[333]:


DWT_mult = create_interval_dwt_multiple(EBTr.loc[:,['C3','Cz','C4']]) ##using 5 levels


# In[335]:


WT_mult_combined = pd.concat([CWT,DWT_mult], axis = 1)
WT_mult_combined = pd.concat([WT_mult_combined, EBTr.loc[:,'event_type']], axis = 1)


# In[336]:


WT_mult_combined = WT_mult_combined.astype(float) 
WT_mult_combined['event_type'] = WT_mult_combined['event_type'].astype(int)


# In[239]:


WT_combined= pd.concat([CWT,DWT], axis =1)
WT_combined = pd.concat([WT_combined, EBTr.loc[:,'event_type']], axis = 1)


# In[274]:


WT_combined = WT_combined.astype(float) 
WT_combined['event_type'] = WT_combined['event_type'].astype(int)


# In[517]:


del WT_unfilt_combined['event_type']


# In[519]:


WT_unfilt_combined = pd.concat([WT_unfilt_combined, epoch_unfiltered.loc[:,'event_type']], axis = 1)


# In[520]:


WT_unfilt_combined['event_type'] = WT_unfilt_combined['event_type'].astype(int)


# ## UNFILTERED PSD

# In[38]:


##create new dataframe for PSD (mean PSD over intervals, 0-6.5, 7 - 30, 30.5 - ...)
##Using EBTr,EBTst, could have used epoch_channelsTr0/1 but I suck

PSD_df = pd.concat([create_interval_psds(EBTr.loc[:,['C3', 'Cz','C4']]), EBTr.loc[:,'event_type']], axis =1 ) #train
PSD_df_0 = PSD_df[PSD_df.loc[:,'event_type'] == 0]
PSD_df_1 = PSD_df[PSD_df.loc[:,'event_type'] == 1]


# In[117]:


PSD_mag_unf = pd.concat([create_interval_psds(epoch_unfiltered.loc[:,['C3','Cz','C4']]), epoch_unfiltered.loc[:,'event_type']], axis = 1)


# In[54]:


x,y = getMeanFreqPSD(epoch_unfiltered.iloc[0].loc['C3'])


# In[109]:


freq_and_psds = getMeanFreqPSD(epoch_unfiltered.iloc[0].loc['C3'])


# **below is where I realized we went wrong with double-logging the PSDS**

# In[132]:


onehzvec = extract_psd_logmag(epoch_unfiltered.iloc[0].loc['C3'])
fig, ax = plt.subplots()
ax.plot(x,y, alpha = 0.5,label = 'MPSD C3 Trial 1 over 125 hz')
ax.scatter(np.arange(125), onehzvec, s = .8, color = 'red')
ax.legend()


# In[43]:


len(x)


# In[439]:


PSD_unf0 = PSD_log_unf[PSD_log_unf.loc[:,'event_type']==0]
PSD_unf1 = PSD_log_unf[PSD_log_unf.loc[:,'event_type']==1]


# In[122]:


PSD_mag_unf0 = PSD_mag_unf[PSD_mag_unf.loc[:,'event_type']==0]
PSD_mag_unf1 = PSD_mag_unf[PSD_mag_unf.loc[:,'event_type']==1]
tp_mag = {}
for i in list(PSD_mag_unf1.columns[0:326]):
    tp_mag[i] = scipy.stats.ttest_rel(PSD_mag_unf1.loc[:,i],PSD_mag_unf0.loc[:,i])
signif_mag_unf = []
for inter, results in tp_mag.items(): 
    if (results[1]<.05):
        signif_mag_unf.append(inter)
PSD_final_mag_unf = PSD_mag_unf.loc[:,signif_mag_unf]
PSD_final_mag_unf = PSD_final_mag_unf.astype(float)
PSD_final_mag_unf['event_type'] = epoch_unfiltered['event_type']
PSD_final_mag_unf['event_type']  = PSD_final_mag_unf['event_type'].astype(int)


# In[41]:


PSD_train = PSD_df


# In[44]:


## Now, we can run t-tests (paired) to figure out which PSD intervals of which channels are actually statistically significant (non-logged)

from scipy import stats
import statistics

tp = {}
for i in list(PSD_df_1.columns[0:326]):
  tp[i] = scipy.stats.ttest_rel(PSD_df_1.loc[:,i],PSD_df_0.loc[:,i])


# In[468]:


tp_unf = {}
for i in list(PSD_unf1.columns[0:326]):
    tp_unf[i] = scipy.stats.ttest_rel(PSD_unf1.loc[:,i],PSD_unf0.loc[:,i])


# In[469]:


signif_unf = []
for inter, results in tp_unf.items(): 
    if (results[1]<.05):
        signif_unf.append(inter)


# In[47]:


signif = []
for inter, results in tp.items(): 
    if (results[1]<.05):
        signif.append(inter)


# In[471]:


PSD_final_log_unf = PSD_log_unf.loc[:,signif_unf]


# In[475]:


PSD_final_log_unf.loc[:,'event_type'] = PSD_train.loc[:,'event_type']


# In[80]:


PSD_final_train = PSD_train.loc[:,signif] ##get significant columns


# In[85]:


PSD_final_train.loc[:,'event_type'] = PSD_train.loc[:,'event_type'] ##append targets to train


# In[496]:


##BELOW IS ALL OUR CLEANED DATA
##GUESS WE'RE ONLY USING THE TRAIN SETS

WT_unfilt_combined;

WT_mult_unfilt_combined;

train_unfiltered_mag;

train_unfiltered_sigdiff;

train_unf_log;

train_undiff_log;

PSD_final_log_unf;
####

PSD_final_train; 
PSD_final_tst;

train_magnitude; 
test_magnitude;

train_signalDiff;
test_signalDiff;

train_logMag; ##use these instead of un-logged
test_logMag;

train_diff_logmag; ##use these instead of un-logged
test_diff_logmag;


WT_mult_combined;
WT_combined;

PSD_final_train;
train_logMag;
train_diff_logmag;


# In[ ]:


##don't know why entries are type "object", need to convert if we use numpy operations to prevent error
PSD_final_train = PSD_final_train.astype(float) 
PSD_final_train['event_type'] = PSD_final_train['event_type'].astype(int)


# In[505]:


train_unf_log = train_unf_log.astype(float)
train_unf_log['event_type'] = epoch_unfiltered['event_type'].astype(int)
train_undiff_log = train_undiff_log.astype(float)
train_undiff_log['event_type'] = epoch_unfiltered['event_type'].astype(int)


# In[500]:


PSD_final_log_unf = PSD_final_log_unf.astype(float)
PSD_final_log_unf['event_type'] = epoch_unfiltered['event_type'].astype(int)


# In[580]:


train_logMag = train_logMag.astype(float)
train_logMag['event_type'] = train_logMag['event_type'].astype(int)

train_diff_logmag = train_diff_logmag.astype(float)
train_diff_logmag['event_type'] = train_diff_logmag['event_type'].astype(int)


# ## Begin Testing - SVM - KNN - DECISION TREE - KMEANS - SPECTRAL

# ### SVM

# In[129]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score

from sklearn.model_selection import train_test_split

from sklearn.calibration import CalibratedClassifierCV


# In[130]:


def SVM(dataset, iter):
    tst_predictions = pd.DataFrame()
    validation_scores = np.zeros(3)
    params_best = {}
    y_true = {}
    acc_scores = np.zeros(3)
    precision_scores = np.zeros(3)
    recall_scores = np.zeros(3)
    f1_scores = np.zeros(3)
    
    
    for i in [1,2,3]:
        
        X_tr, X_tst, y_tr, y_tst = train_test_split(dataset.loc[:,dataset.columns != 'event_type'], dataset.loc[:,'event_type'], test_size= .20, random_state = i, shuffle = True)
        pipeline = Pipeline([('std', StandardScaler()), ('classifier', SVC(max_iter = iter))])
        parameter_grid = ({'classifier__kernel':['rbf'],'classifier__C': np.power(10.,np.arange(-6,6)), 'classifier__gamma': [0.001,0.005,0.01,0.05,0.1,0.5,1,2]},
                             {'classifier__kernel': ['poly'], 'classifier__degree': [2,3], 'classifier__C': np.power(10.,np.arange(-6,6))})
        five_fold = KFold(n_splits = 5, shuffle = True)
        print(X_tr.shape[0]) ##track progress
        GS = GridSearchCV(estimator = pipeline, param_grid = parameter_grid, cv = five_fold, n_jobs = -1, verbose = 0, refit = True)
        best_fit = GS.fit(X_tr,y_tr)
        print(best_fit.best_score_) ##print statement to track progress
        validation_scores[i-1]= best_fit.best_score_
        predictions = best_fit.predict(X_tst)
    

        tst_predictions = pd.concat([tst_predictions, pd.DataFrame(predictions, columns = [str(i)])], axis = 1)
        params_best['best' + str(i)]= best_fit.best_params_
        y_true['y' + str(i)] = y_tst
        acc_scores[i-1] = accuracy_score(y_tst, predictions)
        precision_scores[i-1] = precision_score(y_tst, predictions)
        recall_scores[i-1] = recall_score(y_tst, predictions)
        f1_scores[i-1] = f1_score(y_tst,predictions)
        data = pd.DataFrame(best_fit.cv_results_)

        dict_of_vals = {'best_validation': validation_scores, 'best_params': params_best, 'tst_predictions': tst_predictions,  'y_true': y_true, 'acc_scores': acc_scores, 'precision_scores': precision_scores, 'recall_scores': recall_scores, 'f1_scores': f1_scores}

    return dict_of_vals


# In[133]:


svm_mag_unf = SVM(PSD_final_mag_unf, iter = -1)


# In[137]:


np.mean(svm_mag_unf.get('acc_scores'))


# In[138]:


knn_mag_unf = KNN(PSD_final_mag_unf)


# In[139]:


np.mean(knn_mag_unf.get('acc_scores'))


# In[140]:


DT_mag_unf = DT(PSD_final_mag_unf)


# In[141]:


np.mean(DT_mag_unf.get('acc_scores'))


# In[572]:


svm_unfpsd = SVM(PSD_final_log_unf, iter = -1)


# In[604]:


unf_psd= svm_unfpsd.get('acc_scores')


# In[618]:


np.mean(unf_psd)


# In[573]:


svm_unflog = SVM(train_unf_log, iter = 1000000)


# In[605]:


unf_log = svm_unflog.get('acc_scores')


# In[619]:


np.mean(unf_log)


# In[574]:


svm_diffunf= SVM(train_undiff_log, iter = 1000000)


# In[607]:


svm_diff_unf = svm_diffunf.get('acc_scores')


# In[620]:


np.mean(svm_diff_unf)


# In[575]:


svm_WTunfilt = SVM(WT_unfilt_combined, iter = -1)


# In[608]:


svm_WT_unf = svm_WTunfilt.get('acc_scores')


# In[621]:


np.mean(svm_WT_unf)


# In[576]:


svm_WTCombo = SVM(WT_mult_unfilt_combined, iter = -1)


# In[609]:


svm_WT_unf_multi = svm_WTCombo.get('acc_scores')


# In[622]:


np.mean(svm_WT_unf_multi)


# In[650]:


svm_WTCombo


# In[577]:


### fitlered testing


# In[578]:


svm_PSD_fil = SVM(PSD_final_train, iter = -1)


# In[612]:


svm_psd_fil = svm_PSD_fil.get('acc_scores')


# In[623]:


np.mean(svm_psd_fil)


# In[581]:


svm_logmag = SVM(train_logMag, iter = 1000000)


# In[613]:


svm_mag_fil = svm_logmag.get('acc_scores')


# In[624]:


np.mean(svm_mag_fil)


# In[582]:


svm_diff_logmag = SVM(train_diff_logmag, iter = 1000000)


# In[614]:


svm_diff_fil = svm_diff_logmag.get('acc_scores')


# In[625]:


np.mean(svm_diff_fil)


# In[597]:


svm_WT = SVM(WT_combined, iter = -1)


# In[615]:


svm_WT_fil = svm_WT.get('acc_scores')


# In[626]:


np.mean(svm_WT_fil)


# In[598]:


svm_mult_WT = SVM(WT_mult_combined, iter =-1)


# In[616]:


svm_mult_fil = svm_mult_WT.get('acc_scores')


# In[627]:


np.mean(svm_mult_fil)


# ## KNN

# In[135]:


def KNN(dataset):
    grid = np.arange(1,200,10)
    tst_predictions = pd.DataFrame()
    validation_scores = np.zeros(3)
    params_best = {}
    y_true = {}
    acc_scores = np.zeros(3)
    precision_scores = np.zeros(3)
    recall_scores = np.zeros(3)
    f1_scores = np.zeros(3)


    for j in [1,2,3]:
        
        X_tr, X_tst, y_tr, y_tst = train_test_split(dataset.loc[:,dataset.columns != 'event_type'], dataset.loc[:,'event_type'], test_size= .2, random_state = j, shuffle = True)
        sca = StandardScaler()
        X_tr = sca.fit_transform(X_tr) 
        X_tst = sca.transform(X_tst)
        five_fold = KFold(n_splits = 5, shuffle = True)
        knn = KNeighborsClassifier()
        parameters = {'n_neighbors': list(grid), 'weights': ['uniform','distance']}
        
        print(X_tr.shape[0]) #print statement for progress
        GS = GridSearchCV(estimator = knn, param_grid = parameters, cv = five_fold, n_jobs = 4, verbose = 0, refit = True)
        best_fit = GS.fit(X_tr,y_tr)
        validation_scores[j-1]= best_fit.best_score_
        print(best_fit.best_score_) #print statement for progress
        predictions = best_fit.predict(X_tst)
        
        tst_predictions = pd.concat([tst_predictions, pd.DataFrame(predictions, columns = [str(j)])], axis = 1)
        params_best['best' + str(j)]= best_fit.best_params_
        y_true['y' + str(j)] = y_tst
        acc_scores[j-1] = accuracy_score(y_tst, predictions)
        precision_scores[j-1] = precision_score(y_tst, predictions)
        recall_scores[j-1] = recall_score(y_tst, predictions)
        f1_scores[j-1] = f1_score(y_tst,predictions)
    
    dict_of_vals = {'best_validation': validation_scores, 'best_params': params_best, 'tst_predictions': tst_predictions, 'y_true': y_true, 'acc_scores': acc_scores, 'precision_scores': precision_scores, 'recall_scores': recall_scores, 'f1_scores': f1_scores}
        
    
        
       
    return dict_of_vals
    


# In[524]:


knn_unfpsd = KNN(PSD_final_log_unf)


# In[628]:


np.mean(knn_unfpsd.get('acc_scores'))


# In[525]:


knn_unflog = KNN(train_unf_log)


# In[630]:


np.mean(knn_unflog.get('acc_scores'))


# In[526]:


knn_diffunf= KNN(train_undiff_log)


# In[631]:


np.mean(knn_diffunf.get('acc_scores'))


# In[527]:


knn_WTunfilt = KNN(WT_unfilt_combined)


# In[632]:


np.mean(knn_WTunfilt.get('acc_scores'))


# In[528]:


knn_WTCombo = KNN(WT_mult_unfilt_combined)


# In[633]:


np.mean(knn_WTCombo.get('acc_scores'))


# In[ ]:


### filtered below


# In[298]:


knn_PSD = KNN(PSD_final_train)


# In[634]:


np.mean(knn_PSD.get('acc_scores'))


# In[299]:


knn_diffmag= KNN(train_diff_logmag)


# In[635]:


np.mean(knn_diffmag.get('acc_scores'))


# In[300]:


knn_logmag = KNN(train_logMag)


# In[636]:


np.mean(knn_logmag.get('acc_scores'))


# In[276]:


knn_WT = KNN(WT_combined)


# In[637]:


np.mean(knn_WT.get('acc_scores'))


# In[339]:


KNN_mult_WT = KNN(WT_mult_combined)


# In[638]:


np.mean(KNN_mult_WT.get('acc_scores'))


# ## Decision Trees

# In[136]:


def DT(dataset): ##takes full train set
    tst_predictions = pd.DataFrame()
    validation_scores = np.zeros(3)
    params_best = {}
    y_true = {}
    acc_scores = np.zeros(3)
    precision_scores = np.zeros(3)
    recall_scores = np.zeros(3)
    f1_scores = np.zeros(3)


    for j in [1,2,3]:
        
        pipeline = Pipeline([('std', StandardScaler()), ('classifier', DecisionTreeClassifier())])
        parameter_grid = ({'classifier__criterion':['gini'],'classifier__max_depth': np.arange(1,8), 'classifier__min_samples_leaf': np.arange(1,6), 'classifier__min_samples_split': np.arange(2,8)},
                          {'classifier__criterion': ['entropy'], 'classifier__max_depth': np.arange(1,8), 'classifier__min_samples_leaf': np.arange(1,6), 'classifier__min_samples_split': np.arange(2,8)})
        
        X_tr, X_tst, y_tr, y_tst = train_test_split(dataset.loc[:,dataset.columns != 'event_type'], dataset.loc[:,'event_type'], test_size= .2, random_state = j, shuffle = True)
        five_fold = KFold(n_splits = 5, shuffle = True)
        
        print(X_tr.shape[0]) #print statement for progress
        GS = GridSearchCV(estimator = pipeline, param_grid = parameter_grid, cv = five_fold, n_jobs = 4, verbose = 0, refit = True)
        best_fit = GS.fit(X_tr,y_tr)
        validation_scores[j-1]= best_fit.best_score_
        print(best_fit.best_score_) #print statement for progress
        predictions = best_fit.predict(X_tst)
        
        tst_predictions = pd.concat([tst_predictions, pd.DataFrame(predictions, columns = [str(j)])], axis = 1)
        params_best['best' + str(j)]= best_fit.best_params_
        y_true['y' + str(j)] = y_tst
        acc_scores[j-1] = accuracy_score(y_tst, predictions)
        precision_scores[j-1] = precision_score(y_tst, predictions)
        recall_scores[j-1] = recall_score(y_tst, predictions)
        f1_scores[j-1] = f1_score(y_tst,predictions)
    
    dict_of_vals = {'best_validation': validation_scores, 'best_params': params_best, 'tst_predictions': tst_predictions, 'y_true': y_true, 'acc_scores': acc_scores, 'precision_scores': precision_scores, 'recall_scores': recall_scores,  'f1_scores': f1_scores}
        
    
        
       
    return dict_of_vals


# In[529]:


DT_unfpsd = DT(PSD_final_log_unf)


# In[639]:


np.mean(DT_unfpsd.get('acc_scores'))


# In[530]:


DT_unflog =DT(train_unf_log)


# In[640]:


np.mean(DT_unflog.get('acc_scores'))


# In[531]:


DT_diffunf= DT(train_undiff_log)


# In[641]:


np.mean(DT_diffunf.get('acc_scores'))


# In[532]:


DT_WTunfilt = DT(WT_unfilt_combined)


# In[642]:


np.mean(DT_WTunfilt.get('acc_scores'))


# In[533]:


DT_WTCombo = DT(WT_mult_unfilt_combined)


# In[643]:


np.mean(DT_WTCombo.get('acc_scores'))


# In[ ]:


## filtered begins


# In[301]:


dt_PSD = DT(PSD_final_train)


# In[645]:


np.mean(dt_PSD.get('acc_scores'))


# In[302]:


dt_diffmag= DT(train_diff_logmag)


# In[646]:


np.mean(dt_diffmag.get('acc_scores'))


# In[303]:


dt_logmag = DT(train_logMag)


# In[647]:


np.mean(dt_logmag.get('acc_scores'))


# In[277]:


dt = DT(WT_combined)


# In[648]:


np.mean(dt.get('acc_scores'))


# In[340]:


DT_mult_WT = DT(WT_mult_combined)


# In[649]:


np.mean(DT_mult_WT.get('acc_scores'))


# In[362]:


from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import v_measure_score


# ## K means

# In[389]:


def KM(dataset, tol):
    X_tr = dataset.loc[:,dataset.columns!= 'event_type']
    y_tst = dataset.loc[:,'event_type']
    kmean = sklearn.cluster.KMeans(n_clusters = 2, max_iter = 500)
    kmean.fit(X_tr)
    fowlkes = fowlkes_mallows_score(y_tst, kmean.labels_)
    return 'fowlkes', fowlkes


# In[535]:


km_wt = KM(WT_unfilt_combined, .0001)
km_wt_mult = KM(WT_mult_unfilt_combined,.0001)
km_wt
km_wt_mult


# In[538]:


km_wt


# ## Spectral

# In[380]:


def SC(dataset, gamma):
    X_tr = dataset.loc[:,dataset.columns!= 'event_type']
    y_tst = dataset.loc[:,'event_type']
    spec = SpectralClustering(n_clusters = 2, gamma = gamma, assign_labels = 'discretize')
    fit= spec.fit(X_tr)
    fowlkes = fowlkes_mallows_score(y_tst, fit.labels_)
    return 'fowlkes', fowlkes, 


# In[536]:


sc_wt = SC(WT_unfilt_combined, 1)


# In[537]:


sc_wt_combined = SC(WT_mult_unfilt_combined,1)


# In[539]:


sc_wt


# In[540]:


sc_wt_combined


# In[ ]:




