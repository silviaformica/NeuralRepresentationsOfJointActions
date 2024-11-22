# -*- coding: utf-8 -*-
"""

JOINT DRAWING EXPERIMENT

Fully automatized preprocessing of EEG data
    - load data and define montage
    - preliminary rereferencing to POz (due to Biosemi CMS-DRL)
    - filter [.1, 40 Hz]
    - pyprep NoisyChannels for automatic detection of bad channels
    - epoching
    - ICA and automatic detection of eye movements
    - interpolation of bad channels
    - average reference
    - application of final rejection criteria (150 µV)
    
@author: Silvia Formica
silvia.formica@hu-berlin.de
"""

# Importing modules

import os
import numpy as np
import pandas as pd
import mne
from pyprep.find_noisy_channels import NoisyChannels
import sys


# os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials


#%%

cd = os.getcwd()

#######################################
## LOOPING ACROSS PARTICIPANTS
## Processing one participant at a time
#######################################

for subj in subjlist:
     
    data_path = cd + '/sub-' + subj + '/'
    
    if not os.path.isdir(data_path + 'Results'):
        os.mkdir(data_path + 'Results')
    
    # reading raw data
    raw = mne.io.read_raw_bdf(data_path + 'eeg/Subj' + subj + '.bdf', preload=True)
    
    # adjusting montage info
    # create a list of 'EEG' channels
    types = ['eeg']*73
    # change elements in that list that aren't 'EEG' channels
    types[-1] = 'stim'; types[-2] = 'emg'; types[-3] = 'emg'; types[-4] = 'emg'; types[-5] = 'emg';
    types[-6] = 'emg'; types[-7] = 'emg'; types[-8] = 'misc'; types[-9] = 'misc'
    # create a dictionary of channel names and types
    chtypes_dict = dict(zip(raw.ch_names, types))

    # update the channel types of our RAW data set
    raw.set_channel_types(chtypes_dict)

    # Apply the montage to the raw data
    montage = mne.channels.make_standard_montage('biosemi64')
    raw.set_montage(montage)    
    
    # mne.viz.plot_montage(montage, scale_factor = 10, to_sphere = True)
    # raw.plot_sensors(to_sphere = False, show_names = True)
    
    ## Re referencing to POz
    raw = raw.set_eeg_reference(ref_channels = ['POz'], ch_type = 'eeg')
    
    # Filtering
    filt = raw.filter(.1, 40, n_jobs = 1, fir_design = 'firwin')
    
    # bad channels detection
    nd = NoisyChannels(filt)
    nd.find_all_bads(channel_wise=True)

    # Check channels that go bad and add it in the info
    filt.info['bads'] = nd.get_bads()

    # To see a summary of why channels were marked as bad:
    # a = nd.get_bads(verbose = True)    


    os.chdir(data_path + '/Results/')

    # Saving file with removed channels and motivation
    # this bit of code saves the console output to the desired file
    tem = sys.stdout
    sys.stdout = f = open('InterpolatedChannels.txt', 'a')
    nd.get_bads(verbose = True)
    sys.stdout = tem
    f.close()


    # read events from the filtered dataset
    events = mne.find_events(raw, stim_channel='Status', shortest_event=1)

    ev = events[np.where([True if (str(x[2]).endswith('1') and str(x[1]).endswith('0')) else False for idx, x in enumerate(events[:, :])])]
    
    # getting metadata
    metadata = pd.DataFrame()
    behav_folder = [x for x in os.listdir(data_path + 'beh/') if str(x).startswith('Subj') ]
    files = os.listdir(data_path + 'beh/' + behav_folder[0])
    for idx_f, file in enumerate(files):
        if file.endswith('.csv') and file.startswith('Task'):
            thisf = pd.read_csv(data_path + 'beh/' + behav_folder[0] + '/' + file)
            metadata = pd.concat([metadata, thisf])

    # reordering
    metadata = metadata.sort_values('trial_sort')
    
    
    # Epoching
    epochs = mne.Epochs(filt, ev, proj = False, baseline = None, detrend = 1, tmin = -.5, tmax = 3, metadata = metadata, preload = True)


    # Downsampling to 128 Hz
    epochs.decimate(8)
    
    # create ICA object with desired parameters (99% of variance)
    ica = mne.preprocessing.ICA(n_components = 0.999999, max_iter = 500)
    # do ICA decomposition
    ica.fit(epochs)


    # automatically detect eye movement components based on EOG
    eog_indices_v, eog_scores_v = ica.find_bads_eog(epochs, ch_name = ['EXG5'])
    
    # ica.plot_scores(eog_scores_v)
    
    ica.exclude = np.unique(eog_indices_v)

    # apply ICA to data
    ica.apply(epochs)

    # save discarded ICA components
    tem = sys.stdout
    sys.stdout = f = open('ICA.txt', 'a')
    print(ica.exclude)
    sys.stdout = tem
    f.close()
 

    # interpolating bad electrodes
    epochs = epochs.interpolate_bads()
    
    # average ref
    epochs = epochs.set_eeg_reference(ref_channels = 'average', ch_type = 'eeg')
    
    # rejecting trials in which 150 µV threshold is exceeded
    reject_criteria = dict(eeg=150e-6)      # this is peak to peak!
    
    epochs.drop_bad(reject=reject_criteria)

    ## saving clean epochs
    epochs.save('epochs-epo.fif', overwrite=True)
    

#%%

'''
Summary measures on preprocessing
'''

## Number of ICA components removed per participant

ICA = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    
    with open(data_path + 'ICA.txt') as f:
        lines = f.read()
        
        print(lines)
        
    comp = lines.split('[')[1].split(']')[0]
    
    if len(comp) == 4:
        first = comp.split(', ')[0]
        second = comp.split(', ')[1]
        
        comp = list([int(first), int(second)])

    elif len(comp) == 7:
        first = comp.split(', ')[0]
        second = comp.split(', ')[1]
        third = comp.split(', ')[2]
        
        comp = list([int(first), int(second), int(third)])
        
    elif len(comp) == 1:
        comp = list(comp)
    
    else:
        print('Problem subj' + subj)
        
    ICA.append(comp)


n_comps = [len(x) for x in ICA]

print(
    "discarded ICA components: %s, ±%s" % (np.stack(n_comps).mean(),
    np.stack(n_comps).std()
        )
    )

#%%

## Number of interpolated channels per participant

Chan = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    
    with open(data_path + 'InterpolatedChannels.txt') as f:
        lines = f.read()
        
        splits = lines.split('\'')
        this_p = []
        for idx_c, chan in enumerate(splits):
            if chan != ', ' and len(chan)<= 3:
                this_p.append(chan)
                
    Chan.append(this_p)
                    
n_chan = [len(x) for x in Chan]
               
print(
    "Interpolated channels: %s, ±%s" % (np.stack(n_chan).mean(),
    np.stack(n_chan).std()
        )
    )

#%%

## number of discarded epochs per participants

Discarded = []
slows = []
ignored = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    
    epochs = mne.read_epochs(data_path + 'epochs-epo.fif', preload = False)

    n_slow = (epochs.metadata.starting_time_too_slow == 1).sum()
    slows.append(n_slow)

    epochs = epochs[epochs.metadata.starting_time_too_slow != 1]
    
    # log = np.unique(epochs.drop_log, return_counts = True)

    disc = np.asarray([1 if (len(x)>0) and (x != ('IGNORED', )) else 0 for x in epochs.drop_log]).sum()
    
    Discarded.append(disc)

    ignored.append(np.asarray([1 if x == ('IGNORED', ) else 0 for x in epochs.drop_log]).sum())

    
print(
    "Discarded trials: %s, ±%s" % (np.stack(Discarded).mean(),
    np.stack(Discarded).std()
        )
    )


percent = [Discarded[x]*100/576 for x in range(len(subjlist))]

print(
    "Percent discarded trials: %s, ±%s" % (np.stack(percent).mean(),
    np.stack(percent).std()
        )
    )

#%%

## Number of clean trials entering the analyses

N = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    
    epochs = mne.read_epochs(data_path + 'epochs-epo.fif', preload = False)

    epochs = epochs[epochs.metadata.starting_time_too_slow != 1]
    
    meta = epochs.metadata
    meta = meta.reset_index()
    
    meta['cong'] = ['cong' if meta.loc[t, 'cue_participant_letter'] == meta.loc[t, 'cue_coagent_letter'] else 'incong' for t in range(len(meta))]

    
    N.append(meta.groupby(['task', 'cong']).count()['cue_participant'].mean())

print(
    "N trials per task x cong: %s, ±%s" % (np.stack(N).mean(),
    np.stack(N).std()
        )
    )


percent = [N[x]*100/144 for x in range(len(subjlist))]

print(
    "Percent discarded trials: %s, ±%s" % (np.stack(percent).mean(),
    np.stack(percent).std()
        )
    )



#%%

'''
PLOTS
Manuscript partial Figure 2
'''
import matplotlib.pyplot as plt
import seaborn as sns


# loading individual data and computing grandaverage
All = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    
    All.append(mne.read_epochs(data_path + 'epochs-epo.fif'))

erp = mne.grand_average([x.average().apply_baseline((-.2, 0)) for x in All])


data = erp.copy().pick(['POz']).crop(0, 2).data
time = erp.copy().pick(['POz']).crop(0, 2).times


f, ax = plt.subplots(1, 1) 
sns.despine()
ax.plot(time, data.squeeze(), color = 'k', linewidth = 4)
ax.set_yticklabels([])
ax.set_xticks(np.linspace(0, 2, 11), labels = [str(x) for x in list(np.round(np.linspace(0, 2, 11), decimals = 1))], fontsize= 16, font = 'Arial')
plt.title('POz', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10, loc = 'left')
ax.set_xlabel('Time from cue onset', fontsize= 18, font = 'Arial')
ax.axvline(0, color = 'gray', linestyle = '--')
ax.axvline(0.2, color = 'gray', linestyle = '--')
ax.axvline(0.4, color = 'gray', linestyle = '--')
ax.axvline(0.6, color = 'gray', linestyle = '--')
ax.axvline(0.8, color = 'gray', linestyle = '--')
ax.axvline(1, color = 'gray', linestyle = '--')
ax.axvline(1.2, color = 'gray', linestyle = '--')
ax.axvline(1.4, color = 'gray', linestyle = '--')
ax.axvline(1.6, color = 'gray', linestyle = '--')
ax.axvline(1.8, color = 'gray', linestyle = '--')
ax.axvline(2, color = 'gray', linestyle = '--')

