# -*- coding: utf-8 -*-
"""

JOINT DRAWING EXPERIMENT

Temporal decoding TEST
1. Loading and analyzing decoding of congruent and incongruent combinations
2. Loading and analyzing decoding of participant's and partner's movements
    
@author: Silvia Formica
silvia.formica@hu-berlin.de
"""

# Importing modules

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from mne.stats import permutation_cluster_1samp_test
from scipy import stats as stats

#%%

os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials


#%%

'''
TEMPORAL DECODING
'''

# Loading one epoch for the info
epochs = mne.read_epochs('D:/JointRSA/JointDrawing/sub-00/Results/epochs-epo.fif')
epochs = epochs.pick('eeg')

start = np.where(epochs.times == 0)[0][0]
end = np.where(epochs.times == 2)[0][0]
epochs = epochs.crop(tmin = 0, tmax = 2)


#%%

'''
Congruent combinations
'''
# joint task
ccdd_joint_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    ccdd_joint_time.append(np.load(data_path + '/ccdd_joint_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
ccdd_joint_time_plot = pd.DataFrame()
ccdd_joint_time_plot['Accuracy'] = np.hstack(ccdd_joint_time)
ccdd_joint_time_plot['Time'] = np.tile(timep, len(subjlist))
ccdd_joint_time_plot['Subject'] = np.repeat(subjlist, len(ccdd_joint_time[0]))
ccdd_joint_time_plot['Task'] = 'Joint'


# paralle task
ccdd_parallel_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    ccdd_parallel_time.append(np.load(data_path + '/ccdd_parallel_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
ccdd_parallel_time_plot = pd.DataFrame()
ccdd_parallel_time_plot['Accuracy'] = np.hstack(ccdd_parallel_time)
ccdd_parallel_time_plot['Time'] = np.tile(timep, len(subjlist))
ccdd_parallel_time_plot['Subject'] = np.repeat(subjlist, len(ccdd_parallel_time[0]))
ccdd_parallel_time_plot['Task'] = 'Parallel'



ccdd = pd.concat([ccdd_joint_time_plot, ccdd_parallel_time_plot])



#%%
## STATS

# Joint

X = np.stack(ccdd_joint_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_ccdd_joint_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')
    

# Parallel

X = np.stack(ccdd_parallel_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_ccdd_parallel_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


## Difference

X = np.stack(ccdd_joint_time) - np.stack(ccdd_parallel_time)
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_ccdd_diff = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')




#%%
## Plots

ccdd = pd.concat([ccdd_joint_time_plot, ccdd_parallel_time_plot])

fig, ax = plt.subplots(nrows=2, sharex = True, gridspec_kw={'height_ratios': [10, 1]})
sns.lineplot(data = ccdd, x = 'Time', y = 'Accuracy', hue = 'Task', palette = ['green', 'orange'], errorbar = 'se', ax=ax[0])
#plt.plot(scores.Timepoint, scores.Accuracy)
ax[0].axhline(.5, color = 'k', linestyle = '--',)
ax[0].axvline(0, color = 'gray', linestyle = '-')
ax[0].axvline(0.2, color = 'gray', linestyle = '-')
# ax[0].axvline(2, color = 'gray', linestyle = '-')
# ax[0].axvline(2.2, color = 'gray', linestyle = '-')

# ax[0].axvspan(0, .2, color = 'gray', alpha = .25)
# ax[0].axvspan(2, 2.2, color = 'gray', alpha = .25)

ax[0].legend()
ax[0].set_ylabel('Classification accuracy')
# ax.set_title('Own Movement - Joint')

lowelim = ax[0].get_ylim()[1]-.005

for i_c, c in enumerate(clu_ccdd_joint_time[1]):
    if clu_ccdd_joint_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim, clu_ccdd_joint_time[1][i_c][0].stop-clu_ccdd_joint_time[1][i_c][0].start), x = timep[clu_ccdd_joint_time[1][i_c][0].start-1: clu_ccdd_joint_time[1][i_c][0].stop-1], color='green',  marker = '.')


for i_c, c in enumerate(clu_ccdd_parallel_time[1]):
    if clu_ccdd_parallel_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim-0.0025, clu_ccdd_parallel_time[1][i_c][0].stop-clu_ccdd_parallel_time[1][i_c][0].start), x = timep[clu_ccdd_parallel_time[1][i_c][0].start-1: clu_ccdd_parallel_time[1][i_c][0].stop-1], color='orange',  marker = '.')

plt.subplot(212)
for i_c, c in enumerate(clu_ccdd_diff[1]):
    if clu_ccdd_diff[2][i_c] <= 0.05:
        
        plt.scatter(y = np.repeat(0.5, clu_ccdd_diff[1][i_c][0].stop-clu_ccdd_diff[1][i_c][0].start), x = timep[clu_ccdd_diff[1][i_c][0].start-1: clu_ccdd_diff[1][i_c][0].stop-1], color='k',  marker = '.')

ax[1].set_title('Difference')

plt.suptitle('Congruent combinations')
plt.show()



#%%

'''
Incongruent combinations
'''

# joint task
cddc_joint_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    cddc_joint_time.append(np.load(data_path + '/cddc_joint_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
cddc_joint_time_plot = pd.DataFrame()
cddc_joint_time_plot['Accuracy'] = np.hstack(cddc_joint_time)
cddc_joint_time_plot['Time'] = np.tile(timep, len(subjlist))
cddc_joint_time_plot['Subject'] = np.repeat(subjlist, len(cddc_joint_time[0]))
cddc_joint_time_plot['Task'] = 'Joint'


# parallel task
cddc_parallel_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    cddc_parallel_time.append(np.load(data_path + 'cddc_parallel_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
cddc_parallel_time_plot = pd.DataFrame()
cddc_parallel_time_plot['Accuracy'] = np.hstack(cddc_parallel_time)
cddc_parallel_time_plot['Time'] = np.tile(timep, len(subjlist))
cddc_parallel_time_plot['Subject'] = np.repeat(subjlist, len(cddc_parallel_time[0]))
cddc_parallel_time_plot['Task'] = 'Parallel'


cddc = pd.concat([cddc_joint_time_plot, cddc_parallel_time_plot])


#%%
## STATS

# Joint

X = np.stack(cddc_joint_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_cddc_joint_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')
    

# Parallel

X = np.stack(cddc_parallel_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_cddc_parallel_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


## Difference

X = np.stack(cddc_joint_time) - np.stack(cddc_parallel_time)
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_cddc_diff = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')




#%%
## Plots

cddc = pd.concat([cddc_joint_time_plot, cddc_parallel_time_plot])

fig, ax = plt.subplots(nrows=2, sharex = True, gridspec_kw={'height_ratios': [10, 1]})
sns.lineplot(data = cddc, x = 'Time', y = 'Accuracy', hue = 'Task', palette = ['green', 'orange'], errorbar = 'se', ax=ax[0])
#plt.plot(scores.Timepoint, scores.Accuracy)
ax[0].axhline(.5, color = 'k', linestyle = '--',)
ax[0].axvline(0, color = 'gray', linestyle = '-')
ax[0].axvline(0.2, color = 'gray', linestyle = '-')
# ax[0].axvline(2, color = 'gray', linestyle = '-')
# ax[0].axvline(2.2, color = 'gray', linestyle = '-')

# ax[0].axvspan(0, .2, color = 'gray', alpha = .25)
# ax[0].axvspan(2, 2.2, color = 'gray', alpha = .25)

ax[0].legend()
ax[0].set_ylabel('Classification accuracy')
# ax.set_title('Own Movement - Joint')

lowelim = ax[0].get_ylim()[1]-.005

for i_c, c in enumerate(clu_cddc_joint_time[1]):
    if clu_cddc_joint_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim, clu_cddc_joint_time[1][i_c][0].stop-clu_cddc_joint_time[1][i_c][0].start), x = timep[clu_cddc_joint_time[1][i_c][0].start-1: clu_cddc_joint_time[1][i_c][0].stop-1], color='green',  marker = '.')


for i_c, c in enumerate(clu_cddc_parallel_time[1]):
    if clu_cddc_parallel_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim-0.0025, clu_cddc_parallel_time[1][i_c][0].stop-clu_cddc_parallel_time[1][i_c][0].start), x = timep[clu_cddc_parallel_time[1][i_c][0].start-1: clu_cddc_parallel_time[1][i_c][0].stop-1], color='orange',  marker = '.')

plt.subplot(212)
for i_c, c in enumerate(clu_cddc_diff[1]):
    if clu_cddc_diff[2][i_c] <= 0.05:
        
        plt.scatter(y = np.repeat(0.5, clu_cddc_diff[1][i_c][0].stop-clu_cddc_diff[1][i_c][0].start), x = timep[clu_cddc_diff[1][i_c][0].start-1: clu_cddc_diff[1][i_c][0].stop-1], color='k',  marker = '.')

ax[1].set_title('Difference')

plt.suptitle('Incongruent combinations')
plt.show()

#%%

'''
Participant's movement 
'''
# Joint task
Me_joint_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    Me_joint_time.append(np.load(data_path + '/Me_joint_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
Me_joint_time_plot = pd.DataFrame()
Me_joint_time_plot['Accuracy'] = np.hstack(Me_joint_time)
Me_joint_time_plot['Time'] = np.tile(timep, len(subjlist))
Me_joint_time_plot['Subject'] = np.repeat(subjlist, len(Me_joint_time[0]))
Me_joint_time_plot['Task'] = 'Joint'


# Parallel task
Me_parallel_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    Me_parallel_time.append(np.load(data_path + '/Me_parallel_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
Me_parallel_time_plot = pd.DataFrame()
Me_parallel_time_plot['Accuracy'] = np.hstack(Me_parallel_time)
Me_parallel_time_plot['Time'] = np.tile(timep, len(subjlist))
Me_parallel_time_plot['Subject'] = np.repeat(subjlist, len(Me_parallel_time[0]))
Me_parallel_time_plot['Task'] = 'Parallel'


#%%
## STATS

# Joint

X = np.stack(Me_joint_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_Me_joint_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')
    

# Parallel

X = np.stack(Me_parallel_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_Me_parallel_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


## Difference

X = np.stack(Me_joint_time) - np.stack(Me_parallel_time)
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_Me_diff = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')



#%%
## Plots

Me = pd.concat([Me_joint_time_plot, Me_parallel_time_plot])

fig, ax = plt.subplots(nrows=2, sharex = True, gridspec_kw={'height_ratios': [10, 1]})
sns.lineplot(data = Me, x = 'Time', y = 'Accuracy', hue = 'Task', palette = ['green', 'orange'], errorbar = 'se', ax=ax[0])
#plt.plot(scores.Timepoint, scores.Accuracy)
ax[0].axhline(.5, color = 'k', linestyle = '--',)
ax[0].axvline(0, color = 'gray', linestyle = '-')
ax[0].axvline(0.2, color = 'gray', linestyle = '-')
# ax[0].axvline(2, color = 'gray', linestyle = '-')
# ax[0].axvline(2.2, color = 'gray', linestyle = '-')

# ax[0].axvspan(0, .2, color = 'gray', alpha = .25)
# ax[0].axvspan(2, 2.2, color = 'gray', alpha = .25)

ax[0].legend()
ax[0].set_ylabel('Classification accuracy')
# ax.set_title('Own Movement - Joint')

lowelim = ax[0].get_ylim()[1]-.01

for i_c, c in enumerate(clu_Me_joint_time[1]):
    if clu_Me_joint_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim, clu_Me_joint_time[1][i_c][0].stop-clu_Me_joint_time[1][i_c][0].start), x = timep[clu_Me_joint_time[1][i_c][0].start-1: clu_Me_joint_time[1][i_c][0].stop-1], color='green',  marker = '.')


for i_c, c in enumerate(clu_Me_parallel_time[1]):
    if clu_Me_parallel_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim-0.005, clu_Me_parallel_time[1][i_c][0].stop-clu_Me_parallel_time[1][i_c][0].start), x = timep[clu_Me_parallel_time[1][i_c][0].start-1: clu_Me_parallel_time[1][i_c][0].stop-1], color='orange',  marker = '.')

plt.subplot(212)
for i_c, c in enumerate(clu_Me_diff[1]):
    if clu_Me_diff[2][i_c] <= 0.05:
        
        plt.scatter(y = np.repeat(0.5, clu_Me_diff[1][i_c][0].stop-clu_Me_diff[1][i_c][0].start), x = timep[clu_Me_diff[1][i_c][0].start-1: clu_Me_diff[1][i_c][0].stop-1], color='k',  marker = '.')

ax[1].set_title('Difference')

plt.suptitle('Participants movement')
plt.show()



#%%

'''
Partner's movement
'''

# Joint task
You_joint_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    You_joint_time.append(np.load(data_path + '/You_joint_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
You_joint_time_plot = pd.DataFrame()
You_joint_time_plot['Accuracy'] = np.hstack(You_joint_time)
You_joint_time_plot['Time'] = np.tile(timep, len(subjlist))
You_joint_time_plot['Subject'] = np.repeat(subjlist, len(You_joint_time[0]))
You_joint_time_plot['Task'] = 'Joint'


# Parallel task
You_parallel_time = []

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    
    You_parallel_time.append(np.load(data_path + '/You_parallel_time.npy').mean(axis = 0).mean(axis =0))

timep = epochs.times
You_parallel_time_plot = pd.DataFrame()
You_parallel_time_plot['Accuracy'] = np.hstack(You_parallel_time)
You_parallel_time_plot['Time'] = np.tile(timep, len(subjlist))
You_parallel_time_plot['Subject'] = np.repeat(subjlist, len(You_parallel_time[0]))
You_parallel_time_plot['Task'] = 'Parallel'

You = pd.concat([You_joint_time_plot, You_parallel_time_plot])


#%%
## STATS

# Joint

X = np.stack(You_joint_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_You_joint_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')
    

# Parallel

X = np.stack(You_parallel_time) - .5
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 0  # f-test, so tail > 0
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_You_parallel_time = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')


## Difference

X = np.stack(You_joint_time) - np.stack(You_parallel_time)
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., len(subjlist) - 1)

tail = 1
n_permutations = 1000  # Save some time (the test won't be too sensitive ...)

print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu_You_diff = \
    permutation_cluster_1samp_test(
    X, tail=tail, n_jobs=4, threshold = t_threshold,
    n_permutations=n_permutations, buffer_size=None, out_type='mask')



#%%
## Plots

You = pd.concat([You_joint_time_plot, You_parallel_time_plot])

fig, ax = plt.subplots(nrows=2, sharex = True, gridspec_kw={'height_ratios': [10, 1]})
sns.lineplot(data = You, x = 'Time', y = 'Accuracy', hue = 'Task', palette = ['green', 'orange'], errorbar = 'se', ax=ax[0])
#plt.plot(scores.Timepoint, scores.Accuracy)
ax[0].axhline(.5, color = 'k', linestyle = '--',)
ax[0].axvline(0, color = 'gray', linestyle = '-')
ax[0].axvline(0.2, color = 'gray', linestyle = '-')
# ax[0].axvline(2, color = 'gray', linestyle = '-')
# ax[0].axvline(2.2, color = 'gray', linestyle = '-')

# ax[0].axvspan(0, .2, color = 'gray', alpha = .25)
# ax[0].axvspan(2, 2.2, color = 'gray', alpha = .25)

ax[0].legend()
ax[0].set_ylabel('Classification accuracy')
# ax.set_title('Own Movement - Joint')

lowelim = ax[0].get_ylim()[1]-.01

for i_c, c in enumerate(clu_You_joint_time[1]):
    if clu_You_joint_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim, clu_You_joint_time[1][i_c][0].stop-clu_You_joint_time[1][i_c][0].start), x = timep[clu_You_joint_time[1][i_c][0].start-1: clu_You_joint_time[1][i_c][0].stop-1], color='green',  marker = '.')


for i_c, c in enumerate(clu_You_joint_time[1]):
    if clu_You_joint_time[2][i_c] <= 0.05:
        
        ax[0].scatter(y = np.repeat(lowelim-0.005, clu_You_joint_time[1][i_c][0].stop-clu_You_joint_time[1][i_c][0].start), x = timep[clu_You_joint_time[1][i_c][0].start-1: clu_You_joint_time[1][i_c][0].stop-1], color='orange',  marker = '.')

plt.subplot(212)
for i_c, c in enumerate(clu_You_diff[1]):
    if clu_You_diff[2][i_c] <= 0.05:
        
        plt.scatter(y = np.repeat(0.5, clu_You_diff[1][i_c][0].stop-clu_You_diff[1][i_c][0].start), x = timep[clu_You_diff[1][i_c][0].start-1: clu_You_diff[1][i_c][0].stop-1], color='k',  marker = '.')

ax[1].set_title('Difference')

plt.suptitle('Partners movement')
plt.show()
