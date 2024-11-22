# -*- coding: utf-8 -*-
"""

JOINT DRAWING EXPERIMENT

Spatial decoding TEST
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
from scipy import stats as stats
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

#%%

os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials

#%%

'''
SPATIAL DECODING
'''

# Loading one epoch for the info
epochs = mne.read_epochs('D:/JointRSA/JointDrawing/sub-00/Results/epochs-epo.fif')
epochs = epochs.pick('eeg')
# computing adjacency

adj = mne.channels.find_ch_adjacency(epochs.info, ch_type = 'eeg')[0]


#%%

'''
Congruent combinations
'''

# joint
ccdd_joint_space = []

for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    ccdd_joint_space.append(np.load(data_path + 'ccdd_joint_space.npy').mean(axis = 0).mean(axis =0))

evoked_joint = mne.EvokedArray(np.stack(ccdd_joint_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)


# parallel
ccdd_parallel_space = []
for subj in subjlist:  
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'  
    ccdd_parallel_space.append(np.load(data_path + '/ccdd_parallel_space.npy').mean(axis = 0).mean(axis =0))

evoked_parallel = mne.EvokedArray(np.stack(ccdd_parallel_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)


# difference
evoked_diff = mne.EvokedArray(np.stack(ccdd_joint_space).mean(axis = 0).reshape(64,1) - np.stack(ccdd_parallel_space).mean(axis = 0).reshape(64,1), epochs.info)


## STATS
clu_ccdd_joint_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(ccdd_joint_space)-.5,
                                             adjacency = adj, n_permutations = 1000)

mask_joint = np.zeros((64, 1))

for c in range(len(clu_ccdd_joint_space[2])):
    if clu_ccdd_joint_space[2][c] < 0.05:
        mask_joint[clu_ccdd_joint_space[1][c][0]] = 1


clu_ccdd_parallel_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(ccdd_parallel_space)-.5,
                                             adjacency = adj, n_permutations = 1000)

mask_parallel = np.zeros((64, 1))

for c in range(len(clu_ccdd_parallel_space[2])):
    if clu_ccdd_parallel_space[2][c] < 0.05:
        mask_parallel[clu_ccdd_parallel_space[1][c][0]] = 1


clu_ccdd_diff_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(ccdd_joint_space)-np.stack(ccdd_parallel_space),
                                             adjacency = adj, n_permutations = 1000)

mask_diff = np.zeros((64, 1))

for c in range(len(clu_ccdd_diff_space[2])):
    if clu_ccdd_diff_space[2][c] < 0.05:
        mask_diff[clu_ccdd_diff_space[1][c][0]] = 1
    
'''
Plot
'''

data_to_plot = np.stack([np.stack(ccdd_joint_space).mean(axis = 0)-.5, np.stack(ccdd_parallel_space).mean(axis = 0)-.5, np.stack(ccdd_joint_space).mean(axis = 0) - np.stack(ccdd_parallel_space).mean(axis = 0)]).T

evoked_plot = mne.EvokedArray(data_to_plot, epochs.info)
evoked_plot.info.times = ['Joint', 'Parallel', 'Difference']

# fig, axs = plt.subplots(nrows = 1, ncols = 3)
# evoked_plot.plot_topomap(times = evoked_plot.times, scalings = 100, show_names = False, mask = np.stack([mask_joint, mask_parallel,      mask_diff]).squeeze().T,  mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
#         linewidth=0, markersize=8), axes = axs, time_format = '', colorbar = False)

# fig.suptitle('ccdd')
# axs[0].set_title('Joint')
# axs[1].set_title('Parallel')
# axs[2].set_title('Difference')


# alternative

joint_plot = (np.stack(ccdd_joint_space).mean(axis = 0)).reshape(64, 1)
joint_plot = mne.EvokedArray(joint_plot, epochs.info)
joint_plot.info.times = ['Joint']
cmp = LinearSegmentedColormap.from_list('joint_cmap', ['white','#CF3939'])
norm = mpl.colors.Normalize(vmin=50)
joint_plot.plot_topomap(times = joint_plot.times, scalings = 100,show_names = False, mask = mask_joint, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Joint', colorbar = True, cnorm = norm, vlim = [50,None])

plt.savefig("D:\JointRSA\Paper\Figures/joint_cong.png", bbox_inches="tight", dpi=300)


parallel_plot = (np.stack(ccdd_parallel_space).mean(axis = 0)).reshape(64, 1)
parallel_plot = mne.EvokedArray(parallel_plot, epochs.info)
parallel_plot.info.times = ['Parallel']
cmp = LinearSegmentedColormap.from_list('parallel_cmap', ['white','#198695'])
norm = mpl.colors.Normalize(vmin=50)
parallel_plot.plot_topomap(times = parallel_plot.times, scalings = 100,show_names = False, mask = mask_parallel, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Parallel', colorbar = True, cnorm = norm, vlim = [50,None])

plt.savefig("D:\JointRSA\Paper\Figures/parallel_cong.png", bbox_inches="tight", dpi=300)


diff_plot = (np.stack(ccdd_joint_space).mean(axis = 0) - np.stack(ccdd_parallel_space).mean(axis = 0)).reshape(64, 1)
diff_plot = mne.EvokedArray(diff_plot, epochs.info)
diff_plot.info.times = ['Difference']
cmp = LinearSegmentedColormap.from_list('diff_cmap', ['#198695', 'white', '#CF3939'])
norm = mpl.colors.Normalize()
diff_plot.plot_topomap(times = diff_plot.times, scalings = 100,show_names = False, mask = mask_diff, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Difference', colorbar = True)


#%%
'''
Incongruent combinations
'''
# joint
cddc_joint_space = []
for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    cddc_joint_space.append(np.load(data_path + '/cddc_joint_space.npy').mean(axis = 0).mean(axis =0))

evoked_joint = mne.EvokedArray(np.stack(cddc_joint_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)

# parallel
cddc_parallel_space = []
for subj in subjlist:   
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'   
    cddc_parallel_space.append(np.load(data_path + '/cddc_parallel_space.npy').mean(axis = 0).mean(axis =0))

evoked_parallel = mne.EvokedArray(np.stack(cddc_parallel_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)

# difference
evoked_diff = mne.EvokedArray(np.stack(cddc_joint_space).mean(axis = 0).reshape(64,1) - np.stack(cddc_parallel_space).mean(axis = 0).reshape(64,1), epochs.info)


## STATS

clu_cddc_joint_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(cddc_joint_space)-.5,
                                             adjacency = adj, n_permutations = 1000)

mask_joint = np.zeros((64, 1))

for c in range(len(clu_cddc_joint_space[2])):
    if clu_cddc_joint_space[2][c] < 0.05:
        mask_joint[clu_cddc_joint_space[1][c][0]] = 1


clu_cddc_parallel_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(cddc_parallel_space)-.5,
                                             adjacency = adj, n_permutations = 1000)

mask_parallel = np.zeros((64, 1))

for c in range(len(clu_cddc_parallel_space[2])):
    if clu_cddc_parallel_space[2][c] < 0.05:
        mask_parallel[clu_cddc_parallel_space[1][c][0]] = 1
        
        
clu_cddc_diff_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(cddc_joint_space)-np.stack(cddc_parallel_space),
                                             adjacency = adj, n_permutations = 1000)

mask_diff = np.zeros((64, 1))

for c in range(len(clu_cddc_diff_space[2])):
    if clu_cddc_diff_space[2][c] < 0.05:
        mask_diff[clu_cddc_diff_space[1][c][0]] = 1
    
    
'''
Plot
'''

data_to_plot = np.stack([np.stack(cddc_joint_space).mean(axis = 0)-.5, np.stack(cddc_parallel_space).mean(axis = 0)-.5, np.stack(cddc_joint_space).mean(axis = 0) - np.stack(cddc_parallel_space).mean(axis = 0)]).T

evoked_plot = mne.EvokedArray(data_to_plot, epochs.info)
evoked_plot.info.times = ['Joint', 'Parallel', 'Difference']

# fig, axs = plt.subplots(nrows = 1, ncols = 3)
# evoked_plot.plot_topomap(times = evoked_plot.times, scalings = 100, show_names = False, mask = np.stack([mask_joint, mask_parallel,      mask_diff]).squeeze().T,  mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
#         linewidth=0, markersize=8), axes = axs, time_format = '', colorbar = False)

# fig.suptitle('cddc')
# axs[0].set_title('Joint')
# axs[1].set_title('Parallel')
# axs[2].set_title('Difference')

# alternative

joint_plot = (np.stack(cddc_joint_space).mean(axis = 0)).reshape(64, 1)
joint_plot = mne.EvokedArray(joint_plot, epochs.info)
# joint_plot.info.times = ['Joint']
cmp = LinearSegmentedColormap.from_list('joint_cmap', ['white','#CF3939'])
norm = mpl.colors.Normalize(vmin=50)
joint_plot.plot_topomap(times = joint_plot.times, scalings = 100,show_names = False, mask = mask_joint, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Joint', colorbar = True, cnorm = norm, vlim = [50,None])

plt.savefig("D:\JointRSA\Paper\Figures/joint_incong.png", bbox_inches="tight", dpi=300)



parallel_plot = (np.stack(cddc_parallel_space).mean(axis = 0)).reshape(64, 1)
parallel_plot = mne.EvokedArray(parallel_plot, epochs.info)
parallel_plot.info.times = ['Parallel']
cmp = LinearSegmentedColormap.from_list('parallel_cmap', ['white','#198695'])
norm = mpl.colors.Normalize(vmin=50)
parallel_plot.plot_topomap(times = parallel_plot.times, scalings = 100,show_names = False, mask = mask_parallel, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Parallel', colorbar = True, cnorm = norm, vlim = [50,None])

plt.savefig("D:\JointRSA\Paper\Figures/parallel_incong.png", bbox_inches="tight", dpi=300)


diff_plot = (np.stack(cddc_joint_space).mean(axis = 0) - np.stack(cddc_parallel_space).mean(axis = 0)).reshape(64, 1)
diff_plot = mne.EvokedArray(diff_plot, epochs.info)
diff_plot.info.times = ['Difference']
cmp = LinearSegmentedColormap.from_list('diff_cmap', ['#198695', 'white', '#CF3939'])
# norm = mpl.colors.Normalize()
diff_plot.plot_topomap(times = diff_plot.times, scalings = 100,show_names = False, mask = mask_diff, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Difference', colorbar = True)



#%%

'''
Participant movement
'''
# joint task
Me_joint_space = []
for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    Me_joint_space.append(np.load(data_path + '/Me_joint_space.npy').mean(axis = 0).mean(axis =0))

evoked_joint = mne.EvokedArray(np.stack(Me_joint_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)

# parallel task
Me_parallel_space = []
for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    Me_parallel_space.append(np.load(data_path + '/Me_parallel_space.npy').mean(axis = 0).mean(axis =0))

evoked_parallel = mne.EvokedArray(np.stack(Me_parallel_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)

evoked_diff = mne.EvokedArray(np.stack(Me_joint_space).mean(axis = 0).reshape(64,1) - np.stack(Me_parallel_space).mean(axis = 0).reshape(64,1), epochs.info)



## STATS
# joint task
clu_Me_joint_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(Me_joint_space)-.5,
                                             adjacency = adj, n_permutations = 1000)

mask_joint = np.zeros((64, 1))

for c in range(len(clu_Me_joint_space[2])):
    if clu_Me_joint_space[2][c] < 0.05:
        mask_joint[clu_Me_joint_space[1][c][0]] = 1

# parallel task
clu_Me_parallel_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(Me_parallel_space)-.5,
                                             adjacency = adj, n_permutations = 1000)

mask_parallel = np.zeros((64, 1))

for c in range(len(clu_Me_parallel_space[2])):
    if clu_Me_parallel_space[2][c] < 0.05:
        mask_parallel[clu_Me_parallel_space[1][c][0]] = 1

# difference
clu_Me_diff_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(Me_joint_space)-np.stack(Me_parallel_space),
                                             adjacency = adj, n_permutations = 1000)

mask_diff = np.zeros((64, 1))

for c in range(len(clu_Me_diff_space[2])):
    if clu_Me_diff_space[2][c] < 0.05:
        mask_diff[clu_Me_diff_space[1][c][0]] = 1
     
'''
Plot
'''
# Plotting in jet colormap and subplots
data_to_plot = np.stack([np.stack(Me_joint_space).mean(axis = 0)-.5, np.stack(Me_parallel_space).mean(axis = 0)-.5, np.stack(Me_joint_space).mean(axis = 0) - np.stack(Me_parallel_space).mean(axis = 0)]).T

evoked_plot = mne.EvokedArray(data_to_plot, epochs.info)
evoked_plot.info.times = ['Joint', 'Parallel', 'Difference']

# fig, axs = plt.subplots(nrows = 1, ncols = 3)
# evoked_plot.plot_topomap(times = evoked_plot.times, scalings = 100, show_names = False, mask = np.stack([mask_joint, mask_parallel,mask_diff]).squeeze().T,  mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',linewidth=0, markersize=8), axes = axs, time_format = '', colorbar = False)

# fig.suptitle('Me')
# axs[0].set_title('Joint')
# axs[1].set_title('Parallel')
# axs[2].set_title('Difference')


# Plotting with custom colormaps

joint_plot = (np.stack(Me_joint_space).mean(axis = 0)).reshape(64, 1)
joint_plot = mne.EvokedArray(joint_plot, epochs.info)
joint_plot.info.times = ['Joint']
cmp1 = LinearSegmentedColormap.from_list('joint_cmap', ['white','#CF3939'], N = 50)
norm = mpl.colors.Normalize(vmin=50)
f = joint_plot.plot_topomap(times = joint_plot.times, scalings = 100,show_names = False, mask = mask_joint, cmap = cmp1, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Joint', colorbar = True, cnorm = norm, vlim = [50,None])

plt.savefig("D:\JointRSA\Paper\Figures/joint_me.png", bbox_inches="tight", dpi=300)


parallel_plot = (np.stack(Me_parallel_space).mean(axis = 0)).reshape(64, 1)
parallel_plot = mne.EvokedArray(parallel_plot, epochs.info)
parallel_plot.info.times = ['Parallel']
cmp2 = LinearSegmentedColormap.from_list('parallel_cmap', ['white','#198695'], N = 50)
norm = mpl.colors.Normalize(vmin=50)
parallel_plot.plot_topomap(times = parallel_plot.times, scalings = 100,show_names = False, mask = mask_parallel, cmap = cmp2, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Parallel', colorbar = True, cnorm = norm, vlim = [50,None])

plt.savefig("D:\JointRSA\Paper\Figures/parallel_me.png", bbox_inches="tight", dpi=300)


diff_plot = (np.stack(Me_joint_space).mean(axis = 0) - np.stack(Me_parallel_space).mean(axis = 0)).reshape(64, 1)
diff_plot = mne.EvokedArray(diff_plot, epochs.info)
diff_plot.info.times = ['Difference']
cmp3 = LinearSegmentedColormap.from_list('diff_cmap', ['#198695','white','#CF3939'], N = 50)
norm = mpl.colors.Normalize()
diff_plot.plot_topomap(times = diff_plot.times, scalings = 100,show_names = False, mask = mask_diff, cmap = cmp3, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Difference', colorbar = True)

#%%

'''
Partner's movement
'''

# Joint task
You_joint_space = []

for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    You_joint_space.append(np.load(data_path + '/You_joint_space.npy').mean(axis = 0).mean(axis =0))


evoked_joint = mne.EvokedArray(np.stack(You_joint_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)


# parallel task
You_parallel_space = []

for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatial_temporal_decoding_norm/'
    You_parallel_space.append(np.load(data_path + '/You_parallel_space.npy').mean(axis = 0).mean(axis =0))

evoked_parallel = mne.EvokedArray(np.stack(You_parallel_space).mean(axis = 0).reshape(64,1)-.5, epochs.info)


# difference    
evoked_diff = mne.EvokedArray(np.stack(You_joint_space).mean(axis = 0).reshape(64,1) - np.stack(You_parallel_space).mean(axis = 0).reshape(64,1), epochs.info)


## STATS

#joint
clu_You_joint_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(You_joint_space)-.5,
                                             adjacency = adj, n_permutations = 1000)

mask_joint = np.zeros((64, 1))

for c in range(len(clu_You_joint_space[2])):
    if clu_You_joint_space[2][c] < 0.05:
        mask_joint[clu_You_joint_space[1][c][0]] = 1


# parallel
clu_You_parallel_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(You_parallel_space)-.5,
                                             adjacency = adj, n_permutations = 1000)


mask_parallel = np.zeros((64, 1))

for c in range(len(clu_You_parallel_space[2])):
    if clu_You_parallel_space[2][c] < 0.05:
        mask_parallel[clu_You_parallel_space[1][c][0]] = 1

# difference
clu_You_diff_space = mne.stats.spatio_temporal_cluster_1samp_test(np.stack(You_joint_space)-np.stack(You_parallel_space),
                                             adjacency = adj, n_permutations = 1000)

mask_diff = np.zeros((64, 1))

for c in range(len(clu_You_diff_space[2])):
    if clu_You_diff_space[2][c] < 0.05:
        mask_diff[clu_You_diff_space[1][c][0]] = 1
'''
Plot
'''

data_to_plot = np.stack([np.stack(You_joint_space).mean(axis = 0)-.5, np.stack(You_parallel_space).mean(axis = 0)-.5, np.stack(You_joint_space).mean(axis = 0) - np.stack(You_parallel_space).mean(axis = 0)]).T

evoked_plot = mne.EvokedArray(data_to_plot, epochs.info)
evoked_plot.info.times = ['Joint', 'Parallel', 'Difference']

# fig, axs = plt.subplots(nrows = 1, ncols = 3)
# evoked_plot.plot_topomap(times = evoked_plot.times, scalings = 100, show_names = False, mask = np.stack([mask_joint, mask_parallel,      mask_diff]).squeeze().T,  mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
#         linewidth=0, markersize=8), axes = axs, time_format = '', colorbar = False)

# fig.suptitle('You')
# axs[0].set_title('Joint')
# axs[1].set_title('Parallel')
# axs[2].set_title('Difference')


# alternative

joint_plot = (np.stack(You_joint_space).mean(axis = 0)).reshape(64, 1)
joint_plot = mne.EvokedArray(joint_plot, epochs.info)
joint_plot.info.times = ['Joint']
cmp = LinearSegmentedColormap.from_list('joint_cmap', ['white','#CF3939'])
norm = mpl.colors.Normalize(vmin=50)
joint_plot.plot_topomap(times = joint_plot.times, scalings = 100,show_names = False, mask = mask_joint, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Joint', colorbar = True, cnorm = norm, vlim = [50,None])
plt.savefig("D:\JointRSA\Paper\Figures/joint_you.png", bbox_inches="tight", dpi=300)


parallel_plot = (np.stack(You_parallel_space).mean(axis = 0)).reshape(64, 1)
parallel_plot = mne.EvokedArray(parallel_plot, epochs.info)
parallel_plot.info.times = ['Parallel']
cmp = LinearSegmentedColormap.from_list('parallel_cmap', ['white','#198695'])
norm = mpl.colors.Normalize(vmin=50)
parallel_plot.plot_topomap(times = parallel_plot.times, scalings = 100,show_names = False, mask = mask_parallel, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Parallel', colorbar = True, cnorm = norm, vlim = [50,None])
plt.savefig("D:\JointRSA\Paper\Figures/parallel_you.png", bbox_inches="tight", dpi=300)


diff_plot = (np.stack(You_joint_space).mean(axis = 0) - np.stack(You_parallel_space).mean(axis = 0)).reshape(64, 1)
diff_plot = mne.EvokedArray(diff_plot, epochs.info)
diff_plot.info.times = ['Difference']
cmp = LinearSegmentedColormap.from_list('diff_cmap', ['#198695', 'white', '#CF3939'])
norm = mpl.colors.Normalize()
diff_plot.plot_topomap(times = diff_plot.times, scalings = 100,show_names = False, mask = mask_diff, cmap = cmp, mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='k', 
        linewidth=0, markersize=6), time_format = 'Difference', colorbar = True)




