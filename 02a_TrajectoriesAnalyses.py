
# -*- coding: utf-8 -*-
"""

JOINT DRAWING EXPERIMENT

Analyses of the drawn trajectories
    - loading all distances
    - combining data
    - correcting area non-normality with boxcox
    - preparing dataframes for R analyses
    - plotting
    
@author: Silvia Formica
silvia.formica@hu-berlin.de
"""


import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')    # discarded based on poor delta time in Joint task
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials


#%%
# Loading distances for all pp

Distances = pd.DataFrame()

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    
    this_s = pd.read_csv(data_path + 'dist.csv')
    
    
    # we add to this dataset additional info that we want to run control analyses on
    # therefore we retrieve metadata for each participant
    # an we attach trial-specific info
    
    idxs = np.unique(this_s.trial_sort)
    data_path_beh = 'D:/JointRSA/JointDrawing/sub-' + subj + '/beh/'    
    behav_folder = [x for x in os.listdir(data_path_beh) if str(x).startswith('Subj') ]

    for idx, fold in enumerate(behav_folder):
        Maintask = pd.DataFrame()
        files = os.listdir(data_path_beh + fold)
        for idx_f, file in enumerate(files):
            if file.endswith('.csv') and file.startswith('Task'):
                thisf = pd.read_csv(data_path_beh + fold + '/' + file)
                Maintask = pd.concat([Maintask, thisf])
                
    Maintask_sampled = Maintask[np.isin(Maintask.trial_sort, idxs)]
    
    this_s['pp_mov_dur'] = (np.repeat(Maintask_sampled.pp_mov_dur, 100)).reset_index()['pp_mov_dur']
    this_s['TaskOrder'] = (np.repeat(Maintask_sampled.TaskOrder, 100)).reset_index()['TaskOrder']
    
    Distances = pd.concat([Distances, this_s])
    




#%%

# We want to focus our analyses on AREA

data = Distances.groupby(['Subject','cong', 'task', 'trial_sort'], as_index = False).mean(numeric_only = True).reset_index()

pg.normality(data['area'])
# Area is clearly non normally distributed

# We correct non-normality by applying a boxcox correction
data['boxcox'] = scipy.stats.boxcox(data.area)[0]

scipy.stats.describe(data['boxcox'])


# Saving for analyses in R
data.to_csv('D:/JointRSA/JointDrawing/group/Distances.csv')
data.to_csv('D:/JointRSA/JointDrawing/group/VMI_Analyses/Distances.csv')

## SEE RESULTS OF MODEL FITTING IN R


# Creating an additional dataframe to conduct control analyses on the factor task order
data_TO = Distances.groupby(['Subject','cong', 'task', 'trial_sort', 'TaskOrder'], as_index = False).mean().reset_index()

data_TO['boxcox'] = scipy.stats.boxcox(data_TO.area)[0]

data_TO.to_csv('D:/JointRSA/JointDrawing/group/TaskOrderCheck.csv')
data_TO.to_csv('D:/JointRSA/JointDrawing/group/VMI_Analyses/TaskOrderCheck.csv')



#%%

'''
PLOTS

Manuscript Figure 4

'''

f, ax = plt.subplots(1, 1, figsize = [7,5])
sns.despine()
plt.subplots_adjust(left = 0.12)
# f.tight_layout()

sns.pointplot(x = 'cong',   y = 'boxcox',palette =['#DD7373', '#0892A5'], dodge = True, hue = 'task', data = data.groupby(['Subject', 'task', 'cong'], as_index = False).mean(),  errorbar = ('ci'))
plt.ylabel('Area (Boxcox-corrected)', fontdict=dict(fontsize= 14, font = 'Arial'))
plt.xlabel('Congruency', fontdict=dict(fontsize= 14, font = 'Arial'))
plt.legend(title = 'Task', loc = 'center right')
ax.set_xticklabels(['Congruent', 'Incongruent'], fontdict=dict(fontsize= 14, font = 'Arial'))

plt.show()




#%%

'''
Manuscript Supplementary Figure 1
'''

colors = ['#DD7373', '#C22F2F', '#0892A5', '#066976']

f, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize=(12,8))
# sns.despine()
f.tight_layout()
plt.subplots_adjust(top=0.941,
bottom=0.082,
left=0.07,
right=0.989,
hspace=0.35,
wspace=0.2)

plt.subplot(221)
sns.regplot(data[(data.task == 'Joint') & (data.cong == 'congruent')], x = 'pp_mov_dur', y = 'boxcox', ax = ax[0, 0], color = colors[0], scatter = False)
ax[0, 0].set_title('Joint - Congruent', fontdict=dict(fontsize= 16, font = 'Arial', weight= 'bold'), pad = 10)
ax[0, 0].set_xlabel('')
ax[0, 0].set_ylabel('Area (Boxcox-corrected)', fontdict=dict(fontsize= 14, font = 'Arial'))

plt.subplot(222)
sns.regplot(data[(data.task == 'Joint') & (data.cong == 'incongruent')], x = 'pp_mov_dur', y = 'boxcox', ax = ax[0, 1], color = colors[1], scatter = False)
ax[0, 1].set_title('Joint - Incongruent', fontdict=dict(fontsize= 16, font = 'Arial', weight= 'bold'), pad = 10)
ax[0, 1].set_xlabel('')
ax[0, 1].set_ylabel('')


plt.subplot(223)
sns.regplot(data[(data.task == 'Parallel') & (data.cong == 'congruent')], x = 'pp_mov_dur', y = 'boxcox', ax = ax[1, 0], color = colors[2], scatter = False)
ax[1, 0].set_title('Parallel - Congruent', fontdict=dict(fontsize= 16, font = 'Arial', weight= 'bold'), pad = 10)
ax[1, 0].set_ylabel('Area (Boxcox-corrected)', fontdict=dict(fontsize= 14, font = 'Arial'))
ax[1, 0].set_xlabel('Drawing Time', fontdict=dict(fontsize= 14, font = 'Arial'))

plt.subplot(224)
sns.regplot(data[(data.task == 'Parallel') & (data.cong == 'incongruent')], x = 'pp_mov_dur', y = 'boxcox', ax = ax[1, 1], color = colors[3], scatter = False)
ax[1, 1].set_title('Parallel - Incongruent', fontdict=dict(fontsize= 16, font = 'Arial', weight= 'bold'), pad = 10)
ax[1, 1].set_xlabel('Drawing Time', fontdict=dict(fontsize= 14, font = 'Arial'))
ax[1, 1].set_ylabel('')

plt.show()

f.savefig('D:/JointRSA/Paper/SCAN/For publication/Figure_S2.tif', dpi=600, format="tif", pil_kwargs={"compression": "tiff_lzw"})


#%%
################################
## TESTING THE DIRECTIONAL HP
###############################

data = data.groupby(['Subject','cong', 'task'], as_index = False).mean().reset_index()

## Computing CONGRUENC EFFECT

congEff_joint = data[(data.task == 'Joint') & (data.cong == 'incongruent')].reset_index(drop = True)['boxcox'] - data[(data.task == 'Joint') & (data.cong == 'congruent')].reset_index(drop = True)['boxcox']

congEff_parallel = data[(data.task == 'Parallel') & (data.cong == 'incongruent')].reset_index(drop = True)['boxcox'] - data[(data.task == 'Parallel') & (data.cong == 'congruent')].reset_index(drop = True)['boxcox']

# Checking normality
pg.normality(congEff_joint)
pg.normality(congEff_parallel)

# two-samples one-tailed t-test
stats_congEff = pg.ttest(congEff_joint, congEff_parallel, paired = True, alternative = 'less')





