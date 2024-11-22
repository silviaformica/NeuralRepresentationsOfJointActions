# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:02:56 2024

@author: Silvia
"""

# -*- coding: utf-8 -*-
"""

JOINT DRAWING EXPERIMENT

Spatiotemporal decoding TEST
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
import pingouin as pg

#%%

os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials

#%%

'''
Testing the decoding Congruent and Incongruenct combinations
'''

# loading decoding accuracies for each participant
Me_joint_cong = []
Me_joint_incong = []
Me_parallel_cong = []
Me_parallel_incong = []
Me_joint_cong_perm = []
Me_joint_incong_perm = []
Me_parallel_cong_perm = []
Me_parallel_incong_perm = []

for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatiotemporal_decoding_norm/'
    Me_joint_cong.append(np.load(data_path + '/Me_joint_cong.npy').mean(axis = 1).mean(axis = 0))
    Me_joint_incong.append(np.load(data_path + '/Me_joint_incong.npy').mean(axis = 1).mean(axis = 0))
    Me_parallel_cong.append(np.load(data_path + '/Me_parallel_cong.npy').mean(axis = 1).mean(axis = 0))
    Me_parallel_incong.append(np.load(data_path + '/Me_parallel_incong.npy').mean(axis = 1).mean(axis = 0))
    Me_joint_cong_perm.append(np.load(data_path + '/Me_joint_cong_perm.npy').mean(axis = 1).mean(axis = 0))
    Me_joint_incong_perm.append(np.load(data_path + '/Me_joint_incong_perm.npy').mean(axis = 1).mean(axis = 0))
    Me_parallel_cong_perm.append(np.load(data_path + '/Me_parallel_cong_perm.npy').mean(axis = 1).mean(axis = 0))
    Me_parallel_incong_perm.append(np.load(data_path + '/Me_parallel_incong_perm.npy').mean(axis = 1).mean(axis = 0))
    
    
Results = pd.DataFrame(columns = ['Accuracy', 'Task', 'Congruency', 'Subject'])
Results['Accuracy'] = Me_joint_cong+Me_joint_incong+Me_parallel_cong+Me_parallel_incong
Results['Subject'] = np.tile(subjlist, 4)
Results['Task'] = np.repeat(['Joint', 'Parallel'], 2*len(subjlist))
Results['Congruency'] = np.tile(np.repeat(['Congruent', 'Incongruent'], len(subjlist)), 2)

Results_perm1 = pd.DataFrame(columns = ['Accuracy', 'Task', 'Congruency', 'Subject'])
Results_perm1['Accuracy'] = Me_joint_cong_perm+Me_joint_incong_perm+Me_parallel_cong_perm+Me_parallel_incong_perm
Results_perm1['Subject'] = np.tile(subjlist, 4)
Results_perm1['Task'] = np.repeat(['Joint', 'Parallel'], 2*len(subjlist))
Results_perm1['Congruency'] = np.tile(np.repeat(['Congruent', 'Incongruent'], len(subjlist)), 2)


# Testing for normality
pg.normality(np.asarray(Me_joint_cong))
pg.normality(np.asarray(Me_parallel_cong))
pg.normality(np.asarray(Me_joint_incong))
pg.normality(np.asarray(Me_parallel_incong))
# all normally distributed



# checking what is the average decoding accuracies for the permuted distributions

print(np.asarray(Me_joint_cong_perm).mean())
print(np.asarray(Me_joint_incong_perm).mean())
print(np.asarray(Me_parallel_cong_perm).mean())
print(np.asarray(Me_parallel_incong_perm).mean())

print(np.asarray(Me_joint_cong_perm).std())
print(np.asarray(Me_joint_incong_perm).std())
print(np.asarray(Me_parallel_cong_perm).std())
print(np.asarray(Me_parallel_incong_perm).std())



perm_j_c_p = pg.ttest(np.asarray(Me_joint_cong_perm), 0.5, alternative = 'greater')
perm_j_i_p = pg.ttest(np.asarray(Me_joint_incong_perm), 0.5, alternative = 'greater')
perm_p_c_p = pg.ttest(np.asarray(Me_parallel_cong_perm), 0.5, alternative = 'greater')
perm_p_i_p = pg.ttest(np.asarray(Me_parallel_incong_perm), 0.5, alternative = 'greater')



## One-tailed paired t-test
# Testing for decoding accuracy significantly above chance
stat_me_joint_cong = pg.ttest(np.asarray(Me_joint_cong), np.asarray(Me_joint_cong_perm), alternative = 'greater', paired = True)
stat_me_parallel_cong = pg.ttest(np.asarray(Me_parallel_cong), np.asarray(Me_parallel_cong_perm), alternative = 'greater', paired = True)
stat_me_joint_incong = pg.ttest(np.asarray(Me_joint_incong), np.asarray(Me_joint_incong_perm), alternative = 'greater', paired = True)
stat_me_parallel_incong = pg.ttest(np.asarray(Me_parallel_incong), np.asarray(Me_parallel_incong_perm), alternative = 'greater', paired = True)


## Two-tailed paired samples t-tests
# to test for differences across tasks
stat_incong = pg.ttest(np.asarray(Me_joint_incong), np.asarray(Me_parallel_incong), paired = True)
stat_cong = pg.ttest(np.asarray(Me_joint_cong), np.asarray(Me_parallel_cong), paired = True)


## rmANOVA Task * Congruency and post-hoc tests
stat_anova = pg.rm_anova(Results, dv = 'Accuracy', within = ['Congruency', 'Task'], subject = 'Subject')
stat_anova_ph = pg.pairwise_tests(Results, dv = 'Accuracy', within = ['Congruency', 'Task'], subject = 'Subject')
stat_anova_ph1 = pg.pairwise_tests(Results, dv = 'Accuracy', within = ['Task', 'Congruency'], subject = 'Subject')


#%%
# Saving results to group results folder
Results.to_csv('D:/JointRSA/JointDrawing/group/SpatiotemporalDecodingAccuracies.csv')


#%%

'''
PLOTS
Manuscript Figure 5
'''
# inspiration: https://medium.com/mlearning-ai/getting-started-with-raincloud-plots-in-python-2ea5c2d01c11


f, [ax, ax1] = plt.subplots(1, 2, sharey = True, figsize = [8, 5])
sns.despine()
f.tight_layout()
# plt.subplot_tool()
plt.subplots_adjust(top=0.9, left = 0.08, wspace = 1.2)

colors = ['#DD7373', '#0892A5']

## SUBPLOT 1: Congruent
data = Results[Results.Congruency == 'Congruent']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]


# Boxplot data
bp = ax1.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax1.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])


plt.subplot(122)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])

ax1.axhline(50, color = 'gray', linestyle = '--',)
ax1.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))
ax1.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
# plt.title('Congruent combinations\n CC vs DD', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10)

ax1.set_ylim([37, 70])

## SUBPLOT 2: Inongruent
data = Results[Results.Congruency == 'Incongruent']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]


# Boxplot data
bp = ax.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])

plt.subplot(121)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])
ax.axhline(50, color = 'gray', linestyle = '--',)
ax.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))

ax.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
# plt.title('Incongruent combinations\n CD vs DC', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10)

plt.show()



#%%

'''
Testing the decoding Participant's and Partner's movements
'''

Me_joint = []
Me_parallel = []
You_joint = []
You_parallel = []

Me_joint_perm = []
Me_parallel_perm = []
You_joint_perm = []
You_parallel_perm = []

for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/spatiotemporal_decoding_norm/'
    Me_joint.append(np.load(data_path + '/Me_joint.npy').mean(axis = 1).mean(axis = 0))
    Me_parallel.append(np.load(data_path + '/Me_parallel.npy').mean(axis = 1).mean(axis = 0))
    You_joint.append(np.load(data_path + '/You_joint.npy').mean(axis = 1).mean(axis = 0))
    You_parallel.append(np.load(data_path + '/You_parallel.npy').mean(axis = 1).mean(axis = 0))
    Me_joint_perm.append(np.load(data_path + '/Me_joint_perm.npy').mean(axis = 1).mean(axis = 0))
    Me_parallel_perm.append(np.load(data_path + '/Me_parallel_perm.npy').mean(axis = 1).mean(axis = 0))
    You_joint_perm.append(np.load(data_path + '/You_joint_perm.npy').mean(axis = 1).mean(axis = 0))
    You_parallel_perm.append(np.load(data_path + '/You_parallel_perm.npy').mean(axis = 1).mean(axis = 0))


Results = pd.DataFrame(columns = ['Accuracy', 'Task', 'Variable', 'Subject'])
Results['Accuracy'] = Me_joint+Me_parallel+You_joint+You_parallel
Results['Subject'] = np.tile(subjlist, 4)
Results['Variable'] = np.repeat(['Me', 'You'], 2*len(subjlist))
Results['Task'] = np.tile(np.repeat(['Joint', 'Parallel'], len(subjlist)), 2)


Results_perm2 = pd.DataFrame(columns = ['Accuracy', 'Task', 'Variable', 'Subject'])
Results_perm2['Accuracy'] = Me_joint_perm+Me_parallel_perm+You_joint_perm+You_parallel_perm
Results_perm2['Subject'] = np.tile(subjlist, 4)
Results_perm2['Variable'] = np.repeat(['Me', 'You'], 2*len(subjlist))
Results_perm2['Task'] = np.tile(np.repeat(['Joint', 'Parallel'], len(subjlist)), 2)


# Testing for normality
pg.normality(np.asarray(Me_joint))
pg.normality(np.asarray(Me_parallel))
pg.normality(np.asarray(You_joint))
pg.normality(np.asarray(You_parallel))
# all normally distributed


# checking what is the average decoding accuracies for the permuted distributions

print(np.asarray(Me_joint_perm).mean())
print(np.asarray(You_joint_perm).mean())
print(np.asarray(Me_parallel_perm).mean())
print(np.asarray(You_parallel_perm).mean())

print(np.asarray(Me_joint_perm).std())
print(np.asarray(You_joint_perm).std())
print(np.asarray(Me_parallel_perm).std())
print(np.asarray(You_parallel_perm).std())



perm_j_m_p = pg.ttest(np.asarray(Me_joint_perm), 0.5, alternative = 'greater')
perm_j_y_p = pg.ttest(np.asarray(You_joint_perm), 0.5, alternative = 'greater')
perm_p_m_p = pg.ttest(np.asarray(Me_parallel_perm), 0.5, alternative = 'greater')
perm_p_y_p = pg.ttest(np.asarray(You_parallel_perm), 0.5, alternative = 'greater')



## One-tailed one-sample t-test
# Testing for decoding accuracy significantly above chance
stat_me_joint = pg.ttest(np.asarray(Me_joint), np.asarray(Me_joint_perm), alternative = 'greater', paired = True)
stat_me_parallel = pg.ttest(np.asarray(Me_parallel), np.asarray(Me_parallel_perm), alternative = 'greater', paired = True)
stat_you_joint = pg.ttest(np.asarray(You_joint), np.asarray(You_joint_perm), alternative = 'greater', paired = True)
stat_you_parallel = pg.ttest(np.asarray(You_parallel), np.asarray(You_parallel_perm), alternative = 'greater', paired = True)

## Two-tailed paired samples t-tests
# to test for differences across tasks
stat_me = pg.ttest(np.asarray(Me_joint), np.asarray(Me_parallel), paired = True)
stat_you = pg.ttest(np.asarray(You_joint), np.asarray(You_parallel), paired = True)

#%%

'''
PLOTS
Manuscript Figure 6
'''

f, [ax, ax1] = plt.subplots(1, 2, sharey = True, figsize= [8, 5])
sns.despine()
f.tight_layout()
# plt.subplot_tool()
plt.subplots_adjust(top=0.9, left = 0.08, wspace = 1.2)


## SUBPLOT 1: Participant's movement
data = Results[Results.Variable == 'Me']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]
colors = ['#DD7373', '#0892A5']

# Boxplot data
bp = ax.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])


plt.subplot(121)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])

ax.axhline(50, color = 'gray', linestyle = '--',)
ax.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))
ax.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10)


## SUBPLOT 2: Partner's movement
data = Results[Results.Variable == 'You']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]

# Boxplot data
bp = ax1.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax1.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])

plt.subplot(122)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])
ax1.axhline(50, color = 'gray', linestyle = '--',)
ax1.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))

ax1.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10)

plt.show()





#%%


'''
PLOTS
Manuscript Figure Supplementary Materials
'''


f, [ax, ax1, ax2, ax3] = plt.subplots(1, 4, sharey = True, figsize= [9, 5])
sns.despine()
# f.tight_layout()
# plt.subplot_tool()
plt.subplots_adjust(top=0.9, left = 0.08, wspace = 0.5)
colors = ['#DD7373', '#0892A5']



## SUBPLOT 1: Congruent
data = Results_perm1[Results_perm1.Congruency == 'Congruent']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]


# Boxplot data
bp = ax1.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax1.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])


plt.subplot(142)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])

ax1.axhline(50, color = 'gray', linestyle = '--',)
# ax1.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))
ax1.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('CC vs DD', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), loc = 'left')

ax1.set_ylim([45, 55])

## SUBPLOT 2: Incongruent
data = Results_perm1[Results_perm1.Congruency == 'Incongruent']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]


# Boxplot data
bp = ax.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])

plt.subplot(141)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])
ax.axhline(50, color = 'gray', linestyle = '--',)
ax.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))

ax.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('CD vs DC', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), loc = 'left')




## SUBPLOT 1: Participant's movement
data = Results_perm2[Results_perm2.Variable == 'Me']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]
colors = ['#DD7373', '#0892A5']

# Boxplot data
bp = ax2.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax2.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])


plt.subplot(143)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])

ax2.axhline(50, color = 'gray', linestyle = '--',)
# ax2.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))
ax2.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('Own: C vs D', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), loc = 'left')


## SUBPLOT 2: Partner's movement
data = Results_perm2[Results_perm2.Variable == 'You']
data_x = [data[data.Task =='Joint']['Accuracy']*100,data[data.Task =='Parallel']['Accuracy']*100]

# Boxplot data
bp = ax3.boxplot(data_x, patch_artist = True, vert = True, medianprops = dict(c = 'k'), sym = '')

# Change to the desired color and add transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(1)

# Violinplot data
vp = ax3.violinplot(data_x, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=True)

for idx, b in enumerate(vp['bodies']):
    # Get the center of the plot
    m = np.mean(b.get_paths()[0].vertices[:, 0])
    # Modify it so we only see the upper half of the violin plot
    b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], idx+1, idx+2)
    # Change to the desired color
    b.set_color(colors[idx])

plt.subplot(144)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])
ax3.axhline(50, color = 'gray', linestyle = '--',)
# ax3.set_ylabel('Decoding Accuracy', fontdict=dict(fontsize= 14, font = 'Arial'))

ax3.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('Partner: C vs D', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), loc = 'left')

plt.show()

f.savefig('D:/JointRSA/Paper/SCAN/For publication/FigureS1.tif', dpi=600, format="tif", pil_kwargs={"compression": "tiff_lzw"})

