
# -*- coding: utf-8 -*-
"""

JOINT DRAWING EXPERIMENT

Analyses on the behavioral data:
    - implementation of exclusion criteria
        - catch accuracy below 60% in at least one task
        - delta time exceeding 3 sd from mean for joint task
        - movement dur exceeding 3 sd from mean for parallel task
    - analyses on catch trials
    - analyses on general indicators of behavioral performance
    
@author: Silvia Formica
silvia.formica@hu-berlin.de
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

#%%

# We aimed at collecting 36 good participants
# The original sample analyzed in this first section is 39 participants
# 3 are discarded because of bad EEG data, and failing the catch trial criterium

os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38']

subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials


AllMain = pd.DataFrame()

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/beh/'    
    
    behav_folder = [x for x in os.listdir(data_path) if str(x).startswith('Subj') ]

    for idx, fold in enumerate(behav_folder):
        Maintask = pd.DataFrame()
        files = os.listdir(data_path + fold)
        for idx_f, file in enumerate(files):
            if file.endswith('.csv') and file.startswith('Task'):
                thisf = pd.read_csv(data_path + fold + '/' + file)
                Maintask = pd.concat([Maintask, thisf])
                
    AllMain = pd.concat([AllMain, Maintask])
        
    AllMain = AllMain.reset_index(drop =True)


#%%
'''
Checking performance of the participants to figure out if someone needs to be discarded
Three criteria are checked
    - 3sd form the mean delta time in JOINT task
    - 3sd form the mean movement time in PARALLEL task
    - 3sd from the mean accurate trajectory in both tasks
'''

# Adding relevant factors to the dataframe

for idx, trace in enumerate(AllMain.track): 
    if AllMain.loc[idx, 'cue_participant'] == AllMain.loc[idx, 'cue_coagent']:
        AllMain.loc[idx, 'cong'] = 'congruent'
    else:
        AllMain.loc[idx, 'cong'] = 'incongruent'

    AllMain.loc[idx, 'delta_time'] = AllMain.loc[idx, 'pp_finish_time'] - AllMain.loc[idx, 'coagent_finish_time']
    AllMain.loc[idx, 'coagent_mov_dur'] = AllMain.loc[idx, 'coagent_dur'] * (1/60)
    AllMain.loc[idx, 'delta_abs'] = abs(AllMain.loc[idx, 'pp_finish_time'] - AllMain.loc[idx, 'coagent_finish_time'])

AllMain = AllMain[AllMain.catch == 0]
AllMain = AllMain[AllMain.starting_time_too_slow != 1]
AllMain = AllMain[~ np.isnan(AllMain.pp_finish_time)]
AllMain = AllMain[AllMain.pp_starting_time < 0.8]

AllMain = AllMain.reset_index(drop = True)

####################################
## Checking delta time in JOINT task
####################################

data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True)

mean_dt = data.groupby('task').mean(numeric_only = True)['delta_time']
sd_dt =  data.groupby('task').std()['delta_time']

low_thresh =  mean_dt.loc['Joint'] - 3* sd_dt.loc['Joint']
high_thresh =  mean_dt.loc['Joint'] + 3* sd_dt.loc['Joint']

# subject that would need discard on delta time
discard_delta_joint = data.loc[(data.task == 'Joint') & ((data['delta_time'] < low_thresh) | (data['delta_time'] < low_thresh)), 'Subject']

print(discard_delta_joint)


##########################################
## Checking movement time in PARALLEL task
##########################################

mean_dt = data.groupby('task').mean(numeric_only = True)['pp_mov_dur']
sd_dt =  data.groupby('task').std(numeric_only = True)['pp_mov_dur']

low_thresh =  mean_dt.loc['Parallel'] - 3* sd_dt.loc['Parallel']
high_thresh =  mean_dt.loc['Parallel'] + 3* sd_dt.loc['Parallel']

# subject that would need discard on mov dur time
discard_dur_parallel = data.loc[(data.task == 'Parallel') & ((data['pp_mov_dur'] < low_thresh) | (data['pp_mov_dur'] < low_thresh)), 'Subject']

print(discard_dur_parallel)


##########################################
## Checking trajectory
##########################################

mean_dt = data.groupby('task').mean(numeric_only = True)['accurate_traj']
sd_dt =  data.groupby('task').std(numeric_only = True)['accurate_traj']

low_thresh =  mean_dt.loc['Joint'] - 3* sd_dt.loc['Joint']
high_thresh =  mean_dt.loc['Joint'] + 3* sd_dt.loc['Joint']

# subject that would need discard on trajectory
discard_traj_joint = data.loc[(data.task == 'Joint') & ((data['accurate_traj'] < low_thresh) | (data['accurate_traj'] < low_thresh)), 'Subject']

print(discard_traj_joint)

low_thresh =  mean_dt.loc['Parallel'] - 3* sd_dt.loc['Parallel']
high_thresh =  mean_dt.loc['Parallel'] + 3* sd_dt.loc['Parallel']

# subject that would need discard on delta time
discard_traj_parallel = data.loc[(data.task == 'Parallel') & ((data['accurate_traj'] < low_thresh) | (data['accurate_traj'] < low_thresh)), 'Subject']

print(discard_traj_parallel)


## Subject 8 should be discarded based on their delta time in the Joint task


#%%

'''
Subject 8 is replace by Subject 39 and the exclusion criteria checked again
'''

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')    # discarded based on poor delta time in Joint task
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials


AllMain = pd.DataFrame()

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/beh/'    
    
    
    behav_folder = [x for x in os.listdir(data_path) if str(x).startswith('Subj') ]


    for idx, fold in enumerate(behav_folder):
        Maintask = pd.DataFrame()
        files = os.listdir(data_path + fold)
        for idx_f, file in enumerate(files):
            if file.endswith('.csv') and file.startswith('Task'):
                thisf = pd.read_csv(data_path + fold + '/' + file)
                Maintask = pd.concat([Maintask, thisf])
                
    AllMain = pd.concat([AllMain, Maintask])
        
    AllMain = AllMain.reset_index(drop =True)

#%%
'''
Checking performance of the participants to figure out if someone needs to be discarded
Three criteria are checked
    - 3sd form the mean delta time in JOINT task
    - 3sd form the mean movement time in PARALLEL task
    - 3sd from the mean accurate trajectory in both tasks
'''

# Adding relevant factors to the dataframe

for idx, trace in enumerate(AllMain.track): 
    if AllMain.loc[idx, 'cue_participant'] == AllMain.loc[idx, 'cue_coagent']:
        AllMain.loc[idx, 'cong'] = 'congruent'
    else:
        AllMain.loc[idx, 'cong'] = 'incongruent'

    AllMain.loc[idx, 'delta_time'] = AllMain.loc[idx, 'pp_finish_time'] - AllMain.loc[idx, 'coagent_finish_time']
    AllMain.loc[idx, 'coagent_mov_dur'] = AllMain.loc[idx, 'coagent_dur'] * (1/60)
    AllMain.loc[idx, 'delta_abs'] = abs(AllMain.loc[idx, 'pp_finish_time'] - AllMain.loc[idx, 'coagent_finish_time'])

AllMain = AllMain[AllMain.catch == 0]
AllMain = AllMain[AllMain.starting_time_too_slow != 1]
AllMain = AllMain[~ np.isnan(AllMain.pp_finish_time)]
AllMain = AllMain[AllMain.pp_starting_time < 0.8]

AllMain = AllMain.reset_index(drop = True)

####################################
## Checking delta time in JOINT task
####################################

data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True)

mean_dt = data.groupby('task').mean(numeric_only = True)['delta_time']
sd_dt =  data.groupby('task').std(numeric_only = True)['delta_time']

low_thresh =  mean_dt.loc['Joint'] - 3* sd_dt.loc['Joint']
high_thresh =  mean_dt.loc['Joint'] + 3* sd_dt.loc['Joint']

# subject that would need discard on delta time
discard_delta_joint = data.loc[(data.task == 'Joint') & ((data['delta_time'] < low_thresh) | (data['delta_time'] < low_thresh)), 'Subject']

print(discard_delta_joint)


##########################################
## Checking movement time in PARALLEL task
##########################################

mean_dt = data.groupby('task').mean(numeric_only = True)['pp_mov_dur']
sd_dt =  data.groupby('task').std(numeric_only = True)['pp_mov_dur']

low_thresh =  mean_dt.loc['Parallel'] - 3* sd_dt.loc['Parallel']
high_thresh =  mean_dt.loc['Parallel'] + 3* sd_dt.loc['Parallel']

# subject that would need discard on mov dur time
discard_dur_parallel = data.loc[(data.task == 'Parallel') & ((data['pp_mov_dur'] < low_thresh) | (data['pp_mov_dur'] < low_thresh)), 'Subject']

print(discard_dur_parallel)


##########################################
## Checking trajectory
##########################################

mean_dt = data.groupby('task').mean(numeric_only = True)['accurate_traj']
sd_dt =  data.groupby('task').std(numeric_only = True)['accurate_traj']

low_thresh =  mean_dt.loc['Joint'] - 3* sd_dt.loc['Joint']
high_thresh =  mean_dt.loc['Joint'] + 3* sd_dt.loc['Joint']

# subject that would need discard on trajectory
discard_traj_joint = data.loc[(data.task == 'Joint') & ((data['accurate_traj'] < low_thresh) | (data['accurate_traj'] < low_thresh)), 'Subject']

print(discard_traj_joint)

low_thresh =  mean_dt.loc['Parallel'] - 3* sd_dt.loc['Parallel']
high_thresh =  mean_dt.loc['Parallel'] + 3* sd_dt.loc['Parallel']

# subject that would need discard on delta time
discard_traj_parallel = data.loc[(data.task == 'Parallel') & ((data['accurate_traj'] < low_thresh) | (data['accurate_traj'] < low_thresh)), 'Subject']

print(discard_traj_parallel)


## No participant need discarding!


#%%

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ANALYSES WITH THE FINAL SAMPLE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
  
subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')    # discarded based on too low delta time in Joint task
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials


AllMain = pd.DataFrame()
AllPractice = pd.DataFrame()
AllLearning = pd.DataFrame()

for subj in subjlist:
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/beh/'    
    
    
    behav_folder = [x for x in os.listdir(data_path) if str(x).startswith('Subj') ]

    for idx, fold in enumerate(behav_folder):
        Practice = pd.DataFrame()
        Maintask = pd.DataFrame()
        Learning = pd.DataFrame()
        files = os.listdir(data_path + fold)
        for idx_f, file in enumerate(files):
            if file.endswith('.csv') and (file.startswith('PracticeJoint') or file.startswith('PracticeParallel')):
                thisf = pd.read_csv(data_path + fold + '/' + file)
                Practice = pd.concat([Practice, thisf])
            elif file.endswith('.csv') and file.startswith('Task'):
                thisf = pd.read_csv(data_path + fold + '/' + file)
                Maintask = pd.concat([Maintask, thisf])
            elif file.endswith('.csv') and file.startswith('Practice'):
                thisf = pd.read_csv(data_path + fold + '/' + file)
                Learning = pd.concat([Learning, thisf])
                
                
    AllMain = pd.concat([AllMain, Maintask])
    AllPractice = pd.concat([AllPractice, Practice])
    AllLearning = pd.concat([AllLearning, Learning])
        
    AllMain = AllMain.reset_index(drop =True)
    AllPractice = AllPractice.reset_index(drop =True)
    AllLearning = AllLearning.reset_index(drop = True)


#############################################################################
## descriptive
#############################################################################

print('------- Sample Descriptives -------')

print(
    "Mean Age: %s (±%s), Range = [%s - %s]"
    % (
        np.around(np.mean(AllMain.groupby('Subject').mean(numeric_only = True).Age), decimals=2),
        np.around(np.std(AllMain.groupby('Subject').mean(numeric_only = True).Age), decimals=2),
        np.around(np.min(AllMain.groupby('Subject').mean(numeric_only = True).Age), decimals=2),
        np.around(np.max(AllMain.groupby('Subject').mean(numeric_only = True).Age), decimals=2)
    )
)

print(
    "Gender: %s" % np.unique(AllMain.groupby('Subject').first().Gender),
    np.unique(np.unique(AllMain.groupby('Subject').first().Gender, return_counts=True)[1],
))

print(
    "Handedness: %s" % np.unique(AllMain.groupby('Subject').first().Handedness),
    np.unique(AllMain.groupby('Subject').first().Handedness, return_counts=True)[1],
)

print(
    "Task Order: %s" % np.unique(AllMain.groupby('Subject').first().TaskOrder),
    np.unique(AllMain.groupby('Subject').first().TaskOrder, return_counts=True)[1],
)

print(
    "Cue Combination: %s" % np.unique(AllMain.groupby('Subject').first().CueCombination),
    np.unique(AllMain.groupby('Subject').first().CueCombination, return_counts=True)[1],
)

print("%s participants completed the experiment" % len(AllMain.groupby('Subject').first()))



#%%

#############################################################################
## Practice performance
#############################################################################

print('------- Practice Performance indicators -------')

a = AllPractice.groupby(['Subject', 'task'], as_index = True).max().run.reset_index()

a.run= a.run +1

print(
    "Average # of Practice blocks - JOINT: %s, ±%s" % (str((a[a.task == 'Joint'].run).mean()),
    str(np.around((a[a.task == 'Joint'].run).std(),decimals = 2))))


print(
    "Average # of Practice blocks - PARALLEL: %s, ±%s" % (str((a[a.task == 'Parallel'].run).mean()),
    str(np.around((a[a.task == 'Parallel'].run).std(),decimals = 2))))

# sns.catplot(x = 'task', y = 'run', data = a, kind ='bar', palette = ['#AC2016', '#1670AC'])

pg.normality(a['run'])
# violated normality!

pg.wilcoxon(x=a[a.task == 'Joint']['run'], y = a[a.task == 'Parallel']['run'])



# ttest_prac = pg.pairwise_tests(
#     dv="run",
#     within="task",
#     subject="Subject",
#     data=a,
#     effsize="cohen",
# )
# print("\nNumber of practice runs")
# pg.print_table(ttest_prac, floatfmt=".3f")

## No differences in the number of practice runs per task

#%%

'''
Catch trials analyses
'''

Catches = AllMain[AllMain.catch == 1]
Catches = Catches[Catches.starting_time_too_slow != 1]
Catches = Catches[Catches.pp_starting_time < 0.8]

# sns.catplot(x = 'task', y = 'catch_detected', palette = ['#AC2016', '#1670AC'], data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(),kind ='box')

ttest_catch = pg.pairwise_tests(
    dv="catch_detected",
    within="task",
    subject="Subject",
    data=Catches.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True),
    effsize="cohen",
)
print("\nCatch detection")
pg.print_table(ttest_catch, floatfmt=".3f")


grouped = Catches.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True)
grouped.loc[grouped.task == 'Joint', 'catch_detected'].mean()
grouped.loc[grouped.task == 'Joint', 'catch_detected'].std()
grouped.loc[grouped.task == 'Parallel', 'catch_detected'].mean()
grouped.loc[grouped.task == 'Parallel', 'catch_detected'].std()

# Checking normality
data = Catches.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True)
pg.normality(data['catch_detected'])

# Normality is violated, thus Wilcoxon test is used
pg.wilcoxon(x=data[data.task == 'Joint']['catch_detected'], y = data[data.task == 'Parallel']['catch_detected'])

# Higher accuracy in catch trials detection in Joint task (p = 0.035)

#%%
'''
Regular trials analyses
'''

# Adding useful columns to the dataframe

for idx, trace in enumerate(AllMain.track): 
    if AllMain.loc[idx, 'cue_participant'] == AllMain.loc[idx, 'cue_coagent']:
        AllMain.loc[idx, 'cong'] = 'congruent'
    else:
        AllMain.loc[idx, 'cong'] = 'incongruent'

    AllMain.loc[idx, 'delta_time'] = AllMain.loc[idx, 'pp_finish_time'] - AllMain.loc[idx, 'coagent_finish_time']
    AllMain.loc[idx, 'coagent_mov_dur'] = AllMain.loc[idx, 'coagent_dur'] * (1/60)
    AllMain.loc[idx, 'delta_abs'] = abs(AllMain.loc[idx, 'pp_finish_time'] - AllMain.loc[idx, 'coagent_finish_time'])


# retaining only relevant trials for analysis
AllMain = AllMain[AllMain.catch == 0]
AllMain = AllMain[AllMain.starting_time_too_slow != 1]
AllMain = AllMain[~ np.isnan(AllMain.pp_finish_time)]
AllMain = AllMain[AllMain.pp_starting_time < 0.8]

AllMain = AllMain.reset_index(drop = True)


# Plots

# sns.catplot(x = 'task', y = 'delta_time', palette = ['#AC2016', '#1670AC'],data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True),kind ='box')
# plt.axhline(0,c ='gray', lw = 2, linestyle = '--')

# sns.catplot(x = 'task', y = 'delta_abs', palette = ['#AC2016', '#1670AC'],data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True),kind ='box')
# plt.axhline(0,c ='gray', lw = 2, linestyle = '--')

# sns.catplot(x = 'task', y = 'accurate_traj',palette = ['#AC2016', '#1670AC'],data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True),kind ='box')

# sns.catplot(x = 'task', y = 'pp_mov_dur',palette = ['#AC2016', '#1670AC'],data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True),kind ='box')



# Stats
a = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True)



# checking for violated normality- delta time
pg.normality(dv="delta_time",
group="task",
data=a.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True))


# checking for violated normality - accurate trajectory
pg.normality(dv="accurate_traj",
group="task",
data=a.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True))


# checking for violated normality - starting time
pg.normality(dv="pp_starting_time",
group="task",
data=a.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True))
## pp_starting_time is not normally distributed


# analyses on delta time
ttest_delta_time = pg.pairwise_tests(
    dv="delta_time",
    within="task",
    subject="Subject",
    data=a,
    effsize="cohen",
)
print("\nDelta Time")
pg.print_table(ttest_delta_time, floatfmt=".3f")


# Testing delta time against 0, separately by task
j_0 = pg.ttest(a[a.task == 'Joint']['delta_time'], 0)
p_0 = pg.ttest(a[a.task == 'Parallel']['delta_time'], 0)

# descriptives of delta time
a[a.task == 'Joint']['delta_time'].mean()
a[a.task == 'Joint']['delta_time'].std()

a[a.task == 'Parallel']['delta_time'].mean()
a[a.task == 'Parallel']['delta_time'].std()

# Testing movement duration against 2, separately by task
j_0 = pg.ttest(a[a.task == 'Joint']['pp_mov_dur'], 2)
p_0 = pg.ttest(a[a.task == 'Parallel']['pp_mov_dur'], 2)

# descriptives of movement duration
a[a.task == 'Parallel']['pp_mov_dur'].mean()
a[a.task == 'Parallel']['pp_mov_dur'].std()
a[a.task == 'Joint']['pp_mov_dur'].mean()
a[a.task == 'Joint']['pp_mov_dur'].std()



#%%


'''
PLOTS

Manuscript Figure 3

'''


f, [ax, ax1, ax2] = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]}, figsize= [8, 5] )
sns.despine()
f.tight_layout()
# plt.subplot_tool()
plt.subplots_adjust(top=0.9)

colors = ['#DD7373', '#0892A5']

## SUBPLOT 1: CATCH DETECTION

data=Catches.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True)

data_x = [data[data.task =='Joint']['catch_detected']*100,data[data.task =='Parallel']['catch_detected']*100]


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


plt.subplot(131)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])

ax.set_xticklabels(['Joint', 'Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('Catch accuracy', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10)



## SUBPLOT 2: DELTA TIME


data = AllMain.groupby(['Subject', 'task'], as_index= False).mean(numeric_only = True)

data_x = [data[data.task =='Joint']['delta_time']]

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


plt.subplot(132)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])
plt.axhline(0,c ='gray', lw = 1, linestyle = '--')

ax1.set_xticklabels(['Joint'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('Delta time', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10)

## SUBPLOT 3: PP MOVEMENT TIME


data_x = [data[data.task =='Parallel']['pp_mov_dur']]

# Create a list of colors for the boxplots based on the number of features you have
colors = [colors[1]]

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


plt.subplot(133)
# Scatterplot data
for idx, features in enumerate(data_x):
    # Add jitter effect so the features do not overlap on the y-axis
    y = np.full(len(features), idx + .8)
    idxs = np.arange(len(y))
    out = y.astype(float)
    out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
    y = out
    plt.scatter(y, features, s=5, c=colors[idx])

plt.axhline(2,c ='gray', lw = 1, linestyle = '--')
ax2.set_xticklabels(['Parallel'], fontdict=dict(fontsize= 14, font = 'Arial'))
plt.title('Drawing time', fontdict=dict(fontsize= 18, font = 'Arial', fontweight = 'bold'), pad = 10)



f.savefig('D:/JointRSA/Paper/SCAN/For publication/Figure3.tif', dpi=600, format="tif", pil_kwargs={"compression": "tiff_lzw"})

