
# -*- coding: utf-8 -*-
"""

JOINT DRAWING EXPERIMENT

Preprocessing of the drawn trajectories
    - plotting and visual inspection to discard gross errors
    - interpolation of each trajectory with Cubic spline
    - computation of shapes templates for each participant
    - computation of distances of each trial from participant-specific template
    
@author: Silvia Formica
silvia.formica@hu-berlin.de
"""



import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import pingouin as pg


os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')    # discarded based on poor delta time in Joint task
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials



#%%

################################################
## LOADING ALL DATA
################################################

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

# Adding relevant columns to the dataframe

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



#%%

## NOTE: this step is very time consuming
## interpolated trajectories can be loaded in the next section

'''
This block of code plots for each participant a plot for each of the runs
with all the interpolated trajectories + inversion points of the first derivative

Two clicks on one subplot discards the trial. Close the figure to advance to the next run

'''

print('============= COMPUTING INTERPOLATIONS =============')

# initializing empty data frame for all results
Interp = pd.DataFrame()

# points for interpolation
n_points = 100

plt.close('all')

# defining functions for trials visualization and interactive rejection
def onclick(event):
     if event.dblclick:
        event.inaxes.set_facecolor('#e6aeae')
        f.canvas.draw() 
        to_discard_on_shape.append(int(str(event.inaxes.title).split('\'')[1]))

def on_press(event_key):
    if event_key.key == 'enter':
        print('Finished for this run')
        f.canvas.flush_events()
        f.canvas.mpl_disconnect(cid)
        f.canvas.mpl_disconnect(cid1)
        plt.close('all')



## LOOPINGTHROUGH SUBJECTS

for idx_s, s in enumerate(subjlist):
    subj = AllMain.groupby('Subject').get_group(int(s))
    this_subj = pd.DataFrame()
    print('-------------- Subj ' + s + '--------------')

    to_discard_on_shape = []

    for task_name in np.unique(subj.task).tolist():
        task = subj[subj.task == task_name]
        for run in np.unique(task.run).tolist():
            
            print('Working on ' + str(task_name) + ', run ' + str(run))
            data_plot = task[task.run == run] 
            data_plot = data_plot.reset_index()
            
            f, axs = plt.subplots(8, 5, sharex = True, sharey = True)
             
            f.canvas.mpl_connect('button_press_event', onclick)
            cid = f.canvas.mpl_connect('button_press_event', onclick)
            
            f.canvas.mpl_connect('key_press_event', on_press)
            cid1 = f.canvas.mpl_connect('key_press_event', on_press)

    
            for idx, trace in enumerate(data_plot.track):
                        
                subj = subj.reset_index(drop = True)
                
                temp = pd.DataFrame()
                
                #print('trial ' + str(subj.loc[idx, 'trial_sort']))
                
                # READING IN THE TRAJECTORIES FROM STRINGS
                trace = trace[1:-1].replace('], [', '] - [').split(' - ')
                # note that we are switching x and y to make the trajectories mathematically manageable
                y = [eval(thisp)[0] for thisp in trace]
                x = [eval(thisp)[1] for thisp in trace]    

                # x needs to be strictly increasing to be interpolated
                # check if there are duplicates in x or y    
                # y_uni = np.unique(y, return_index = True)
                x_uni = np.unique(x, return_index = True)
                x_keep = [xi for idx,xi in enumerate(x) if idx in x_uni[1] ]
                y_keep = [yi for idx,yi in enumerate(y) if idx in x_uni[1] ]
                   
                x_keep = sorted(x_keep)
                y_keep = list(reversed(y_keep))

                # COMPUTING CUBIC INTERPOLATION
                f_interp = scipy.interpolate.CubicSpline(x = x_keep, y = y_keep)
                new_x = np.linspace(np.stack(x_keep).min(), np.stack(x_keep).max(), num = n_points) 
                desired_y = f_interp(new_x)        
                
                # finding the vertex of this interpolated trajectory
                t = [new_x[np.asarray(desired_y).argmax()]]

                # computing first derivative
                first_der = f_interp(new_x, nu=1)
                
                # find indexes of new_x where the first derivative changes sign (inversion points)
                inver_idx= np.where(np.sign(first_der[:-1]) != np.sign(first_der[1:]))[0] + 1
                # get the two inversion points around the maximum t

                ## sometimes the edges of new_x are not included
                inver_idx= list(inver_idx)
                
                # adding indices of the edges of the trajectory
                inver_idx.append(0)
                inver_idx.append(len(new_x)-1)

                idx_t = np.where(new_x == t[0])[0][0]
                
                # in case the indx of the maxuimumn is not in there
                if 1 not in abs(inver_idx - idx_t):
                    inver_idx.append(idx_t)
                    
                inver_idx= sorted(np.unique(inver_idx))
                
                
                
                # saving results for each trial
                temp['x'] = new_x
                temp['y'] = desired_y
                temp['Subject'] = data_plot.loc[idx, 'Subject']
                temp['task'] = data_plot.loc[idx, 'task']
                temp['shape']  = data_plot.loc[idx, 'cue_participant_letter']
                temp['cong'] = data_plot.loc[idx, 'cong']
                temp['trial_sort']  = data_plot.loc[idx, 'trial_sort']
                temp['traj_point'] = np.arange(n_points)
                
                
                # for trials in which the trajectory is acceptable
                # (i.e., the maximum is within the edges)
                # we compute a linear and a quadratic b-spline using the maximum as knot
                # We will use this later to define if the correct shape was drawn
                
                if (t[0] > new_x[0] and t[0]< new_x[-1]):
                    spl_lin = scipy.interpolate.LSQUnivariateSpline(new_x, desired_y, t, k =1)
                    linearfit = spl_lin(new_x)
                    spl_quad = scipy.interpolate.LSQUnivariateSpline(new_x, desired_y, t, k =2)
                    quadraticfit = spl_quad(new_x)
                    
                    temp['res_lin'] = spl_lin.get_residual()
                    temp['res_quad'] = spl_quad.get_residual()
                
                this_subj = pd.concat([this_subj, temp])
                
                # Plotting all trials in one run
                # scatter point are the original trajectories
                # lines are the interpolated splines
                # grey lines are the inversion points (ideally should be
                # the edges and the maximum)
                
                plt.subplot(8, 5, idx+1)
                plt.suptitle(task_name + str(run))
                plt.scatter(x_keep, y_keep, color = 'k')
                plt.plot(new_x, desired_y, color = 'r')
                plt.title(str(data_plot.loc[idx, 'trial_sort']))
                for inv in inver_idx:
                    plt.axvline(new_x[inv], color = 'gray')
                    
       
            try:
                while f.number in plt.get_fignums():
                    plt.pause(0.1)
            except:
                plt.close(f.number)
                raise

    # Adding to the results for this subj the info on the trials discarded based on visual inspection
    this_subj['discard_on_shape'] = np.zeros(len(this_subj))
    for i_d, discarded in enumerate(to_discard_on_shape):
        this_subj.loc[this_subj.trial_sort == discarded, 'discard_on_shape'] = 1
    
    this_subj.to_csv('D:/JointRSA/JointDrawing/sub-' + s + '/Results/interp_traj.csv')

    Interp = pd.concat([Interp, this_subj])





#%%

'''
LOADING ALL INTERPOLATIONS
'''

Interp = pd.DataFrame()

for subj in subjlist:
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/' 
    this_s = pd.read_csv(data_path + 'interp_traj.csv')
    Interp = pd.concat([Interp, this_s])


#%%

'''
Checking the number of visually discarded trajectories
'''

grouped = Interp.groupby(['Subject', 'task', 'cong'], as_index = False).sum()
# dividing by 100 because it is the number of trajectory points
grouped['discard_on_shape'] = grouped['discard_on_shape']/100


# plt.figure()
# sns.pointplot(grouped, x = 'task', y = 'discard_on_shape', hue = 'cong', errorbar = 'ci', dodge = True)

# since it is a count measure, it is non normally distributed
pg.normality(grouped['discard_on_shape'])


## We need a glm with poisson
grouped.to_csv('D:/JointRSA/JointDrawing/group/VisualCheck.csv')

## SEE JAMOVI OUTPUT FILE


## They don't make significantly more errors in one or the other condition
grouped = grouped.groupby(['Subject'], as_index = False).sum()

grouped.discard_on_shape.mean()
grouped.discard_on_shape.std()


percent = [grouped.loc[x, 'discard_on_shape']*100/480 for x in range(len(subjlist))]

print(
    "Percent discarded trials: %s, ±%s" % (np.stack(percent).mean(),
    np.stack(percent).std()
        )
    )


# DISCARDING TRIALS BASED ON VISUAL INSPECTION
Interp = Interp[Interp.discard_on_shape == 0]



#%%

'''
Checking the number of error trials
these are trials in which the b-spline fit does not match the instructed shape
(i.e., the participant drew did not draw the instructed shape)
'''

data = Interp.groupby(['Subject', 'task', 'cong', 'shape', 'trial_sort'], as_index= False).mean()

for t in range(len(data)):
    if data.loc[t, 'shape'] == 'c' and data.loc[t, 'res_lin'] > data.loc[t, 'res_quad']:
        data.loc[t, 'error_shape'] = 0
    elif data.loc[t, 'shape'] == 'd' and data.loc[t, 'res_lin'] < data.loc[t, 'res_quad']:
        data.loc[t, 'error_shape'] = 0
    else:
        data.loc[t, 'error_shape'] = 1
    
grouped = data.groupby(['Subject', 'task', 'cong'], as_index = False).sum()

# plt.figure()
# sns.pointplot(grouped, x = 'task', y = 'error_shape', hue = 'cong', errorbar = 'ci')

# since it is a count measure, it is non normally distributed
pg.normality(grouped['error_shape'])

## We need a glm with poisson
grouped.to_csv('D:/JointRSA/JointDrawing/group/ErrorShapeCheck.csv')

## SEE JAMOVI OUTPUT FILE
 
## They don't make significantly more errors in one or the other condition
grouped = data.groupby(['Subject'], as_index = False).sum()

grouped.error_shape.mean()
grouped.error_shape.std()

grouped.error_shape.sum()


N_trials = data.groupby(['Subject'], as_index = False).count().task

percent = [grouped.loc[x, 'error_shape']*100/N_trials[x] for x in range(len(subjlist))]

print(
    "Percent discarded trials: %s, ±%s" % (np.stack(percent).mean(),
    np.stack(percent).std()
        )
    )



# We are not discarding significantly more in any condition
Interp['res_diff'] = Interp['res_lin'] - Interp['res_quad']

Interp.loc[(Interp['shape'] == 'c') & (Interp['res_diff'] > 0), 'error_shape'] = 0
Interp.loc[(Interp['shape'] == 'd') & (Interp['res_diff'] < 0), 'error_shape'] = 0
Interp.loc[(Interp['shape'] == 'c') & (Interp['res_diff'] < 0), 'error_shape'] = 1
Interp.loc[(Interp['shape'] == 'd') & (Interp['res_diff'] > 0), 'error_shape'] = 1

# DISCARDING TRIALS BASED ON ERROR SHAPES
Interp = Interp[Interp.error_shape == 0]


#%%

'''
Counting number of trials entering the final analysis
'''

# counting avg # of trials per pp
data = Interp.groupby(['Subject', 'trial_sort'], as_index= False).mean()
data1 = data.groupby(['Subject'], as_index = False).count()

data1['x'].mean()
data1['x'].std()

# check n of trials per condition
Interp.groupby(['Subject', 'task', 'cong'], as_index = True).count()['x'].mean()/100
Interp.groupby(['Subject', 'task', 'cong'], as_index = True).count()['x'].std()/100



#%%

'''
COMPUTING SHAPE TEMPLATES
'''

# avg shapes
avgs = Interp.groupby(['Subject', 'shape', 'traj_point'], as_index = False).mean()


# sns.relplot(x = 'x', y = 'y', data = avgs,  estimator = None, units = 'Subject', col='shape', kind ='line', linewidth = 1)
# plt.suptitle('Average trajectories for each participant')
# plt.show()




#%%

'''
COMPUTING DISTANCE BETWEEN TRIALS AND SHAPE TEMPLATES
For each trial we compute two measures:
    - pairwise euclidean distances between points of the two trajectories
    - area between the two trajectories
'''

# defining formula to compute area between two lines
def area(p):
    # for p: 2D vertices of a polygon:
    # area = 1/2 abs(sum(p0 ^ p1 + p1 ^ p2 + ... + pn-1 ^ p0))
    # where ^ is the cross product
    return np.abs(np.cross(p, np.roll(p, 1, axis=0)).sum()) / 2


print('============= COMPUTING DISTANCES =============')

Distances = pd.DataFrame()

# compute distance of each trial from average

for idx_s, s in enumerate(subjlist):
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + s + '/Results/'
    
    # loading average template per participant
    templ_c = avgs[(avgs['shape'] == 'c') & (avgs.Subject == int(s))].reset_index(drop = True)
    templ_d = avgs[(avgs['shape'] == 'd') & (avgs.Subject == int(s))].reset_index(drop = True)
    
    xy_c = np.c_[templ_c['x'], templ_c['y']] 
    xy_d = np.c_[templ_d['x'], templ_d['y']] 
    
    print('-------------- Subj ' + s + '--------------')

    this_s = pd.DataFrame()
    sub = Interp.groupby('Subject').get_group(int(s))
    trials = np.unique(sub.trial_sort)
    
    for trial in trials:
        
        #print('trial ' + str(trial))

        this_t = sub.groupby('trial_sort').get_group(trial)
        temp = pd.DataFrame()
        temp['x'] = this_t['x']
        temp['y'] = this_t['y']
        temp['Subject'] = this_t['Subject']
        temp['task'] = this_t['task']
        temp['shape'] = this_t['shape']
        temp['cong'] = this_t['cong']
        temp['traj_point'] = this_t['traj_point']
        temp['trial_sort'] = this_t['trial_sort']
        temp['discard_on_shape'] = this_t['discard_on_shape'] 
        temp['error_shape'] = this_t['error_shape']
        
        xy1 = np.c_[this_t['x'], this_t['y']]


        if temp.iloc[0]['shape'] == 'c':
                 
            temp['distance'] = np.diagonal(scipy.spatial.distance.cdist(np.asarray(list(zip(temp['x'], temp['y']))),np.asarray( list(zip(templ_c['x'], templ_c['y']))), metric = 'euclidean'))
            
            p = np.r_[xy_c, xy1[::-1]]
            temp['area'] = area(p)

            
        elif temp.iloc[0]['shape'] == 'd':
            
            temp['distance'] = np.diagonal(scipy.spatial.distance.cdist(np.asarray(list(zip(temp['x'], temp['y']))),np.asarray( list(zip(templ_d['x'], templ_d['y']))), metric = 'euclidean')  )  

            p = np.r_[xy_d, xy1[::-1]]
            temp['area'] = area(p)

            
        this_s = pd.concat([this_s, temp])
        
    this_s.to_csv(data_path + 'dist.csv')
    
    Distances = pd.concat([Distances, this_s], ignore_index=True)


to_save = Distances.groupby(['Subject', 'task', 'cong', 'shape', 'trial_sort'], as_index = True).mean()
to_save.to_csv('D:/JointRSA/JointDrawing/group/Distances.csv')




