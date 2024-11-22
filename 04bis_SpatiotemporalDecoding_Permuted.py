# -*- coding: utf-8 -*-

"""

JOINT DRAWING EXPERIMENT

Spatiotemporal decoding
1. Decoding between congruent and incongruent shape combinations
2. Decoding participant's and partner's movements
    
@author: Silvia Formica
silvia.formica@hu-berlin.de
"""

# Importing modules

import os
import numpy as np
import mne
from mne.decoding import cross_val_multiscore, Vectorizer, Scaler, LinearModel
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import random
import itertools

####################################################################
## FUNCTION TO CREATE SUPERTRIALS

# Arguments of the function:
# epochs_list = list of arrays of the conditions to create supertrials from
# n_sup = total n of supertrials to create
# n_trials = how many trials to take from each array in epohs_list
# n_bins = how many temporal bins to average
# norm = if set to 1, the created super trials are normalized along the time dimension
# (i.e., the average voltage is subtracted from all timepoints before binning)
####################################################################

def supertrialer(epochs_list, n_sup, n_trials, n_bins, norm=0):
    
    epochs_avg = []

    ## randomizing trial
    epochs_list = [x[random.sample(list(np.arange(len(x))), k=len(x))] for x in epochs_list]

    for sup_idx in range(n_sup):
        
        # averaging n_trials from each array in the list
        partitions = []
        for idx, arr in enumerate(epochs_list):
            partitions.append(arr[sup_idx*n_trials:sup_idx*n_trials+n_trials].mean(axis = 0))
        
        ep_avg = np.mean(np.stack(partitions), axis = 0)
        
        # Normalization
        if norm == 1:
            averaged = ep_avg.mean(axis = 1)
            ep_avg_norm = ep_avg - averaged.reshape(64, 1)
            ep_avg = ep_avg_norm
        
        # binning in time
        bin_edges = np.round(np.histogram_bin_edges(np.arange(ep_avg.shape[1]), bins = n_bins))
        temp = ([ep_avg[:, int(bin_edges[x]):int(bin_edges[x+1])] for x in range(len(bin_edges)-1)])
        temp1 = [x.mean(axis = 1) for x in temp]
        temp2 = np.stack(temp1).T
        epochs_avg.append(temp2.flatten())
       
    epochs_avg = np.stack(epochs_avg)
    
    # print('# supertrials = ' + str(n_sup))
    # print('# trials for each supertrial = ' + str(n_trials*len(epochs_list)))
    return epochs_avg

#%%

os.chdir('D:/JointRSA/JointDrawing/')

subjlist  = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

subjlist.remove('08')
subjlist.remove('17')    # bad EEG data
subjlist.remove('21')    # problem with EEG recording
subjlist.remove('23')    # low performance on catch trials


## Setting parameters
n_perm = 100   # Number of permutations
n_sup = 10     # Number of supertrials
n_bins = 10    # Number of time bins

# Define a decoding pipeline:
clf = make_pipeline(
    Scaler(scalings = 'mean'),                   # 2) normalize features across trials
    Vectorizer(),                      # 1) vectorize across time and channels
    LinearModel(                        # 3) fits a logistic regression
        LDA(solver = 'lsqr', shrinkage = 'auto')
    )
)


#%%

'''
Decoding Congruent and Incongruenct combinations
'''

# We take two trials for each array in epochs_list
n_trials = 2

for subj in subjlist:
    
    print('----- Subject ' + str(subj))       
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    os.chdir('D:/JointRSA/JointDrawing/sub-' + subj + '/Results/')
    # data_path = ('/home/formicas/JointDrawing/sub-' + subj + '/')   # if running this on server
    # os.chdir('/home/formicas/JointDrawing/sub-' + subj + '/')    
    
    epochs = mne.read_epochs(data_path + 'epochs-epo.fif', verbose =False)
    
    # create directory for saving
    if not os.path.isdir(data_path + '/spatiotemporal_decoding_norm'):
        os.mkdir(data_path + '/spatiotemporal_decoding_norm')    
    os.chdir(data_path + '/spatiotemporal_decoding_norm/')

    
    ## Keeping only relevant trials for decoding analyses
    epochs = epochs.pick('eeg').apply_baseline((-0.2, 0))
      
    ## Data preparation
    # rejecting missed catch trials
    epochs = epochs[~((epochs.metadata.catch == 1) & (epochs.metadata.catch_detected == 0))]
    # removing false alarms
    epochs = epochs[~((epochs.metadata.catch == 0) & (np.isnan(epochs.metadata.pp_finish_time)== 1))]
    
    ## Computing correct cue pair for this participant and updating metadata
    meta = epochs.metadata
    
    # Four possible symbolic cues are grouped in all possible combinations of pairs
    possible_cues = ['#', '%', '&', '$']
    circle_cues = list(itertools.combinations(possible_cues, 2))
    diamond_cues = []
    for pair in circle_cues:
        a = [x for x in possible_cues if x not in pair]
        diamond_cues.append(tuple(a))
    
    # For each participant, one combination is selected (counterbalanced)
    thisp_circle_cues = circle_cues[int(meta.iloc[0]['CueCombination'])]
    thisp_diamond_cues = diamond_cues[int(meta.iloc[0]['CueCombination'])]
    a = list(thisp_circle_cues)
    b = list(thisp_diamond_cues)
    
    pair1 = [a[0], b[0]]
    pair2 = [a[1], b[1]]
    
    meta['cue_pair'] = [1 if meta.loc[idx, 'cue_participant'] and meta.loc[idx, 'cue_coagent'] in pair1 else 2 for idx in meta.index]
    
    epochs.metadata = meta
    
    epochs = epochs.crop(tmin = 0 , tmax = 2)
    timep = epochs.times

    ##############################################################################
    ## JOINT TASK
    ##############################################################################
    
    ## Extracting data per condition

    cc_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    dc_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    cc_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    dc_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    cd_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    dd_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    cd_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    dd_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    
    # getting data from epochs
    cc_1_j_data = cc_1_j.get_data()
    dc_1_j_data = dc_1_j.get_data()
    cc_2_j_data = cc_2_j.get_data()
    dc_2_j_data = dc_2_j.get_data()  
    cd_1_j_data = cd_1_j.get_data()
    dd_1_j_data = dd_1_j.get_data()
    cd_2_j_data = cd_2_j.get_data()
    dd_2_j_data = dd_2_j.get_data() 
    
    # finding condition with lowest n of trials
    n_epochs = np.asarray([len(x) for x in [cc_1_j, cc_2_j, cd_1_j, cd_2_j, dc_1_j, dc_2_j, dd_1_j, dd_2_j]]).min()


    ##############################################################################
    ## JOINT TASK -- Congruent combinations
    ##############################################################################

    all_scores = []
    coef = []
    filt = []
    
    for perm in range(n_perm):
        print('')
        print(' -- joint permutation ' + str(perm))
        
        me_c = supertrialer([cc_1_j_data, cc_2_j_data], n_sup, n_trials, n_bins, 1)
        me_d = supertrialer([dd_1_j_data, dd_2_j_data], n_sup, n_trials, n_bins, 1)
    
        ## Creating input for classifier
        X = np.vstack([me_c, me_d])
        y_supertrials = np.tile([0,1], len(me_c))

        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('Me_joint_cong_perm', np.stack(all_scores))



    ##############################################################################
    ## JOINT TASK -- Incongruent combinations
    ##############################################################################

    all_scores = []
    coef = []
    filt = []
    
    for perm in range(n_perm):
        you_c = supertrialer([dc_1_j_data, dc_2_j_data], n_sup,n_trials, n_bins, 1)
        you_d = supertrialer([cd_1_j_data, cd_2_j_data], n_sup,n_trials, n_bins, 1)
    
        ## Creating input for classifier
        X = np.vstack([you_c, you_d])
        y_supertrials = np.tile([0,1], len(you_c))

        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('Me_joint_incong_perm', np.stack(all_scores))

    ##############################################################################
    ## PARALLEL TASK
    ##############################################################################
    
    ## Extracting data per condition

    cc_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    dc_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    cc_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    dc_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    cd_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    dd_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    cd_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    dd_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    # getting data from epochs
    cc_1_p_data = cc_1_p.get_data()
    dc_1_p_data = dc_1_p.get_data()
    cc_2_p_data = cc_2_p.get_data()
    dc_2_p_data = dc_2_p.get_data()  
    cd_1_p_data = cd_1_p.get_data()
    dd_1_p_data = dd_1_p.get_data()
    cd_2_p_data = cd_2_p.get_data()
    dd_2_p_data = dd_2_p.get_data() 
    
    # finding condition with lowest n of trials
    n_epochs = np.asarray([len(x) for x in [cc_1_p, cc_2_p, cd_1_p, cd_2_p, dc_1_p, dc_2_p, dd_1_p, dd_2_p]]).min()


    ##############################################################################
    ## PARALLEL TASK -- Congruent combinations
    ##############################################################################

    all_scores = []
    coef = []
    filt = []
    
    for perm in range(n_perm):
        me_c = supertrialer([cc_1_p_data, cc_2_p_data], n_sup,n_trials, n_bins, 1)
        me_d = supertrialer([dd_1_p_data, dd_2_p_data], n_sup,n_trials, n_bins, 1)
    
        ## Creating input for classifier
        X = np.vstack([me_c, me_d])
        y_supertrials = np.tile([0,1], len(me_c))
        
        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('Me_parallel_cong_perm', np.stack(all_scores))

    ##############################################################################
    ## PARALLEL TASK -- Incongruent combinations
    ##############################################################################

    all_scores = []
    coef = []
    filt = []
    
    for perm in range(n_perm):
        print('')
        print(' -- parallel permutation ' + str(perm))
        you_c = supertrialer([dc_1_p_data, dc_2_p_data], n_sup,n_trials, n_bins, 1)
        you_d = supertrialer([cd_1_p_data, cd_2_p_data], n_sup,n_trials, n_bins, 1)
    
        ## Creating input for classifier
        X = np.vstack([you_c, you_d])
        y_supertrials = np.tile([0,1], len(you_c))

        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('Me_parallel_incong_perm', np.stack(all_scores))



#%%

'''
Decoding Participant's and Partner's movements
'''
# We take one trial for each array in epochs_list
# to equate the number of trials entering the supertrials with the previous analysis
n_trials = 1


for subj in subjlist:
    
    print('----- Subject ' + str(subj))       
    
    
    data_path = 'D:/JointRSA/JointDrawing/sub-' + subj + '/Results/'
    os.chdir('D:/JointRSA/JointDrawing/sub-' + subj + '/Results/')
    # data_path = ('/home/formicas/JointDrawing/sub-' + subj + '/')
    # os.chdir('/home/formicas/JointDrawing/sub-' + subj + '/')    
    
    epochs = mne.read_epochs(data_path + 'epochs-epo.fif', verbose =False)
    
    # create directory for saving
    if not os.path.isdir(data_path + '/spatiotemporal_decoding_norm'):
        os.mkdir(data_path + '/spatiotemporal_decoding_norm')    
    os.chdir(data_path + '/spatiotemporal_decoding_norm/')

    ## Keeping only relevant trials for decoding analyses
    epochs = epochs.pick('eeg').apply_baseline((-0.2, 0))
      
    ## Data preparation
    # rejecting missed catch trials
    epochs = epochs[~((epochs.metadata.catch == 1) & (epochs.metadata.catch_detected == 0))]
    # removing false alarms
    epochs = epochs[~((epochs.metadata.catch == 0) & (np.isnan(epochs.metadata.pp_finish_time)== 1))]
    
    ## Computing correct cue pair for this participant and updating metadata
    meta = epochs.metadata
    
    # Four possible symbolic cues are grouped in all possible combinations of pairs
    possible_cues = ['#', '%', '&', '$']
    circle_cues = list(itertools.combinations(possible_cues, 2))
    diamond_cues = []
    for pair in circle_cues:
        a = [x for x in possible_cues if x not in pair]
        diamond_cues.append(tuple(a))
    
    # For each participant, one combination is selected (counterbalanced)
    thisp_circle_cues = circle_cues[int(meta.iloc[0]['CueCombination'])]
    thisp_diamond_cues = diamond_cues[int(meta.iloc[0]['CueCombination'])]
    a = list(thisp_circle_cues)
    b = list(thisp_diamond_cues)
    
    pair1 = [a[0], b[0]]
    pair2 = [a[1], b[1]]
    
    meta['cue_pair'] = [1 if meta.loc[idx, 'cue_participant'] and meta.loc[idx, 'cue_coagent'] in pair1 else 2 for idx in meta.index]
    
    epochs.metadata = meta
    
    epochs = epochs.crop(tmin = 0 , tmax = 2)
    timep = epochs.times


    #%% 
    
    ##############################################################################
    ## JOINT TASK
    ##############################################################################
    
    ## Extracting data per condition

    cc_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    dc_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    cc_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    dc_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    cd_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    dd_1_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    cd_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    dd_2_j = epochs[(epochs.metadata.task == 'Joint') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    # getting data from epochs
    cc_1_j_data = cc_1_j.get_data()
    dc_1_j_data = dc_1_j.get_data()
    cc_2_j_data = cc_2_j.get_data()
    dc_2_j_data = dc_2_j.get_data()  
    cd_1_j_data = cd_1_j.get_data()
    dd_1_j_data = dd_1_j.get_data()
    cd_2_j_data = cd_2_j.get_data()
    dd_2_j_data = dd_2_j.get_data() 
    
    # finding condition with lowest n of trials
    n_epochs = np.asarray([len(x) for x in [cc_1_j, cc_2_j, cd_1_j, cd_2_j, dc_1_j, dc_2_j, dd_1_j, dd_2_j]]).min()

    ##############################################################################
    ## JOINT TASK -- Participant's movement
    ##############################################################################

    all_scores = []
    
    for perm in range(n_perm):
        print('')
        print(' -- joint permutation ' + str(perm))
        
        # creating supertrials
        me_c = supertrialer([cc_1_j_data, cc_2_j_data, cd_1_j_data, cd_2_j_data], n_sup, n_trials, n_bins, 1)
        me_d = supertrialer([dd_1_j_data, dd_2_j_data, dc_1_j_data, dc_2_j_data], n_sup, n_trials , n_bins, 1)
    
        ## Creating input for classifier
        X = np.vstack([me_c, me_d])
        y_supertrials = np.tile([0,1], len(me_c))
    
        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('Me_joint_perm', np.stack(all_scores))

    ##############################################################################
    ## JOINT TASK -- Partner's movement
    ##############################################################################

    all_scores = []
    
    for perm in range(n_perm):
        you_c = supertrialer([cc_1_j_data, cc_2_j_data, dc_1_j_data, dc_2_j_data], n_sup,n_trials, n_bins, 1)
        you_d = supertrialer([dd_1_j_data, dd_2_j_data, cd_1_j_data, cd_2_j_data], n_sup,n_trials, n_bins, 1)
    
    
        ## Creating input for classifier
        X = np.vstack([you_c, you_d])
        y_supertrials = np.tile([0,1], len(you_c))

    
        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('You_joint_perm', np.stack(all_scores))

#%%

    ##############################################################################
    ## PARALLEL TASK
    ##############################################################################
    
    ## Extracting data per condition

    cc_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    dc_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 1)]
    
    cc_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    dc_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'c') & (epochs.metadata.cue_pair == 2)]
    
    cd_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    dd_1_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 1)]
    
    cd_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'c') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    dd_2_p = epochs[(epochs.metadata.task == 'Parallel') & (epochs.metadata.cue_participant_letter == 'd') & (epochs.metadata.cue_coagent_letter == 'd') & (epochs.metadata.cue_pair == 2)]
    
    # getting data from epochs
    cc_1_p_data = cc_1_p.get_data()
    dc_1_p_data = dc_1_p.get_data()
    cc_2_p_data = cc_2_p.get_data()
    dc_2_p_data = dc_2_p.get_data()  
    cd_1_p_data = cd_1_p.get_data()
    dd_1_p_data = dd_1_p.get_data()
    cd_2_p_data = cd_2_p.get_data()
    dd_2_p_data = dd_2_p.get_data() 
    
    # finding condition with lowest n of trials
    n_epochs = np.asarray([len(x) for x in [cc_1_p, cc_2_p, cd_1_p, cd_2_p, dc_1_p, dc_2_p, dd_1_p, dd_2_p]]).min()

    ##############################################################################
    ## PARALLEL TASK -- Participant's movement
    ##############################################################################

    all_scores = []

    for perm in range(n_perm):
        me_c = supertrialer([cc_1_p_data, cc_2_p_data, cd_1_p_data, cd_2_p_data], n_sup,n_trials, n_bins, 1)
        me_d = supertrialer([dd_1_p_data, dd_2_p_data, dc_1_p_data, dc_2_p_data], n_sup,n_trials, n_bins, 1)
    
    
        ## Creating input for classifier
        X = np.vstack([me_c, me_d])
        y_supertrials = np.tile([0,1], len(me_c))
    
    
        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('Me_parallel_perm', np.stack(all_scores))

    ##############################################################################
    ## PARALLEL TASK -- Partner's movement
    ##############################################################################

    all_scores = []

    for perm in range(n_perm):
        print('')
        print(' -- parallel permutation ' + str(perm))
        you_c = supertrialer([cc_1_p_data, cc_2_p_data, dc_1_p_data, dc_2_p_data], n_sup,n_trials, n_bins, 1)
        you_d = supertrialer([dd_1_p_data, dd_2_p_data, cd_1_p_data, cd_2_p_data], n_sup,n_trials, n_bins, 1)
    
    
        ## Creating input for classifier
        X = np.vstack([you_c, you_d])
        y_supertrials = np.tile([0,1], len(you_c))

    
        # cross-validated classification
        scores = cross_val_multiscore(clf, X, y_supertrials, cv = 5, n_jobs = 4, verbose=False)
        
        all_scores.append(scores)

    # Saving results
    np.save('You_parallel_perm', np.stack(all_scores))
