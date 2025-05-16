#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### IMPORT SOME STUFF
#import what is necessary and some things that may be unnecessary
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import numpy as np
import pandas as pd
import random
import seaborn as sns
import math
from scipy import stats
from scipy.interpolate import griddata
import copy
import pickle
import itertools
import importlib
import os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### NUMPY SETTINGS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
np.set_printoptions(precision = 2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### BESPOKE FUNCTIONS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# from build_object_rumPOMDP import * 
from build_object_rumPOMDP_twocostlyactions import * 
from convenience_functions_rumPOMDP import * 
from modelFunctions_rumPOMDP import * 
from sample_sequence_functions_rumPOMDP import * 
from rewards_rumPOMDP import * 
from hidden_states_rumPOMDP import * 
from qvalues_rumPOMDP_twocostlyactions import *
from sample_sequence_functions_rumPOMDP_twocostlyactions import *

# X1 = Not injured (inferred)
# X2 = Injured (inferred)
# A1 = Commit to an activity (terminates episode)
# A2 = Not commit to activity (terminates episode) 
# A3 = Move the injured body part (more informative if true state injured, otherwise less informative)
# A4 = Not move the injured body part (less informative action)

# A1 in X1 gives R = +100
# A2 in X1 gives R = -100
# A1 in X2 gives R = -800 (vary)
# A2 in X2 gives R = +100

def run_two_costly_action_sim(true_state, figFolder): 
    obvStep = np.array([1]) #...this discretizes the observations which go from 1 to 100
    if true_state == "recovered_01": #gain>pain
        xObvs = [['A3_md15.0_std1_15_std2_15', 42.5, 57.5, 15, 15], ['A4_md5.0_std1_30_std2_30', 47.5, 52.5, 30, 30]] # different info gain
        allRewards = np.array([[ 100 ,  -100,    -400. ,  100 ]])
        A3Cost =  np.array([-4]) # this reduces in the late stages of recovery 
        A4Cost =  np.array([-0.5]) 
    elif true_state == "recovered_02": #pain>gain
        xObvs = [['A3_md5.0_std1_30_std2_30', 47.5, 52.5, 30, 30], ['A4_md5.0_std1_30_std2_30', 47.5, 52.5, 30, 30]] # same info gain
        allRewards = np.array([[ 100 ,  -100,    -400. ,  100 ]])
        A3Cost =  np.array([-4]) # this reduces in the late stages of recovery 
        A4Cost =  np.array([0]) 
    elif true_state == "recovered_03": #pain>gain
        xObvs = [['A3_md5.0_std1_30_std2_30', 42.5, 57.5, 30, 30], ['A4_md5.0_std1_30_std2_30', 47.5, 52.5, 30, 30]] # same info gain
        allRewards = np.array([[ 100 ,  -100,    -400. ,  100 ]])
        A3Cost =  np.array([-16]) # this reduces in the late stages of recovery 
        A4Cost =  np.array([-0.5]) 

    figPrint = 'true'
    

    sim = POMDP(xObvs,obvStep)
    sim.belief_transition_matrix()
    sim.belief_transition_matrix_plot(figFolder)

    sim.value_iteration(allRewards,A3Cost, A4Cost,figPrint,figFolder)
    print('finished value iteration')

    rewardID = range(1)#...specify which rewards and costs you want to print for
    thresh1, thresh2 = qvalues_figure(sim,rewardID,figFolder)
    print(thresh1, thresh2)

    costID = range(len(A3Cost))
    beliefStart = 0.5*(thresh1+thresh2)
    maxSamples  = 100 #...maximum number of samples on each trial
    noTrials = 1000 #...number of independent trials/agents

    obv_mismatch = "no_mismatch"
    if true_state=="injured_01" or true_state=="injured_02":
        sample_sequence_observation_accumulated_cost(sim,rewardID, beliefStart,maxSamples,noTrials,obv_mismatch,obv_truestate=1)
    else:
        sample_sequence_observation_accumulated_cost(sim,rewardID, beliefStart,maxSamples,noTrials,obv_mismatch,obv_truestate=0)
    # sample_sequence_figure(sim,rewardID,ratioIdx,figFolder)
    
    return sim, thresh1, thresh2


# thresh1 = 0.04 and thresh2 = 0.74 or 0.89 for recovered_01

# thresh1 = 0.09 and thresh2 = 0.51 for recovered_02


figFolder = os.path.join('simulations/behaviour_info_restriction_simple/figures')


true_state = "recovered_01"
sim_01, thresh1_01, thresh2_01 = run_two_costly_action_sim(true_state, figFolder)
print(sim_01.posterior_beliefs.shape)
print(sim_01.posterior_beliefs_ending_in_A1.shape)
print(sim_01.posterior_beliefs_ending_in_A2.shape)

true_state = "recovered_03" #"recovered_02"
sim_02, thresh1_02, thresh2_02 = run_two_costly_action_sim(true_state, figFolder)
print(sim_02.posterior_beliefs.shape)
print(sim_02.posterior_beliefs_ending_in_A1.shape)
print(sim_02.posterior_beliefs_ending_in_A2.shape)



fig,axs = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(8)
yLims = [0,1]
axs.plot(np.mean(sim_01.posterior_beliefs, axis=0),'blue',linewidth=5, label='Average beliefs')
axs.hlines(thresh1_01, 0, 100, colors='k', linestyle="dashed", linewidth=5)
axs.hlines(thresh2_01, 0, 100,  colors='k', linestyle="dashed", linewidth=5)
axs.plot(np.mean(sim_01.posterior_beliefs_ending_in_A1, axis=0),'#5E5EDA',linewidth=3, label='$a_{act}$ chosen')
axs.plot(np.mean(sim_01.posterior_beliefs_ending_in_A2, axis=0),'#DC267F',linewidth=3, label='$a_{r&r}$ chosen')
axs.legend()
# axs.set(xlim = [0,8], xticks = [0, 4, 8], xticklabels=[0.1, 0.5, 0.9])
axs.set(ylim = yLims)
axs.spines['left'].set_linewidth(3)  
axs.spines['bottom'].set_linewidth(3)
axs.spines['right'].set_linewidth(0)  
axs.spines['top'].set_linewidth(0)  
plt.savefig(figFolder+'/samples_plot/beliefs_over_steps_infogain>phasicpain.png', dpi=1500)
plt.close(fig)

fig,axs = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(5)
yLims = [0,1]
axs.bar(['$a_{act}$\n chosen', '$a_{r&r}$\n chosen'], 0.001*np.array([len(sim_01.posterior_beliefs_ending_in_A1), len(sim_01.posterior_beliefs_ending_in_A2)]), color=['#5E5EDA', '#DC267F'], width=0.5)
axs.set(ylim = yLims)
axs.spines['left'].set_linewidth(3)  
axs.spines['bottom'].set_linewidth(3)
axs.spines['right'].set_linewidth(0)  
axs.spines['top'].set_linewidth(0)  
plt.savefig(figFolder+'/samples_plot/numchosen_infogain>phasicpain.png', dpi=1500)
plt.close(fig)


fig,axs = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(8)
yLims = [0,1]
axs.plot(np.mean(sim_02.posterior_beliefs, axis=0),'blue',linewidth=5, label='Average beliefs')
axs.hlines(thresh1_02, 0, 100, colors='k', linestyle="dashed", linewidth=5)
axs.hlines(thresh2_02, 0, 100,  colors='k', linestyle="dashed", linewidth=5)
axs.plot(np.mean(sim_02.posterior_beliefs_ending_in_A1, axis=0),'#5E5EDA',linewidth=3, label='$a_{act}$ chosen')
axs.plot(np.mean(sim_02.posterior_beliefs_ending_in_A2, axis=0),'#DC267F',linewidth=3, label='$a_{r&r}$ chosen')
axs.legend()
# axs.set(xlim = [0,8], xticks = [0, 4, 8], xticklabels=[0.1, 0.5, 0.9])
axs.set(ylim = yLims)
axs.spines['left'].set_linewidth(3)  
axs.spines['bottom'].set_linewidth(3)
axs.spines['right'].set_linewidth(0)  
axs.spines['top'].set_linewidth(0)  
plt.savefig(figFolder+'/samples_plot/beliefs_over_steps_phasicpain>infogain.png', dpi=1500)
plt.close(fig)


fig,axs = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(5)
yLims = [0,1]
axs.bar(['$a_{act}$\n chosen', '$a_{r&r}$\n chosen'], 0.001*np.array([len(sim_02.posterior_beliefs_ending_in_A1), len(sim_02.posterior_beliefs_ending_in_A2)]), color=['#5E5EDA', '#DC267F'], width=0.5)
axs.set(ylim = yLims)
axs.spines['left'].set_linewidth(3)  
axs.spines['bottom'].set_linewidth(3)
axs.spines['right'].set_linewidth(0)  
axs.spines['top'].set_linewidth(0)  
plt.savefig(figFolder+'/samples_plot/numchosen_phasicpain>infogain.png', dpi=1500)
plt.close(fig)