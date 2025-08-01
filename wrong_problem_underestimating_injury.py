"""
This module simulates a Partially Observable Markov Decision Process (POMDP) environment
to analyze the effects of different belief states on decision-making in the context of injury 
assessment and management. The simulation evaluates the outcomes of various actions taken 
based on the true state of the system (injured or not injured) and computes the associated 
rewards and costs.

Key components of the simulation include:
- Importing necessary libraries for data manipulation, visualization, and POMDP functionality.
- Defining the reward structure based on the true state and actions taken.
- Running simulations for different initial belief states and collecting results.
- Visualizing the average rewards over multiple trials and saving the results for further analysis.

Functions:
- run_one_costly_action_sim: Simulates a single costly action in a POMDP environment.
"""
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
from build_object_rumPOMDP import * 
from convenience_functions_rumPOMDP import * 
from modelFunctions_rumPOMDP import * 
from sample_sequence_functions_rumPOMDP import * 
from rewards_rumPOMDP import * 
from hidden_states_rumPOMDP import * 
from qvalues_rumPOMDP import *
from sample_sequence_functions_rumPOMDP import *

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

def run_one_costly_action_sim(true_state, figFolder, b0=0.5): 
    """
    Simulates a single costly action in a POMDP (Partially Observable Markov Decision Process) environment.

    Parameters:
    true_state (str): The true state of the system, which influences the observations and rewards.
    figFolder (str): The folder path where figures will be saved.
    b0 (float, optional): The initial belief state. Defaults to 0.5, usually set to midpoint of thresholds.

    Returns:
    tuple: A tuple containing the POMDP simulation object, and two threshold values (thresh1, thresh2).
    """
    obvStep = np.array([1]) #...this discretizes the observations which go from 1 to 100
    if true_state == "injured_01": #gain>pain
        xObvs = ['A3_md15.0_std1_15_std2_15', 42.5, 57.5, 15, 15]
        allRewards = np.array([[ 100 ,  -100,    -400. ,  100 ]])
        A3Cost =  np.array([-4]) 

    figPrint = 'true'
    
    sim = POMDP(xObvs,obvStep)
    sim.belief_transition_matrix()
    sim.belief_transition_matrix_plot(figFolder)

    sim.value_iteration(allRewards,A3Cost,figPrint,figFolder)
    print('finished value iteration')

    rewardID = range(1)#...specify which rewards and costs you want to print for
    costID = range(len(A3Cost))
    thresh1, thresh2 = qvalues_figure(sim,rewardID, costID, figFolder)
    print(thresh1, thresh2)

    beliefStart = b0
    maxSamples  = 100 #...maximum number of samples on each trial
    noTrials = 1000 #...number of independent trials/agents

    obv_mismatch = "no_mismatch"
    # injured_01
    sample_sequence_observation_accumulated_cost(sim,rewardID, costID, beliefStart,maxSamples,noTrials,obv_mismatch,obv_truestate=1, A3Cost=A3Cost)
   
    return sim, thresh1, thresh2



figFolder = os.path.join('simulations/behaviour_underestimation/figures/')


true_state = "injured_01"
avg_r = []
std_r = []
for b0 in np.arange(0, 1.1, 0.1):
    sim, thresh1, thresh2 = run_one_costly_action_sim(true_state, figFolder, b0)
    all_r = []
    for t in range(1000): #no agents episodes      
        r= -sim.cumulative_phasic_pain[t]
        # print(r,sim.noA3[t], sim.noA4[t])

        all_r.append(r)
    avg_r.append(np.mean(all_r))
    std_r.append(np.mean(std_r))
    print(np.mean(all_r))

with open(figFolder+'avg_r.npy', 'wb') as f:
    np.save(f, np.array(avg_r))
with open(figFolder+'std_r.npy', 'wb') as f:
    np.save(f, np.array(std_r))

# avg_r = np.load(figFolder+'avg_r.npy')
# std_r = np.load(figFolder+'std_r.npy')
# thresh1 = 0
# thresh2 = 0.92


fig,axs = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(8)
yLims = [0, 1]
axs.plot(np.array(avg_r),'black',linewidth=5)
# axs.errorbar(np.arange(0, len(avg_r), 1), np.array(avg_r), yerr=np.array(std_r))
axs.axvline(thresh1*10, linestyle="--", color="k", zorder=0, linewidth=3)
axs.axvline(thresh2*10, linestyle="--", color="k", zorder=0, linewidth=3)
# axs.legend()
axs.set(xlim = [0,10], xticks = [0, 5, 10], xticklabels=[0, 0.5, 1])
# axs.set(ylim = yLims)
axs.spines['left'].set_linewidth(3)  
axs.spines['bottom'].set_linewidth(3)
axs.spines['right'].set_linewidth(0)  
axs.spines['top'].set_linewidth(0)  
plt.savefig(figFolder+'/samples_plot/PhasicPain_till_termination_diffstartingbeliefs.png', dpi=1500)
plt.close(fig)


