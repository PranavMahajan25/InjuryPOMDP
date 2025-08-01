"""
This script simulates a Partially Observable Markov Decision Process (POMDP) 
to analyze decision-making in the context of injury recovery. The simulation 
focuses on the effects of different actions on the belief states regarding 
the true state of an injury, and it generates visualizations to illustrate 
the results.

Key components of the simulation include:
- Importing necessary libraries for data manipulation, plotting, and POMDP functions.
- Defining a function `run_one_costly_action_sim` that simulates a single costly action 
    based on the true state of the system and generates relevant figures.
- Running multiple trials with varying initial belief states to observe the 
    impact on posterior beliefs and decision-making.
- Plotting the results to visualize how beliefs evolve over time and how they 
    relate to thresholds for decision-making.

The script also saves figures generated during the simulation to a specified 
directory for further analysis.
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### IMPORT SOME STUFF
#import what is necessary and some things that may be unnecessary
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Patch
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
    Simulates a single costly action in a POMDP (Partially Observable Markov Decision Process) 
    for a given true state and generates relevant figures.

    Parameters:
    true_state (str): The true state of the system, which influences the observation and rewards.
    figFolder (str): The directory where figures will be saved.
    b0 (float, optional): The initial belief state. Defaults to 0.5, usually set to midpoint of thresholds

    Returns:
    tuple: A tuple containing the POMDP simulation object, 
           and two threshold values calculated during the simulation.
    """
    obvStep = np.array([1]) #...this discretizes the observations which go from 1 to 100
    if true_state == "recovered_01": #gain>pain
        xObvs = ['A3_md15.0_std1_15_std2_15', 42.5, 57.5, 15, 15] # different info gain
        allRewards = np.array([[ 100 ,  -100,    -400. ,  100 ]])
        A3Cost =  np.array([-4]) # this reduces in the late stages of recovery  

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
    # recovered_01
    sample_sequence_observation_accumulated_cost(sim,rewardID, costID, beliefStart,maxSamples,noTrials,obv_mismatch,obv_truestate=0)
    # sample_sequence_figure(sim,rewardID,ratioIdx,figFolder)
    
    return sim, thresh1, thresh2



figFolder = os.path.join('simulations/behaviour_overestimation/figures')

true_state = "recovered_01"
sim_01, thresh1_01, thresh2_01 = run_one_costly_action_sim(true_state, figFolder)
print(sim_01.posterior_beliefs.shape)
print(sim_01.posterior_beliefs_ending_in_A1.shape)
print(sim_01.posterior_beliefs_ending_in_A2.shape)

sims_01 = []
sims_01_A1_mean = []
sims_01_A2_mean = []
sims_01_A3_mean = []
# sims_01_A4_mean = []
for b0 in np.arange(0, 1.1, 0.1):
    sim_01, thresh1_01, thresh2_01 = run_one_costly_action_sim(true_state, figFolder, b0)
    sims_01.append(sim_01)
    sims_01_A1_mean.append(0.001*sim_01.posterior_beliefs_ending_in_A1.shape[0])
    sims_01_A2_mean.append(0.001*sim_01.posterior_beliefs_ending_in_A2.shape[0])
    sims_01_A3_mean.append(np.mean(sim_01.noA3))
    # sims_01_A4_mean.append(np.mean(sim_01.noA4))
    print(b0)
    print(sim_01.posterior_beliefs.shape)
    print(sim_01.posterior_beliefs_ending_in_A1.shape)
    print(sim_01.posterior_beliefs_ending_in_A2.shape)

blues = cm.get_cmap("Blues", 10)
fig,axs = plt.subplots(1,1)
fig.set_figheight(8)
fig.set_figwidth(8)
yLims = [0,1]
i=0
for sim_01 in sims_01:
    axs.plot(np.mean(sim_01.posterior_beliefs, axis=0),linewidth=4, color=blues(i))
    i+=1
axs.hlines(thresh1_01, 0, 100, colors='k', linestyle="dashed", linewidth=5)
axs.hlines(thresh2_01, 0, 100,  colors='k', linestyle="dashed", linewidth=5)
# axs.plot(np.mean(sim_01.posterior_beliefs_ending_in_A1, axis=0),'#5E5EDA',linewidth=3, label='$A_{activity}$ chosen')
# axs.plot(np.mean(sim_01.posterior_beliefs_ending_in_A2, axis=0),'#DC267F',linewidth=3, label='$A_{rest}$ chosen')
# axs.set(xlim = [0,8], xticks = [0, 4, 8], xticklabels=[0.1, 0.5, 0.9])
axs.set(ylim = yLims)
axs.spines['left'].set_linewidth(3)  
axs.spines['bottom'].set_linewidth(3)
axs.spines['right'].set_linewidth(0)  
axs.spines['top'].set_linewidth(0)  
plt.savefig(figFolder+'/samples_plot/beliefs_over_steps_infogain>phasicpain.png', dpi=1500)
plt.close(fig)


thresh1_01 = 0
thresh2_01 = 0.92

blues = cm.get_cmap("Blues", 10)
fig, axs = plt.subplots(1, 1)
fig.set_figheight(8)
fig.set_figwidth(10)
yLims = [0, 1]

# Starting belief states
starting_beliefs = np.arange(0, 1.1, 0.1)

# # Plot lines with different colors
# for i, sim_01 in enumerate(sims_01):
#     axs.plot(np.mean(sim_01.posterior_beliefs, axis=0), linewidth=4, color=blues(i))
#     with open(figFolder+'/post_beliefs_'+str(i)+'.npy', 'wb') as f:
#         np.save(f, np.mean(sim_01.posterior_beliefs, axis=0))

# Plot lines with different colors
for i in range(11):
    arr = np.load(figFolder+'/post_beliefs_'+str(i)+'.npy')
    axs.plot(arr, linewidth=4, color=blues(i))
    

# Add dashed lines for thresholds
axs.hlines(thresh1_01, 0, 100, colors='k', linestyle="dashed", linewidth=5)
axs.hlines(thresh2_01, 0, 100, colors='k', linestyle="dashed", linewidth=5)

# Set plot aesthetics
axs.set(ylim=yLims)
axs.spines['left'].set_linewidth(3)
axs.spines['bottom'].set_linewidth(3)
axs.spines['right'].set_linewidth(0)
axs.spines['top'].set_linewidth(0)

# Create a legend mapping colors to starting belief states
legend_elements = [
    Patch(facecolor=blues(i), edgecolor='none', label=f"$b_0$={starting_belief:.1f}")
    for i, starting_belief in enumerate(starting_beliefs)
]
axs.legend(
    handles=legend_elements, 
    title="Starting Beliefs", 
    loc="center left", 
    bbox_to_anchor=(1, 0.5)  # Places the legend outside the plot
)
fig.subplots_adjust(right=0.75)  # Add space to the right for the legend

# Save the figure
plt.savefig(figFolder + '/samples_plot/beliefs_over_steps_infogain>phasicpain.png', dpi=1500)
plt.close(fig)

