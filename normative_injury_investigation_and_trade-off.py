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
import copy

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


plt.rcParams.update({'font.size': 20})

figPrint = 'true'
figFolder = os.path.join('simulations/behaviour_infogain/figures')

# X1 = Not injured (inferred)
# X2 = Injured (inferred)
# A1 = Commit to an activity (terminates episode)
# A2 = Not commit to activity (terminates episode) 
# A3 = Move the injured body part (more informative if true state injured, otherwise less informative)
# A4 = Not move the injured body part (less informative action)
# A1 in X1 gives R = +100
# A2 in X1 gives R = -50 (or -100)
# A1 in X2 gives R = -800 (vary)
# A2 in X2 gives R = +100

selves = []
# xObvs_injured = ['A3_md15.0_std1_15_std2_15', 42.5, 57.5, 15, 15]
xObvs_injured = ['A3_md5.0_std1_30_std2_30', 47.5, 52.5, 30, 30]


A3Cost =  np.array([-4]) # this reduces in the late stages of recovery 
obvStep = np.array([1]) #...this discretizes the observations which go from 1 to 100
sim = POMDP(xObvs_injured,obvStep)
sim.belief_transition_matrix()
sim.belief_transition_matrix_plot(figFolder)
allRewards = np.array([[ 100 ,  -100,    -400 ,  100 ]])
allCosts = A3Cost
sim.value_iteration(allRewards,A3Cost,figPrint,figFolder)
print('finished value iteration')
rewardID = range(len(allRewards)) #...specify which rewards and costs you want to print for
costID = range(len(allCosts))
# print(rewardID, costID)
thresh1a, thresh2a = qvalues_figure(sim,rewardID, costID,figFolder)
selves.append(sim)

qvalues_figure4(figFolder)

# beliefStart = 0.5
# sample_sequence_observation_accumulated_cost(sim,rewardID,costID,beliefStart,100,1,"no_mismatch", obv_truestate=0)
# qvalues_figure5(sim, rewardID, costID, figFolder) # injury investigation
# print("done")

A3Cost =  np.array([-16]) # this reduces in the late stages of recovery 
obvStep = np.array([1]) #...this discretizes the observations which go from 1 to 100
sim2 = POMDP(xObvs_injured,obvStep)
sim2.belief_transition_matrix()
sim2.belief_transition_matrix_plot(figFolder)
allRewards = np.array([[ 100 ,  -100,    -400 ,  100 ]])
allCosts = A3Cost
sim2.value_iteration(allRewards,A3Cost,figPrint,figFolder)
print('finished value iteration')
rewardID = range(len(allRewards)) #...specify which rewards and costs you want to print for
costID = range(len(allCosts))
# print(rewardID, costID)
thresh1b, thresh2b = qvalues_figure(sim2,rewardID, costID,figFolder)
selves.append(sim2)


# intersections = [thresh1a, thresh2a, thresh1b, thresh2b]

# qvalues_figure3_balance(selves, intersections, figFolder)




# A3Cost =  np.array([-8])
# obvStep = np.array([1])
# sim800 = POMDP(xObvs_injured,obvStep)
# sim800.belief_transition_matrix()
# sim800.belief_transition_matrix_plot(figFolder)
# allRewards = np.array([[ 100 ,  -100,    -800 ,  100 ]])
# allCosts = A3Cost
# sim800.value_iteration(allRewards,A3Cost,figPrint,figFolder)

# sim600 = POMDP(xObvs_injured,obvStep)
# sim600.belief_transition_matrix()
# sim600.belief_transition_matrix_plot(figFolder)
# allRewards = np.array([[ 100 ,  -100,    -600 ,  100 ]])
# allCosts = A3Cost
# sim600.value_iteration(allRewards,A3Cost,figPrint,figFolder)

# sim400 = POMDP(xObvs_injured,obvStep)
# sim400.belief_transition_matrix()
# sim400.belief_transition_matrix_plot(figFolder)
# allRewards = np.array([[ 100 ,  -100,    -400 ,  100 ]])
# allCosts = A3Cost
# sim400.value_iteration(allRewards,A3Cost,figPrint,figFolder)

# sim200 = POMDP(xObvs_injured,obvStep)
# sim200.belief_transition_matrix()
# sim200.belief_transition_matrix_plot(figFolder)
# allRewards = np.array([[ 100 ,  -100,    -200 ,  100 ]])
# allCosts = A3Cost
# sim200.value_iteration(allRewards,A3Cost,figPrint,figFolder)

# qvalues_figure2_punishmentsens([sim800, sim600, sim400, sim200], figFolder)