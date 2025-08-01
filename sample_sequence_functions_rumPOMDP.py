import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats

import importlib

from modelFunctions_rumPOMDP import *
from convenience_functions_rumPOMDP import *

def sample_sequence_observation_accumulated_cost(self, rewardID, costID, beliefStart, maxSample, noTrials, obv_mismatch, obv_truestate=1, A3Cost=np.array([-4])):
    """
    Samples a sequence of observations and accumulates costs based on the given parameters.

    Parameters:
    -----------
    rewardID : array-like
        An array of reward identifiers used to index the reward values.
    
    costID : array-like
        An array of cost identifiers used to index the cost values.
    
    beliefStart : float
        The initial belief value from which to start sampling.
    
    maxSample : int
        The maximum number of samples to draw in each trial.
    
    noTrials : int
        The number of trials to perform.
    
    obv_mismatch : str
        A string indicating whether to use mismatched observations ('mismatch') or standard observations.
    
    obv_truestate : int, optional
        The true state of the observation (default is 1).
    
    A3Cost : numpy.ndarray, optional
        An array representing the cost associated with action A3 (default is an array with a single value of -4).

    Returns:
    --------
    None
        The function modifies the instance variables of the class it belongs to, storing results of the sampling process.
    
    Notes:
    ------
    - The function performs Bayesian updating of beliefs based on sampled observations and action values.
    - It tracks the number of samples taken before termination and the choices made during the sampling process.
    - Results are stored in instance variables for further analysis.
    """
    # Initialize variables
    avTermChoice = np.full([len(rewardID),len(costID)],np.nan)
    avTermCost = np.full([len(rewardID),len(costID)],np.nan)
    self.noSamples_t = np.full([len(rewardID),len(costID),noTrials],np.nan)
    self.termChoice_t = np.full([len(rewardID),len(costID),noTrials],np.nan)
    self.noSamples_av = np.full([len(rewardID),len(costID),3],np.nan)
    self.posterior_beliefs = np.zeros([noTrials, maxSample]) 
    self.noA3 = np.full([noTrials],np.nan)
    self.cumulative_phasic_pain = np.full([noTrials],np.nan)
    self.posterior_beliefs_ending_in_A1 = []
    self.posterior_beliefs_ending_in_A2 = []

    for r,rewards in enumerate(rewardID):
        
            for c,cost in enumerate(costID):
                
                beliefValues = self.beliefValues[rewards,cost,:]
                actionValues = self.actionValues[rewards,cost,:,:]
                beliefRange = self.beliefRange
                
                termChoice = np.full([noTrials],np.nan)
                termCost = np.full([noTrials],np.nan)
                termQ = np.full([noTrials],np.nan)

                for t in range(noTrials):
                    
                    if obv_mismatch=="mismatch":
                    #    print('mismatching std or means')
                        obvSequence = np.random.normal(loc=self.xMeans_mismatch[obv_truestate],scale=self.xStd_mismatch[obv_truestate],size=maxSample)
                        # obvSequence = np.random.normal(loc=self.xMeans_mismatch[1],scale=self.xStd_mismatch[1],size=maxSample)
                      #  print(self.xStd_mismatch[1])

                    else:
                     #   print('std and means as standard')
                        obvSequence = np.random.normal(loc=self.xMeans[obv_truestate], scale=self.xStd[obv_truestate], size=maxSample)

                    qVals_samp = np.zeros([maxSample,3])

                    prior_samp = np.zeros([maxSample]) 
                    posterior_samp = np.zeros([maxSample])

                    margEvid_samp = np.zeros([maxSample])

                    choiceProbs_samp = np.zeros([maxSample,len(costID),3])
                    choiceMade_samp = np.zeros([maxSample])
                    phasicpainA3_samp = np.zeros([maxSample])

                    idealChoice_samp = np.zeros([maxSample])

                    samp = 0
                    prior_samp[samp] = beliefStart
                    posterior_samp[samp] = beliefStart 
                    
                    sampling = True
                    maxSample_below = True
                    
                    while sampling and maxSample_below:
                    
                        #get the qvalues
                        ibelief = index_finder(beliefRange,prior_samp[samp])
                        qVals_samp[samp,:] = actionValues[ibelief,:] 
                        
                        #get the best choice
                        getWhere = np.where(qVals_samp[samp,:]==np.max(qVals_samp[samp,:]))
                        idealChoice_samp[samp] = getWhere[0][0]

                        # print("choice = ", idealChoice_samp[samp])

                        choiceMade_samp[samp] = idealChoice_samp[samp]
                        
                        #end if the choice is to stop sampling
                        sampling= idealChoice_samp[samp]==2 #keep running if it is 2 (sample action)

                        if sampling:
                            phasicpainA3_samp[samp] = prior_samp[samp] * A3Cost[0]
                        
                        maxSample_below = samp<maxSample #keep running if t is less than maxSample

                        if not sampling: #...end if they chose a terminating action
                            break

                        #update the belief for the loop
                        iobv = index_finder(self.obvs,obvSequence[samp])
                        
                        priorsX = self.obvPrior[:,iobv]
                        priorsX = np.reshape(priorsX,(2,1))
                        
                        _,posterior_samp[samp]  = bayes_theorem(prior_samp[samp],priorsX,'-')

                        # print(obvSequence[samp], iobv, priorsX, prior_samp[samp], posterior_samp[samp])

                        maxSample_below = samp<(maxSample-1) #...if there 
                        
                        if not maxSample_below:
                            break
                            
                        prior_samp[samp+1] = posterior_samp[samp]
                        samp = samp +1
 
                    self.noSamples_t[r,c,t] = samp #...for every trial, the number of samples before termination
                    self.termChoice_t[r,c,t] = idealChoice_samp[samp]
                    # print(self.noSamples_t[r,c,t], self.termChoice_t[r,c,t]) 
                
                    ##### new addition
                    self.posterior_beliefs[t, :] = prior_samp
                    self.posterior_beliefs[t, samp:maxSample] = prior_samp[samp] 

                    self.noSamples_av[r,c,0] = np.nanmean(self.noSamples_t[r,c,self.termChoice_t[r,c,:]==0]) 
                    self.noSamples_av[r,c,1] = np.nanmean(self.noSamples_t[r,c,self.termChoice_t[r,c,:]==1]) 
                    
                    self.noSamples_av[r,c,2] = np.nanmean(self.noSamples_t[r,c,:]) #...average number of samples across kRuns

                    # ##### new addition
                    self.noA3[t] = np.sum(choiceMade_samp==2)
                    self.cumulative_phasic_pain[t] = np.sum(phasicpainA3_samp)

                    if choiceMade_samp[samp] == 0:
                        self.posterior_beliefs_ending_in_A1.append(self.posterior_beliefs[t, :])
                    elif choiceMade_samp[samp] == 1:
                        self.posterior_beliefs_ending_in_A2.append(self.posterior_beliefs[t, :]) 

    #end of trials
    self.posterior_beliefs_ending_in_A1 = np.array(self.posterior_beliefs_ending_in_A1)
    self.posterior_beliefs_ending_in_A2 = np.array(self.posterior_beliefs_ending_in_A2)

        
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ### Plotting code - Plot 2 shows a heatmap of sampling
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# def sample_sequence_figure(self,allRewards,allCosts,rewardID,costID,ratioidx,figSave):
                                                        
#     plotRewards = self.allRewards
#     rewardRatio = plotRewards[:,ratioidx[0]]/plotRewards[:,ratioidx[1]]
    
#     fig2,axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [1,1]}, sharex=False, sharey=True)            
#     fig2.set_figheight(5)
#     fig2.set_figwidth(7)
    
#     plt.rcParams.update({'font.size': 18})

#     lines = [":","dashed","-"]

#     lineColors = ["#FE6100","#FEC400","#FEEB00",'black','black']
            
#     for action in range(2): 
                
#         xData = np.arange(0,len(rewardRatio),1)#rewardRatio
            
#         for c in range(len(costID)):
            
#             yData_1 = self.decisionThreshold[:,c,action]
            
#             axs[action].plot(xData,yData_1,color='black',linewidth = 1,linestyle=lines[c])
            
#         axs[action].set_title(self.name, fontsize=10)
               
#         axs[action].set_ylim([0,1])
#         axs[action].set_yticks([0,0.5,1])
        
#         axs[action].set_xlim([-1,len(xData)-0.5])            
#         axs[action].set_xticks(xData)                 
#         axs[action].set_xticklabels(rewardRatio)                
#         axs[action].tick_params(axis='x', rotation=45, width=4)
        
#         axs2 = axs[action].twinx()    
        
#         #run for the sampling
#         for c in range(len(costID)):     
            
#             yData_2 = self.noSamples_av[:,c,action]
#             axs2.plot(xData,yData_2,color=lineColors[action],linewidth = 3,linestyle=lines[c])

#         axs2.set_ylim([0,50])        

#         axs[action].spines['left'].set_linewidth(3)  
#         axs[action].spines['bottom'].set_linewidth(3)
#         axs[action].spines['right'].set_linewidth(3)  
#         axs[action].spines['top'].set_linewidth(3)  
        
#     plt.setp(axs[0], ylabel='Belief State')
#     plt.setp(axs[1], xlabel='Loss:Reward Ratio (logarithmic)')
#     plt.setp(axs2, ylabel='mean no. samples')
#     fig2.tight_layout()
    
#     if (figSave != ""): #...print individual to file

#         # create the full path for the subfolder
#         subPath = os.path.join(figSave, "sample_sequence/")

#         # check if the subfolder already exists
#         if not os.path.exists(subPath):
#         # create the subfolder if it does not exist
#             os.makedirs(subPath)

#         plt.savefig(subPath+'sample_seq_'+str(self.name)+'.pdf')
#         plt.savefig(subPath+'sample_seq'+str(self.name)+'.svg')


    